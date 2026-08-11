"""P1 instrument gates as executable tests.

Every test here corresponds to a blocking gate in spec v4 section 9. If any
fails, P2 is a waste of compute.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.backends import CharTokenizer, DummyBackend
from cdx.config import (
    Action,
    Arm,
    ExperimentConfig,
    Framing,
    GameConfig,
    HorizonMode,
    OpponentPolicy,
    Payoffs,
    ReadoutMode,
    ScaffoldConfig,
)
from cdx.db import Store
from cdx.game import Game, build_opponent, replay
from cdx.optimal import episode_regret, realised_value, solve, summary_table
from cdx.runner import Cell, Runner
from cdx.scaffold import ScaffoldBuilder, TokenParityError
from cdx.seeding import EpisodeKey


# ---------------------------------------------------------------- payoffs

def test_payoff_matrix_validity_is_enforced():
    Payoffs()  # canonical, must not raise
    with pytest.raises(ValueError, match="Not a prisoner's dilemma"):
        Payoffs(temptation=1, reward=3, punishment=2, sucker=0)
    with pytest.raises(ValueError, match="Alternating exploitation"):
        Payoffs(temptation=10, reward=3, punishment=1, sucker=0)


def test_tft_stability_threshold():
    # (T-R)/(T-P) = (5-3)/(5-1) = 0.5. The configured gamma=0.9 clears it.
    assert Payoffs().tft_stability_threshold == pytest.approx(0.5)


def test_stochastic_horizon_rejects_gamma_below_threshold():
    with pytest.raises(ValueError, match="below the TFT-stability threshold"):
        GameConfig(horizon_mode=HorizonMode.STOCHASTIC, continuation_probability=0.3)


# ---------------------------------------------------- optimal play / regret

def test_optimal_play_matches_hand_computed_values():
    """Regression test on the exact DP. These numbers underwrite the entire
    sign-flip prediction; if they drift, the paper's central claim drifts."""
    cfg = GameConfig()
    expected = {
        "tft": (62, 24, 38, "down"),
        "grim": (62, 24, 38, "down"),
        "allc": (100, 100, 0, "up"),
        "alld": (20, 20, 0, "up"),
    }
    for row in summary_table(cfg):
        opt, alld, regret, direction = expected[row["opponent"]]
        assert row["optimal"] == opt, row
        assert row["alld"] == alld, row
        assert row["regret_of_alld"] == regret, row
        assert row["predicted_direction"] == direction, row


def test_optimal_sequence_against_tft_is_cooperate_then_defect_last():
    seq = solve(OpponentPolicy.TFT, GameConfig()).sequence
    assert seq[:-1] == (Action.COOPERATE,) * 19
    assert seq[-1] is Action.DEFECT


def test_sign_flip_directions_are_opposite():
    """The core robustness property: the same intervention is predicted to move
    defection in opposite directions depending on the opponent."""
    cfg = GameConfig()
    tft = solve(OpponentPolicy.TFT, cfg).defection_rate
    allc = solve(OpponentPolicy.ALLC, cfg).defection_rate
    assert tft < 0.5 < allc


def test_regret_is_never_negative():
    cfg = GameConfig()
    for policy in (OpponentPolicy.TFT, OpponentPolicy.GRIM, OpponentPolicy.ALLC, OpponentPolicy.ALLD):
        for actions in (
            [Action.DEFECT] * cfg.horizon,
            [Action.COOPERATE] * cfg.horizon,
            [Action.COOPERATE, Action.DEFECT] * (cfg.horizon // 2),
        ):
            assert episode_regret(policy, cfg, actions) >= 0


def test_dp_agrees_with_live_engine():
    """optimal.py reimplements the opponents independently of game.py. This test
    is what makes that duplication safe rather than a liability."""
    cfg = GameConfig()
    key = EpisodeKey("t", 0, Arm.BASELINE, "m", ReadoutMode.LOGIT, OpponentPolicy.TFT)
    for policy in (OpponentPolicy.TFT, OpponentPolicy.GRIM, OpponentPolicy.ALLC, OpponentPolicy.ALLD):
        seq = solve(policy, cfg).sequence
        state = replay(cfg, build_opponent(policy, key), seq)
        assert state.agent_score == realised_value(policy, cfg, seq)
        assert state.agent_score == solve(policy, cfg).value


# ----------------------------------------------------------------- seeding

def test_seeds_are_stable_across_processes():
    """Python's builtin hash() is salted per process. If seeding ever regresses
    to it, this catches it."""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from cdx.seeding import EpisodeKey;"
        "from cdx.config import Arm, ReadoutMode, OpponentPolicy;"
        "print(EpisodeKey('r', 7, Arm.TREATMENT, 'm', ReadoutMode.LOGIT, OpponentPolicy.TFT).seed())"
    ) % str(Path(__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    expected = EpisodeKey("r", 7, Arm.TREATMENT, "m", ReadoutMode.LOGIT, OpponentPolicy.TFT).seed()
    assert int(out.stdout.strip()) == expected


def test_seed_depends_on_every_coordinate():
    base = EpisodeKey("r", 1, Arm.TREATMENT, "m", ReadoutMode.LOGIT, OpponentPolicy.TFT)
    variants = [
        EpisodeKey("r2", 1, Arm.TREATMENT, "m", ReadoutMode.LOGIT, OpponentPolicy.TFT),
        EpisodeKey("r", 2, Arm.TREATMENT, "m", ReadoutMode.LOGIT, OpponentPolicy.TFT),
        EpisodeKey("r", 1, Arm.BASELINE, "m", ReadoutMode.LOGIT, OpponentPolicy.TFT),
        EpisodeKey("r", 1, Arm.TREATMENT, "m2", ReadoutMode.LOGIT, OpponentPolicy.TFT),
        EpisodeKey("r", 1, Arm.TREATMENT, "m", ReadoutMode.SCRATCHPAD, OpponentPolicy.TFT),
        EpisodeKey("r", 1, Arm.TREATMENT, "m", ReadoutMode.LOGIT, OpponentPolicy.ALLC),
    ]
    seeds = {base.seed()} | {v.seed() for v in variants}
    assert len(seeds) == len(variants) + 1


# ---------------------------------------------------------------- scaffold

def _builder() -> ScaffoldBuilder:
    return ScaffoldBuilder(CharTokenizer(), ScaffoldConfig())


def test_token_parity_is_exact_for_every_placebo_arm():
    builder = _builder()
    cfg = GameConfig()
    key = EpisodeKey("t", 0, Arm.TREATMENT, "m", ReadoutMode.LOGIT, OpponentPolicy.TFT)
    opponent = build_opponent(OpponentPolicy.TFT, key)
    game = Game(cfg, opponent, key)
    donor = replay(cfg, build_opponent(OpponentPolicy.TFT, key), [Action.DEFECT] * 5)

    for turn in range(6):
        for arm in (Arm.PLACEBO_NONDIAGNOSTIC, Arm.PLACEBO_STALE, Arm.PLACEBO_SYNTACTIC):
            treatment, placebo = builder.build_pair(arm, game.state, Framing.SEMANTIC, donor=donor)
            assert treatment.n_tokens == placebo.n_tokens, (arm, turn)
        game.step(Action.COOPERATE if turn % 2 else Action.DEFECT)


def test_parity_failure_raises_rather_than_warns():
    """A parity violation invalidates the causal estimate. It must halt the run."""
    builder = _builder()
    with pytest.raises(ValueError, match="does not inject"):
        builder.build_pair(Arm.BASELINE, Game(GameConfig(), build_opponent(
            OpponentPolicy.TFT,
            EpisodeKey("t", 0, Arm.BASELINE, "m", ReadoutMode.LOGIT, OpponentPolicy.TFT),
        ), EpisodeKey("t", 0, Arm.BASELINE, "m", ReadoutMode.LOGIT, OpponentPolicy.TFT)).state,
            Framing.SEMANTIC)


def test_no_single_token_filler_available_is_fatal():
    """Measured: Mistral-7B-Instruct-v0.3 has no single-token newline while
    Qwen2.5 does. The builder must fall through candidates and halt loudly if
    none is single-token, rather than padding with a multi-token filler and
    overshooting the target."""
    with pytest.raises(TokenParityError, match="No single-token filler"):
        ScaffoldBuilder(CharTokenizer(), ScaffoldConfig(filler_candidates=("  ", "abc")))


def test_filler_falls_through_to_first_single_token_candidate():
    builder = ScaffoldBuilder(
        CharTokenizer(), ScaffoldConfig(filler_candidates=("multi", "\n", " "))
    )
    assert builder.filler_text == "\n"


def test_undisclosed_horizon_is_not_leaked_into_the_prompt():
    """Leaking the horizon changes the equilibrium and silently invalidates the
    cell."""
    from cdx.scaffold import PromptAssembler

    tok = CharTokenizer()
    assembler = PromptAssembler(tok, ScaffoldConfig())
    cfg = GameConfig(horizon_mode=HorizonMode.UNDISCLOSED)
    key = EpisodeKey("t", 0, Arm.BASELINE, "m", ReadoutMode.LOGIT, OpponentPolicy.TFT)
    ids = assembler.assemble(
        game_config=cfg,
        state=Game(cfg, build_opponent(OpponentPolicy.TFT, key), key).state,
        framing=Framing.SEMANTIC,
        block=None,
        instruction_suffix="",
    )
    text = tok.decode(ids)
    assert "lasts exactly" not in text
    assert "20 rounds" not in text


# ------------------------------------------------------------- determinism

def _run(tmp_path: Path, run_id: str, n: int) -> list[tuple]:
    store = Store(tmp_path / f"{run_id}.db")
    experiment = ExperimentConfig(run_id=run_id)
    runner = Runner(experiment, DummyBackend(), CharTokenizer(), store)
    runner.seed_donor_pool()
    runner.run_cells(
        [Cell(Arm.TREATMENT, OpponentPolicy.TFT, ReadoutMode.LOGIT, Framing.SEMANTIC, n)],
        model_id="dummy/deterministic",
    )
    rows = store._conn.execute(
        "SELECT episode_id, turn, agent_action FROM turns ORDER BY episode_id, turn"
    ).fetchall()
    store.close()
    return rows


def test_engine_is_bit_identical_across_runs(tmp_path):
    assert _run(tmp_path / "a", "det", 8) == _run(tmp_path / "b", "det", 8)


def test_resume_reproduces_uninterrupted_run(tmp_path):
    """Simulates the SIGKILL resume gate: a partial run followed by a resumed
    run must equal an uninterrupted run."""
    full = _run(tmp_path / "full", "resume", 10)

    partial_dir = tmp_path / "partial"
    _run(partial_dir, "resume", 4)          # first pass, dies after 4
    resumed = _run(partial_dir, "resume", 10)  # resume to 10

    assert resumed == full


def test_resume_skips_completed_episodes(tmp_path):
    store = Store(tmp_path / "s.db")
    experiment = ExperimentConfig(run_id="skip")
    runner = Runner(experiment, DummyBackend(), CharTokenizer(), store)
    runner.seed_donor_pool()
    cells = [Cell(Arm.BASELINE, OpponentPolicy.ALLC, ReadoutMode.LOGIT, Framing.ABSTRACT, 5)]

    first = runner.run_cells(cells, model_id="dummy/deterministic")
    second = runner.run_cells(cells, model_id="dummy/deterministic")
    store.close()

    assert first == {"run": 5, "skipped": 0}
    assert second == {"run": 0, "skipped": 5}


def test_action_surface_forms_match_what_the_prompt_asks_for():
    """REGRESSION. The first live pilot ran semantic framing while the backend
    measured probability mass on the abstract X/Y tokens. Result: off-task rate
    1.0000, every action decided by the tie-break, and a GO verdict on garbage.

    Whatever string the prompt tells the model to emit must appear in that
    framing's surface forms.
    """
    from cdx.backends import ACTION_SURFACE_FORMS_BY_FRAMING
    from cdx.scaffold import render_action

    assert set(ACTION_SURFACE_FORMS_BY_FRAMING) == set(Framing)
    for framing, by_action in ACTION_SURFACE_FORMS_BY_FRAMING.items():
        assert set(by_action) == {Action.COOPERATE, Action.DEFECT}
        for action, forms in by_action.items():
            expected = render_action(action, framing)
            assert any(f.strip() == expected for f in forms), (framing, action, forms)


def test_stochastic_horizon_respects_hard_cap(tmp_path):
    cfg = GameConfig(horizon_mode=HorizonMode.STOCHASTIC, continuation_probability=0.99)
    key = EpisodeKey("t", 0, Arm.BASELINE, "m", ReadoutMode.LOGIT, OpponentPolicy.ALLC)
    game = Game(cfg, build_opponent(OpponentPolicy.ALLC, key), key)
    steps = 0
    while game.should_continue() and steps < 1000:
        game.step(Action.COOPERATE)
        steps += 1
    assert steps <= cfg.horizon
