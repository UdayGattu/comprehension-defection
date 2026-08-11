"""Tests for the pre-registered probe and analysis code.

These verify the statistics are correct BEFORE any real data exists, which is
the only time such verification is credible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.analysis import (
    ProportionDiff,
    compute_ate,
    compute_sign_flip,
    min_detectable_effect,
    power_audit,
    required_n,
)
from cdx.config import Action, Arm, Framing, GameConfig, OpponentPolicy
from cdx.game import build_opponent, replay
from cdx.probe import (
    PROBE_SUITE,
    PROBE_SUITE_HASH,
    ProbeKind,
    ProbeMethod,
    ProbeResult,
    check_stale_echo,
    normalise,
    score_answer,
)
from cdx.seeding import EpisodeKey


# ------------------------------------------------------------------- power

def test_required_n_matches_the_spec_figure():
    """N=1600 in the spec comes from here. If this drifts, the spec is wrong."""
    assert 1500 <= required_n(0.5, 0.05) <= 1600


def test_n30_cannot_resolve_a_small_effect():
    """The concrete reason the spec rejects N=30: the minimum detectable effect
    is ~36 pp, so a null result there would be meaningless."""
    assert min_detectable_effect(30) > 0.30
    assert min_detectable_effect(1600) < 0.06


def test_mde_shrinks_with_n():
    values = [min_detectable_effect(n) for n in (30, 100, 400, 1600)]
    assert values == sorted(values, reverse=True)


# -------------------------------------------------------------- statistics

def test_identical_proportions_are_not_significant():
    d = ProportionDiff(0.5, 0.5, 1000, 1000)
    assert d.diff == 0
    assert not d.significant
    assert d.p_value > 0.9


def test_large_clear_difference_is_significant():
    d = ProportionDiff(0.9, 0.5, 1000, 1000)
    assert d.significant
    assert d.p_value < 1e-6
    lo, hi = d.ci95
    assert lo > 0


def test_small_sample_hides_a_real_difference():
    """Same effect size, two sample sizes: only the large one is detectable.
    This is the underpowered-null failure mode made concrete."""
    small = ProportionDiff(0.60, 0.50, 30, 30)
    large = ProportionDiff(0.60, 0.50, 1600, 1600)
    assert not small.significant
    assert large.significant


def test_power_audit_flags_underpowered_cells():
    from cdx.analysis import CellStats

    cells = [
        CellStats("m", "3", "tft", "logit", 600, 30, 0.5, None, None, 0.0, 0.0),
        CellStats("m", "3b", "tft", "logit", 32000, 1600, 0.5, None, None, 0.0, 0.0),
    ]
    warnings = power_audit(cells)
    assert len(warnings) == 1
    assert "n=30" in warnings[0]


# ------------------------------------------------------------------ probes

def test_probe_hash_is_stable():
    from cdx.probe import probe_suite_hash

    assert probe_suite_hash() == PROBE_SUITE_HASH
    assert len(PROBE_SUITE_HASH) == 64


def test_probe_suite_covers_score_action_and_rounds():
    assert {s.kind for s in PROBE_SUITE} == {
        ProbeKind.OWN_SCORE,
        ProbeKind.OPPONENT_LAST,
        ProbeKind.ROUNDS_PLAYED,
    }


def _state(actions: list[Action]):
    key = EpisodeKey("t", 0, Arm.TREATMENT, "m", "logit", OpponentPolicy.TFT)  # type: ignore[arg-type]
    return replay(GameConfig(), build_opponent(OpponentPolicy.TFT, key), actions)


def test_scoring_is_exact_not_approximate():
    """A near-miss must score zero. Partial credit would let a confused model
    look comprehending."""
    state = _state([Action.COOPERATE] * 4)
    spec = next(s for s in PROBE_SUITE if s.kind is ProbeKind.OWN_SCORE)
    true = spec.truth(state, Framing.SEMANTIC)
    assert score_answer(spec, true, state, Framing.SEMANTIC) == 1
    assert score_answer(spec, str(int(true) + 1), state, Framing.SEMANTIC) == 0


def test_scoring_tolerates_surrounding_prose():
    state = _state([Action.COOPERATE] * 4)
    spec = next(s for s in PROBE_SUITE if s.kind is ProbeKind.OWN_SCORE)
    true = spec.truth(state, Framing.SEMANTIC)
    assert score_answer(spec, f"I think it is {true}.", state, Framing.SEMANTIC) == 1


def test_cpr_requires_all_components():
    partial = ProbeResult(ProbeMethod.REPLAY, {
        ProbeKind.OWN_SCORE: 1,
        ProbeKind.OPPONENT_LAST: 0,
        ProbeKind.ROUNDS_PLAYED: 1,
    })
    full = ProbeResult(ProbeMethod.REPLAY, {k: 1 for k in ProbeKind})
    assert partial.cpr == 0
    assert full.cpr == 1


def test_stale_echo_detects_donor_values():
    """Manipulation check: an answer matching the donor rather than the truth
    proves the model read the injected block."""
    true_state = _state([Action.COOPERATE] * 4)
    donor = _state([Action.DEFECT] * 2)
    spec = next(s for s in PROBE_SUITE if s.kind is ProbeKind.OWN_SCORE)

    donor_answer = spec.truth(donor, Framing.SEMANTIC)
    true_answer = spec.truth(true_state, Framing.SEMANTIC)
    assert donor_answer != true_answer

    assert check_stale_echo(spec, donor_answer, true_state, donor, Framing.SEMANTIC)
    assert not check_stale_echo(spec, true_answer, true_state, donor, Framing.SEMANTIC)


def test_normalise_handles_actions_and_numbers():
    assert normalise("  Defect. ") == "defect"
    assert normalise("The score is 14") == "14"
    assert normalise("none") == "none"


def test_normalise_ignores_runaway_generation():
    """REGRESSION. Observed live: the model answered 'none' correctly, then kept
    generating a new conversation turn containing the word 'cooperating'.
    Scanning the whole string scored a correct answer as wrong, deflating CPR."""
    assert normalise("none.<|endoftext|>Human: Are you cooperating?") == "none"
    assert normalise("0\n\nThe interaction has not yet occurred") == "0"
    assert normalise("4\n\n[HISTORY]\nRound 1: you=Cooperate") == "4"
    assert normalise("Cooperate. The interaction has been going") == "cooperate"


def test_action_answers_are_not_misread_as_numbers():
    """REGRESSION. Observed live: identical answers scored OK at turn 4 and WRONG
    at turn 8, because the trailing prose at turn 8 contained a digit and the
    parser looked for numbers before action words."""
    from cdx.probe import AnswerType

    polluted = "Cooperate. The interaction has been going for 8 rounds"
    assert normalise(polluted, AnswerType.ACTION) == "cooperate"
    assert normalise(polluted) == "8"  # the old, wrong behaviour, made explicit


def test_number_answers_ignore_action_words():
    from cdx.probe import AnswerType

    assert normalise("30 after cooperating throughout", AnswerType.NUMBER) == "30"


def test_every_probe_declares_its_answer_type():
    from cdx.probe import AnswerType

    for spec in PROBE_SUITE:
        assert isinstance(spec.answer_type, AnswerType), spec.kind


def test_first_segment_cuts_at_prompt_leakage():
    from cdx.probe import first_segment

    assert first_segment("12\nYou have cooperated in all four") == "12"
    assert first_segment("none <|im_end|>") == "none"
    assert first_segment("6 To calculate the total") == "6 To calculate the total"


# --------------------------------------------------------------- sign flip

def _cell(model, arm, opponent, rate, n=1600):
    from cdx.analysis import CellStats

    return CellStats(model, arm, opponent, "logit", n * 20, n, rate, None, None, 0.0, 0.0)


def test_sign_flip_holds_when_directions_are_opposite_and_predicted():
    cells = [
        _cell("m", Arm.TREATMENT.value, "tft", 0.20),
        _cell("m", Arm.PLACEBO_NONDIAGNOSTIC.value, "tft", 0.80),
        _cell("m", Arm.TREATMENT.value, "allc", 0.95),
        _cell("m", Arm.PLACEBO_NONDIAGNOSTIC.value, "allc", 0.50),
    ]
    result = compute_sign_flip(compute_ate(cells, GameConfig()))[0]
    assert result.holds
    assert "HOLDS" in result.verdict


def test_sign_flip_fails_when_both_move_the_same_way():
    """Both dropping is what a generic prompt artifact looks like. The criterion
    must reject it."""
    cells = [
        _cell("m", Arm.TREATMENT.value, "tft", 0.20),
        _cell("m", Arm.PLACEBO_NONDIAGNOSTIC.value, "tft", 0.80),
        _cell("m", Arm.TREATMENT.value, "allc", 0.20),
        _cell("m", Arm.PLACEBO_NONDIAGNOSTIC.value, "allc", 0.80),
    ]
    result = compute_sign_flip(compute_ate(cells, GameConfig()))[0]
    assert not result.holds
    assert "FAILS" in result.verdict


def test_sign_flip_reports_underpowered_rather_than_failure():
    cells = [
        _cell("m", Arm.TREATMENT.value, "tft", 0.48, n=30),
        _cell("m", Arm.PLACEBO_NONDIAGNOSTIC.value, "tft", 0.52, n=30),
        _cell("m", Arm.TREATMENT.value, "allc", 0.52, n=30),
        _cell("m", Arm.PLACEBO_NONDIAGNOSTIC.value, "allc", 0.48, n=30),
    ]
    result = compute_sign_flip(compute_ate(cells, GameConfig()))[0]
    assert not result.holds
    assert "UNDERPOWERED" in result.verdict


def test_ate_defaults_to_placebo_not_baseline():
    """Comparing against Arm 1 confounds the estimate with prompt perturbation.
    The default contrast must be the matched placebo."""
    cells = [
        _cell("m", Arm.BASELINE.value, "tft", 0.90),
        _cell("m", Arm.TREATMENT.value, "tft", 0.20),
        _cell("m", Arm.PLACEBO_NONDIAGNOSTIC.value, "tft", 0.80),
    ]
    ate = compute_ate(cells, GameConfig())[0]
    assert ate.contrast == "3-vs-3b"
    assert ate.stats.diff == pytest.approx(-0.60)
