"""Episode runner with crash-safe resume.

Resume works by skipping episode coordinates already durably present in the
database. Because seeds are derived from those coordinates, a replacement
episode is byte-identical to the one that would have run in an uninterrupted
pass — so resuming cannot alter the experiment. This is what makes cheap
preemptible instances safe to use.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

from .backends import Decision, LLMBackend
from .config import (
    Action,
    Arm,
    ExperimentConfig,
    Framing,
    OpponentPolicy,
    ReadoutMode,
)
from .db import EpisodeRecord, Store, TurnRecord, encode_top_tokens
from .game import Game, GameState, build_opponent, replay
from .optimal import episode_regret, solve
from .scaffold import PromptAssembler, ScaffoldBuilder, Tokenizer
from .seeding import EpisodeKey, purpose_rng

logger = logging.getLogger(__name__)

_INSTRUCTION = {
    Framing.ABSTRACT: "\nRespond with exactly one character: X or Y.\n",
    Framing.SEMANTIC: "\nRespond with exactly one word: Cooperate or Defect.\n",
}

# SCRATCHPAD needs its own instruction, and this is not optional.
#
# The LOGIT instruction says "respond with exactly one word". Asking a model
# that has been told to answer in one word to first generate 128 tokens of
# reasoning produces exactly what you would expect: it emits "Cooperate" and
# stops. Observed in the first smoke run - every scratchpad was the bare answer,
# zero of them longer than 20 characters, and the readout degenerated into an
# expensive re-reading of a decision the model had already made.
#
# So the instruction is a function of READOUT, not just framing. The two
# conditions therefore do not share a prompt, which is unavoidable - you cannot
# demand one word and step-by-step reasoning at once - and must be stated in the
# methods rather than glossed.
#
# What is preserved: this suffix is IDENTICAL ACROSS ARMS within a readout, so
# the treatment/placebo contrast is unaffected. Only the readout factor moves.
#
# IT MUST NOT SPECIFY AN OUTPUT FORMAT.
#     A first attempt ended "...then give your choice as exactly one word:
#     Cooperate or Defect." Llama-3.1 honoured both clauses and produced ~520
#     characters of reasoning. Qwen2.5 latched onto the last, most specific
#     instruction and emitted the bare word: min = avg = max = 9 characters
#     across every turn of every cell.
#
#     The format clause is unnecessary anyway. The action is read from the
#     logit position after "Final answer:", never parsed from this text, and
#     the rules section above already names the two options. So the instruction
#     asks for reasoning and says nothing about how to answer.
#
# Both framings are identical here on purpose - no action labels appear, so the
# instruction cannot leak a lexical cue into the abstract condition.
_INSTRUCTION_SCRATCHPAD = {
    Framing.ABSTRACT: (
        "\nBefore choosing, reason step by step about the current state, the "
        "other player's behaviour so far, and how many rounds remain.\n"
    ),
    Framing.SEMANTIC: (
        "\nBefore choosing, reason step by step about the current state, the "
        "opponent's behaviour so far, and how many rounds remain.\n"
    ),
}


# THE HORIZON CONFOUND, AND THE ABLATION THAT MEASURES IT.
#
# _INSTRUCTION_SCRATCHPAD above names three things: the current state, the
# opponent's behaviour, and HOW MANY ROUNDS REMAIN. That last clause is a
# mistake with a name. Finite horizons induce backward induction, and backward
# induction is the textbook argument for defecting from round one. exp4 found
# Llama's defection rate rising ~6x from LOGIT to SCRATCHPAD (0.102 -> 0.579 in
# arm 3b) and the lexical container effect collapsing from -0.22 to +0.02.
#
# Neither of those can be attributed to reasoning, because the instruction that
# accompanies reasoning also hands the model a defection heuristic AND directs
# its attention at the state block. Two rival explanations, same data:
#
#   "CoT shatters the cooperative prior"  vs  "naming the horizon does"
#   "CoT kills the placebo effect"        vs  "'reason about the current
#                                              state' does, by replacing
#                                              passive priming with active
#                                              attention"
#
# The cross-readout comparison cannot separate them. The WITHIN-readout
# contrasts are untouched - the confound is applied identically to every arm -
# so exp4's ATE_true and its two perturbation figures all stand. Only the
# LOGIT-vs-SCRATCHPAD claim is unsafe.
#
# MINIMAL asks for reasoning and names NOTHING: no state, no opponent, no
# horizon, no action, no output format. Running it against GUIDED separates the
# effect of reasoning from the effect of what the instruction points at.
#
# Because it names nothing, it is byte-identical across framings - unlike
# GUIDED, which must say "opponent" or "other player". That is a second, quieter
# improvement: under MINIMAL the two framings differ ONLY in the scaffold, so
# the instruction cannot contribute any part of the semantic/abstract gap.
_INSTRUCTION_SCRATCHPAD_MINIMAL = {
    Framing.ABSTRACT: "\nBefore choosing, think step by step.\n",
    Framing.SEMANTIC: "\nBefore choosing, think step by step.\n",
}

# GUIDED is the default so that re-running exp4 from a later commit reproduces
# exp4. A variant must be requested explicitly, and --scratchpad-prompt lands in
# run_meta.config_json, so which prompt produced which database is recoverable
# from the artefact alone.
SCRATCHPAD_PROMPTS = {
    "guided": _INSTRUCTION_SCRATCHPAD,
    "minimal": _INSTRUCTION_SCRATCHPAD_MINIMAL,
}
DEFAULT_SCRATCHPAD_PROMPT = "guided"


def instruction_for(
    framing: Framing,
    readout: ReadoutMode,
    scratchpad_prompt: str = DEFAULT_SCRATCHPAD_PROMPT,
) -> str:
    """The instruction line, which depends on how the action will be read.

    Takes no arm argument, and must never take one: an arm-dependent suffix
    would confound every contrast in the study.
    """
    if readout is not ReadoutMode.SCRATCHPAD:
        return _INSTRUCTION[framing]
    try:
        table = SCRATCHPAD_PROMPTS[scratchpad_prompt]
    except KeyError:
        raise ValueError(
            f"unknown scratchpad prompt {scratchpad_prompt!r}; "
            f"expected one of {sorted(SCRATCHPAD_PROMPTS)}"
        ) from None
    return table[framing]


@dataclass(frozen=True)
class Cell:
    """One experimental cell. The cross-product of cells defines the run."""

    arm: Arm
    opponent: OpponentPolicy
    readout_mode: ReadoutMode
    framing: Framing
    n_episodes: int


@dataclass
class EpisodeResult:
    key: EpisodeKey
    state: GameState
    decisions: list[Decision]
    regret: int | None
    prompt_hash: str


class Runner:
    def __init__(
        self,
        experiment: ExperimentConfig,
        backend: LLMBackend,
        tokenizer: Tokenizer,
        store: Store,
        scratchpad_prompt: str = DEFAULT_SCRATCHPAD_PROMPT,
    ) -> None:
        if scratchpad_prompt not in SCRATCHPAD_PROMPTS:
            raise ValueError(
                f"unknown scratchpad prompt {scratchpad_prompt!r}; "
                f"expected one of {sorted(SCRATCHPAD_PROMPTS)}"
            )
        self.experiment = experiment
        self.backend = backend
        self.tokenizer = tokenizer
        self.store = store
        self.scratchpad_prompt = scratchpad_prompt
        self.builder = ScaffoldBuilder(tokenizer, experiment.scaffold)
        self.assembler = PromptAssembler(tokenizer, experiment.scaffold)
        self._donor_pool: list[GameState] = []

    # ---- donor pool ------------------------------------------------------

    def seed_donor_pool(self, size: int = 32) -> None:
        """Pre-generate plausible game states for the stale-state placebo.

        Generated from scripted self-play rather than from live episodes, so the
        pool is fixed before any treatment data exists and cannot leak
        information between arms.
        """
        game_config = self.experiment.game
        for i in range(size):
            key = EpisodeKey(
                run_id=f"{self.experiment.run_id}-donor",
                episode_id=i,
                arm=Arm.BASELINE,
                model_id="donor",
                readout_mode=ReadoutMode.LOGIT,
                opponent=OpponentPolicy.TFT,
            )
            rng = purpose_rng(key, "donor_actions")
            actions = [
                Action.DEFECT if rng.random() < 0.5 else Action.COOPERATE
                for _ in range(rng.randint(1, game_config.horizon - 1))
            ]
            self._donor_pool.append(
                replay(game_config, build_opponent(OpponentPolicy.TFT, key), actions)
            )

    def _draw_donor(self, key: EpisodeKey, exclude_turn_index: int) -> GameState:
        if not self._donor_pool:
            self.seed_donor_pool()
        rng = purpose_rng(key, "donor_selection")
        # Prefer a donor whose turn index differs, so the stale block is
        # unambiguously not the current state.
        candidates = [d for d in self._donor_pool if d.turn_index != exclude_turn_index]
        return rng.choice(candidates or self._donor_pool)

    # ---- single episode --------------------------------------------------

    def run_episode(self, key: EpisodeKey, framing: Framing) -> EpisodeResult:
        game_config = self.experiment.game
        opponent = build_opponent(key.opponent, key)
        game = Game(game_config, opponent, key)
        decisions: list[Decision] = []
        prompt_digest = hashlib.sha256()

        while game.should_continue():
            block = None
            if key.arm.injects_block:
                donor = (
                    self._draw_donor(key, game.state.turn_index)
                    if key.arm is Arm.PLACEBO_STALE
                    else None
                )
                _, block = self.builder.build_pair(
                    key.arm, game.state, framing, donor=donor
                )

            prompt_ids = self.assembler.assemble(
                game_config=game_config,
                state=game.state,
                framing=framing,
                block=block,
                instruction_suffix=instruction_for(
                    framing, key.readout_mode, self.scratchpad_prompt
                ),
            )
            prompt_digest.update(str(prompt_ids).encode("utf-8"))

            turn_seed = purpose_rng(key, f"turn{game.state.turn_index}").getrandbits(63)
            decision = self.backend.decide(
                prompt_ids, key.readout_mode, turn_seed, framing=framing
            )
            decisions.append(decision)

            if decision.is_off_task:
                logger.warning(
                    "off-task decision: episode=%s turn=%d action_mass=%.4f",
                    key.episode_id,
                    game.state.turn_index,
                    decision.action_mass_total,
                )

            game.step(decision.action)

        regret = None
        if key.opponent.is_scripted and key.opponent is not OpponentPolicy.QTABLE:
            try:
                regret = episode_regret(
                    key.opponent, game_config, game.state.agent_history
                )
            except ValueError:
                regret = None

        return EpisodeResult(
            key=key,
            state=game.state,
            decisions=decisions,
            regret=regret,
            prompt_hash=prompt_digest.hexdigest()[:32],
        )

    # ---- persistence -----------------------------------------------------

    def persist(self, result: EpisodeResult, framing: Framing) -> None:
        state, key = result.state, result.key
        game_config = self.experiment.game

        optimal_seq: Sequence[Action] | None = None
        if key.opponent.is_scripted and key.opponent is not OpponentPolicy.QTABLE:
            try:
                optimal_seq = solve(key.opponent, game_config).sequence
            except ValueError:
                optimal_seq = None

        turns: list[TurnRecord] = []
        for i, (turn, decision) in enumerate(zip(state.turns, result.decisions)):
            turns.append(
                TurnRecord(
                    turn=turn.index,
                    agent_action=turn.agent_action,
                    opponent_action=turn.opponent_action,
                    agent_payoff=turn.agent_payoff,
                    optimal_action=optimal_seq[i] if optimal_seq and i < len(optimal_seq) else None,
                    turn_regret=None,
                    logit_mass_c=decision.logit_mass_cooperate,
                    logit_mass_d=decision.logit_mass_defect,
                    action_mass_total=decision.action_mass_total,
                    logit_gap=decision.logit_gap,
                    scaffold_tokens=None,
                    scaffold_pad=None,
                    top_tokens=encode_top_tokens(decision.top_tokens) if decision.top_tokens else None,
                    scratchpad=decision.scratchpad,
                )
            )

        record = EpisodeRecord(
            key=key,
            model_revision=self.backend.model_config.revision,
            framing=framing,
            horizon_mode=game_config.horizon_mode,
            horizon=game_config.horizon,
            temperature=self.backend.model_config.temperature,
            config_fingerprint=self.experiment.fingerprint(),
            prompt_hash=result.prompt_hash,
            n_turns=len(state.turns),
            agent_score=state.agent_score,
            opponent_score=state.opponent_score,
            defection_count=sum(a is Action.DEFECT for a in state.agent_history),
            episode_regret=result.regret,
        )
        self.store.write_episode(
            record, turns, datetime.now(timezone.utc).isoformat(timespec="seconds")
        )

    # ---- driver ----------------------------------------------------------

    def run_cells(self, cells: Iterable[Cell], model_id: str) -> dict[str, int]:
        """Run every episode in every cell, skipping already-completed work.

        Returns counts so a resumed run reports what it skipped rather than
        silently doing nothing.
        """
        completed = self.store.completed_keys(self.experiment.run_id)
        stats = {"run": 0, "skipped": 0}

        for cell in cells:
            for episode_id in range(cell.n_episodes):
                key = EpisodeKey(
                    run_id=self.experiment.run_id,
                    episode_id=episode_id,
                    arm=cell.arm,
                    model_id=model_id,
                    readout_mode=cell.readout_mode,
                    opponent=cell.opponent,
                )
                coords = (
                    episode_id,
                    cell.arm.value,
                    model_id,
                    cell.readout_mode.value,
                    cell.opponent.value,
                )
                if coords in completed:
                    stats["skipped"] += 1
                    continue
                result = self.run_episode(key, cell.framing)
                self.persist(result, cell.framing)
                stats["run"] += 1

        logger.info("run=%d skipped=%d", stats["run"], stats["skipped"])
        return stats