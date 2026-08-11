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
    ) -> None:
        self.experiment = experiment
        self.backend = backend
        self.tokenizer = tokenizer
        self.store = store
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
                instruction_suffix=_INSTRUCTION[framing],
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
