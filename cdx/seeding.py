"""Hash-derived seeding.

Every episode's randomness is a pure function of its coordinates. Nothing is
carried between episodes, so execution order is irrelevant and resuming after a
crash cannot alter any trajectory.

This replaces the usual approach of serialising and restoring RNG state, which
is fragile: it requires restoring Python's, NumPy's and torch's generators in
the right order, and any omission silently produces a different run that still
*looks* seeded. Deriving the seed instead eliminates the bug class rather than
managing it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import Arm, OpponentPolicy, ReadoutMode, stable_hash

# 63 bits, not 64: SQLite's INTEGER is a *signed* 64-bit value, so an unsigned
# 64-bit seed overflows on insert. Caught by the test suite rather than in
# production, which is the entire point of running the gates first.
_SEED_BITS = 63
_SEED_MASK = (1 << _SEED_BITS) - 1


@dataclass(frozen=True)
class EpisodeKey:
    """Uniquely identifies one episode. Also the primary key in the database
    and the sole input to seed derivation."""

    run_id: str
    episode_id: int
    arm: Arm
    model_id: str
    readout_mode: ReadoutMode
    opponent: OpponentPolicy

    def coordinate_string(self) -> str:
        return ":".join(
            (
                self.run_id,
                str(self.episode_id),
                self.arm.value,
                self.model_id,
                self.readout_mode.value,
                self.opponent.value,
            )
        )

    def seed(self) -> int:
        """Deterministic 64-bit seed. Identical across processes, machines and
        Python versions."""
        return int(stable_hash(self.coordinate_string())[:16], 16) & _SEED_MASK

    def rng(self) -> random.Random:
        """A generator private to this episode.

        Always use this rather than the module-level `random` functions. Global
        state is shared across episodes and would couple them to execution
        order, defeating the entire point of derived seeding.
        """
        return random.Random(self.seed())


def derive_subseed(key: EpisodeKey, purpose: str) -> int:
    """A seed for a named sub-stream within an episode (e.g. 'donor_selection',
    'stochastic_termination').

    Separate streams prevent a change in one component's consumption pattern
    from shifting every downstream draw, which would otherwise make an unrelated
    code change look like a behavioural effect.
    """
    return int(stable_hash(f"{key.coordinate_string()}#{purpose}")[:16], 16) & _SEED_MASK


def purpose_rng(key: EpisodeKey, purpose: str) -> random.Random:
    return random.Random(derive_subseed(key, purpose))
