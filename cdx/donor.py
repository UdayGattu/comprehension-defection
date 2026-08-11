"""Donor selection for Arm 3c (stale state).

WHAT ARM 3c IS
    The treatment template, rendered from a DIFFERENT episode's state. Same
    fields, same shape, same token count - but the numbers belong to someone
    else's game. It isolates the effect of state being WRONG from the effect of
    state being ABSENT (3b) or MEANINGLESS (3d).

    NOT YET WIRED: THE SCAFFOLD-ECHO CHECK
        If a probe answer reproduced the DONOR's numbers rather than the true
        ones, the model would have demonstrably read the block - resolving the
        ambiguity Experiment 1 left open between "did not read it" and "read it
        and could not use it".

        The `scaffold_echo` column exists in the schema and NOTHING WRITES TO
        IT. Wiring it needs the donor's rendered score carried into the row
        writer and compared against the probe answer, which is a schema and
        plumbing change, not a one-liner. Until then Arm 3c delivers the
        BEHAVIOURAL contrast (3c vs 3 = the cost of state being wrong) and not
        the instrument check. Do not claim the latter in a write-up.

WHY THIS IS NOT TRIVIAL
    Episodes advance in lockstep, so every candidate donor sits at the same
    turn as its recipient. Only the score and the last opponent action can
    differ. Early on they usually do not: at turn 0 every episode has score 0
    and no last move, so no distinct donor exists and Arm 3c is momentarily
    identical to Arm 3.

    That degeneracy is unavoidable and must be MEASURED, not hidden. A run
    where most turns fall back to an identical donor has not tested anything,
    and the analysis needs to know that before it interprets the arm.

DETERMINISM
    Selection is driven by a caller-supplied RNG derived from the episode key,
    so a re-run reproduces the same donors. It must never draw from an RNG
    stream that also drives play, or donor selection would shift the actions it
    is supposed to leave untouched.
"""

from __future__ import annotations

import random
from typing import Sequence


def state_fingerprint(state) -> tuple:
    """The fields the treatment template actually renders.

    Two states with the same fingerprint produce byte-identical blocks, so a
    donor sharing one is no manipulation at all.
    """
    last = state.last_opponent_action()
    return (
        state.agent_score,
        state.opponent_score,
        state.turn_index,
        None if last is None else last.value,
    )


def select_donor(
    states: Sequence,
    index: int,
    rng: random.Random,
    max_draws: int = 64,
) -> tuple[object | None, bool]:
    """Choose a donor for ``states[index]``.

    Returns (donor, degenerate). ``degenerate`` is True when no candidate with a
    different fingerprint was found within ``max_draws``; the caller then falls
    back to the recipient's own state and MUST record that the turn provided no
    manipulation.

    Sampling with replacement rather than shuffling: a shuffle is O(n) per
    episode per turn, which at 1,600 episodes x 20 turns is millions of
    operations for no benefit. With max_draws=64 the chance of missing an
    available distinct donor is negligible unless distinct states are very rare
    - and in that case the arm is degenerate anyway, which is what we report.
    """
    n = len(states)
    if n == 0:
        raise ValueError("empty donor pool")
    if n < 2:
        # One live episode: no donor exists. Return the recipient's own state
        # and flag it, rather than None. Returning None would make the caller
        # pass donor=None into build_pair, which raises - killing a cell late in
        # a stochastic run, on rented hardware, for a case that is simply
        # degenerate. Degenerate is a measurement, not an error.
        return states[0], True

    target = state_fingerprint(states[index])
    for _ in range(max_draws):
        j = rng.randrange(n)
        if j == index:
            continue
        if state_fingerprint(states[j]) != target:
            return states[j], False

    # Exhaustive fallback: cheap when n is small, and removes the doubt that a
    # distinct donor existed but random sampling happened to miss it.
    for j, candidate in enumerate(states):
        if j != index and state_fingerprint(candidate) != target:
            return candidate, False

    return states[index], True


class DonorStats:
    """Degeneracy accounting for one cell.

    Report this alongside the arm. A 3c cell that was 80% degenerate did not
    test stale state for most of its turns, and any effect estimate from it is
    diluted by that fraction.
    """

    def __init__(self) -> None:
        self.total = 0
        self.degenerate = 0
        self.by_turn: dict[int, list[int]] = {}

    def record(self, turn: int, degenerate: bool) -> None:
        self.total += 1
        self.degenerate += int(degenerate)
        row = self.by_turn.setdefault(turn, [0, 0])
        row[0] += int(degenerate)
        row[1] += 1

    @property
    def rate(self) -> float:
        return self.degenerate / max(self.total, 1)

    def summary(self) -> dict:
        """Keys are namespaced with ``donor_``.

        The cell result already carries a ``by_turn`` holding comprehension per
        turn. Merging an unnamespaced dict into it would silently overwrite the
        manipulation check with donor statistics - a corruption that would look
        like plausible numbers rather than an error.
        """
        return {
            "donor_degenerate_rate": self.rate,
            "donor_draws": self.total,
            # Keys stringified to match the cell result's other by-turn maps.
            # These dicts round-trip through JSON for cell-level resume, and
            # JSON turns int keys into strings anyway - so an int here would
            # mean the same field had different key types before and after a
            # resume.
            "donor_by_turn": {
                str(t): d / max(n, 1) for t, (d, n) in sorted(self.by_turn.items())
            },
        }
        