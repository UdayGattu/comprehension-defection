#!/usr/bin/env python3
"""Derive ScaffoldConfig.treatment_block_tokens for a real tokenizer.

WHY THIS EXISTS
    Token parity is the foundation of the causal claim: treatment and placebo
    blocks must occupy the same number of tokens at the same position, so the
    only thing that differs between arms is whether the content is
    decision-relevant.

    Character-level padding cannot deliver that. Zero-padding happened to be
    constant for Llama-3.1 and cost 49.7% of treatment score comprehension
    because the model read the leading zero. Space-padding is not constant at
    all - scripts/tokenizer_check.py reports lengths=[5, 6]. BPE, not character
    count, decides how long a rendered field is.

    So blocks are rendered naturally and padded on TOKEN IDS to a fixed target.
    This script finds that target: the longest block any reachable game state
    produces, plus a margin.

RUN ONCE PER (MODEL, FRAMING) BEFORE SPENDING GPU TIME. Write the printed value
into ScaffoldConfig.treatment_block_tokens and commit it. Running with the
default of 0 falls back to legacy behaviour and can raise TokenParityError
mid-sweep, which on rented hardware costs a cell.

    python scripts/calibrate_block.py --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.config import Action, Arm, Framing, ScaffoldConfig
from cdx.scaffold import ScaffoldBuilder, TokenParityError

RULE = "=" * 74

# 20 rounds, temptation payoff 5, so 100 is the ceiling. Sweep past it.
MAX_SCORE = 120
MAX_TURN = 20


class StubState:
    """Minimal GameState surface the scaffold templates touch."""

    def __init__(self, agent: int, opponent: int, turn: int, last) -> None:
        self.agent_score = agent
        self.opponent_score = opponent
        self.turn_index = turn
        self._last = last
        self.turns = ()

    def last_opponent_action(self):
        return self._last


def reachable_states():
    """Every shape a block can take. Exhaustive on the dimensions that matter:
    digit count, turn count, and which action label appears."""
    for turn in range(0, MAX_TURN + 1):
        for score in range(0, MAX_SCORE + 1):
            for last in (None, Action.COOPERATE, Action.DEFECT):
                yield StubState(score, max(score - 3, 0), turn, last)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    from cdx.backends_mlx import NoSpecialTokenizer

    return NoSpecialTokenizer(AutoTokenizer.from_pretrained(model_id))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--framing", default=Framing.SEMANTIC.value)
    ap.add_argument("--margin", type=int, default=2)
    args = ap.parse_args()

    framing = Framing(args.framing)
    tokenizer = load_tokenizer(args.model)
    builder = ScaffoldBuilder(tokenizer, ScaffoldConfig())

    print(f"\n{RULE}\nBLOCK CALIBRATION\n{RULE}")
    print(f"  model    {args.model}")
    print(f"  framing  {framing.value}")
    print(f"  filler   {builder.filler_text!r} -> id {builder._filler_id}")

    states = list(reachable_states())
    per_template: dict[str, tuple[int, int]] = {}
    for name, fn in (
        ("treatment", builder.treatment_text),
        ("nondiagnostic (3b)", builder.nondiagnostic_text),
        ("syntactic (3d)", builder.syntactic_text),
    ):
        lengths = {len(tokenizer.encode(fn(s, framing))) for s in states}
        per_template[name] = (min(lengths), max(lengths))

    print(f"\n  {'template':<22}{'min':>6}{'max':>6}{'varies':>9}")
    print("  " + "-" * 43)
    for name, (lo, hi) in per_template.items():
        print(f"  {name:<22}{lo:>6}{hi:>6}{'yes' if lo != hi else 'no':>9}")

    target = max(hi for _, hi in per_template.values()) + args.margin
    print(f"\n  Variation is EXPECTED and fine — ID-level padding absorbs it.")
    print(f"  What matters is that no block ever exceeds the target.")

    # Arm 3c reuses the treatment template on a DIFFERENT episode's state, so
    # its worst case is the treatment maximum, already covered above.
    print(f"\n{RULE}\nRESULT\n{RULE}")
    print(f"  parity target for this tokeniser = {target}")
    print("\n  This is a REPORT, not an action. Leave")
    print("  ScaffoldConfig.treatment_block_tokens at 0: ScaffoldBuilder derives")
    print("  the same value at construction from whichever tokeniser it is given.")
    print("  Hardcoding it would be correct for this model and silently wrong")
    print("  for the next one, which breaks any multi-model comparison.")
    print("\n  Record the number in the methods section, per model, alongside")
    print("  the filler token — both differ by tokeniser and must be disclosed.")

    # Prove it before anyone trusts it.
    cfg = ScaffoldConfig(treatment_block_tokens=target)
    verifier = ScaffoldBuilder(tokenizer, cfg)
    checked = 0
    for state in states:
        for arm in (Arm.PLACEBO_NONDIAGNOSTIC, Arm.PLACEBO_SYNTACTIC):
            try:
                t, o = verifier.build_pair(arm, state, framing)
            except TokenParityError as exc:
                print(f"\n  VERIFY FAILED at score={state.agent_score} "
                      f"turn={state.turn_index} arm={arm.value}: {exc}")
                return 1
            if not (t.n_tokens == o.n_tokens == target):
                print(f"\n  VERIFY FAILED: {t.n_tokens} vs {o.n_tokens} vs {target}")
                return 1
            checked += 1
    donor = StubState(MAX_SCORE, MAX_SCORE, MAX_TURN, Action.DEFECT)
    for state in states[::17]:
        t, o = verifier.build_pair(Arm.PLACEBO_STALE, state, framing, donor=donor)
        if not (t.n_tokens == o.n_tokens == target):
            print(f"\n  VERIFY FAILED on stale donor")
            return 1
        checked += 1

    print(f"\n  VERIFIED  {checked:,} block pairs, all exactly {target} tokens.")
    print(f"  Commit the config change together with this output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())