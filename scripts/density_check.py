#!/usr/bin/env python3
"""Report cross-template block density under a real model tokeniser.

WHY THIS EXISTS
    `scripts/gpu_run.py` aborts a run when a template's filler fraction diverges
    from the default template's by more than `TEMPLATE_DENSITY_TOLERANCE`. That
    gate prints only the WORST block type - enough to stop a bad run, not enough
    to decide what to do about it. This prints all of them.

    It cannot be usefully run on a laptop. The divergence is a property of the
    BPE vocabulary: under the character-level stub the tests use, `original` and
    `reworded` match to 3.2 points; under Llama-3.1 they differ by 15.2. The
    number that decides anything is only observable where the real tokeniser is.

WHY --arms MATTERS
    Block types are rendered by arms:
        3, 3s, 3m, 3c -> treatment_text
        3b            -> nondiagnostic_text
        3d            -> syntactic_text
        1             -> no block at all
    A divergence in a block type no arm in the run renders cannot reach that
    run's estimand. exp8 runs "1 3b 3 3s 3m" and never renders the syntactic
    placebo. The syntactic body still sets the parity target through
    `_derive_block_tokens`, so it is always REPORTED - but it is only counted
    against the tolerance when an arm actually renders it.

USAGE
    python3 scripts/density_check.py meta-llama/Llama-3.1-8B-Instruct
    python3 scripts/density_check.py Qwen/Qwen2.5-7B-Instruct --arms 1 3b 3 3s 3m
    python3 scripts/density_check.py char          # CPU stub, no download

EXIT
    0  every COUNTED block type within tolerance
    1  at least one over
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cdx.config import Action, Framing, ScaffoldConfig          # noqa: E402
from cdx.scaffold import (                                      # noqa: E402
    DEFAULT_STATE_TEMPLATE,
    STATE_TEMPLATES,
    TEMPLATE_DENSITY_TOLERANCE,
    ScaffoldBuilder,
)

BLOCK_FOR_ARM = {
    "3": "treatment", "3s": "treatment", "3m": "treatment", "3c": "treatment",
    "3b": "nondiag", "3d": "syntactic", "1": None,
}
BLOCKS = ("treatment", "nondiag", "syntactic")
# The short names above are what the arm map and the report use; these are the
# actual builder methods. Kept explicit rather than derived by string surgery,
# because "nondiag" -> "nondiag_text" is wrong and fails only at runtime.
METHOD = {
    "treatment": "treatment_text",
    "nondiag": "nondiagnostic_text",
    "syntactic": "syntactic_text",
}


class CharTokenizer:
    """The stub the test suite uses. One token per character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _State:
    """Duck-types the four attributes the block templates read."""

    def __init__(self, agent, opponent, turn, last):
        self.agent_score = agent
        self.opponent_score = opponent
        self.turn_index = turn
        self.turns = ()
        self._last = last

    def last_opponent_action(self):
        return self._last


# Several states, because treatment density moves with the turn index by a few
# points while the CROSS-TEMPLATE gap does not. Comparing at matched states is
# what makes the gap meaningful.
STATES = (
    _State(0, 0, 0, None),
    _State(24, 18, 8, Action.COOPERATE),
    _State(97, 61, 19, Action.DEFECT),
)


def density(builder, block, state, framing):
    text = getattr(builder, METHOD[block])(state, framing)
    return len(builder.tokenizer.encode(text)) / builder.block_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF model id, or 'char' for the CPU stub")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arms the run renders; restricts what counts against "
                         "the tolerance. Default: count every block type.")
    ap.add_argument("--framing", default=Framing.SEMANTIC.value)
    args = ap.parse_args()

    if args.model == "char":
        tok = CharTokenizer()
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)

    framing = Framing(args.framing)

    counted = set(BLOCKS)
    if args.arms:
        counted = {BLOCK_FOR_ARM[a] for a in args.arms if BLOCK_FOR_ARM.get(a)}

    print("model      " + args.model)
    print("framing    " + framing.value)
    print("tolerance  " + format(TEMPLATE_DENSITY_TOLERANCE, ".0%"))
    print("counted    " + ", ".join(sorted(counted))
          + ("   (from --arms " + " ".join(args.arms) + ")" if args.arms else ""))
    print("")

    builders = {}
    for name in sorted(STATE_TEMPLATES):
        b = ScaffoldBuilder(tok, ScaffoldConfig(), state_template=name)
        builders[name] = b
        print("  " + name.ljust(20) + "target=" + str(b.block_tokens))
        for block in BLOCKS:
            vals = [density(b, block, s, framing) for s in STATES]
            lo, hi = min(vals), max(vals)
            mark = "" if block in counted else "   (not rendered by --arms)"
            print("     " + block.ljust(11)
                  + format(lo, "6.1%") + " - " + format(hi, "6.1%") + mark)
    print("")

    ref = builders[DEFAULT_STATE_TEMPLATE]
    failures = []
    print("  cross-template gaps vs " + DEFAULT_STATE_TEMPLATE + ":")
    for name in sorted(STATE_TEMPLATES):
        if name == DEFAULT_STATE_TEMPLATE:
            continue
        b = builders[name]
        for block in BLOCKS:
            gap = max(abs(density(b, block, s, framing)
                          - density(ref, block, s, framing)) for s in STATES)
            over = gap > TEMPLATE_DENSITY_TOLERANCE
            if block in counted:
                status = "OVER" if over else "ok"
                if over:
                    failures.append(name + "/" + block)
            else:
                status = "OVER (not counted)" if over else "ok (not counted)"
            print("    " + name.ljust(20) + block.ljust(11)
                  + format(gap, "6.1%") + "  " + status)

    print("")
    if failures:
        print("  FAIL  over tolerance in a block type the run renders: "
              + ", ".join(failures))
        return 1
    print("  PASS  every counted block type within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
