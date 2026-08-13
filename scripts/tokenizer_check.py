#!/usr/bin/env python3
"""P1 GATE: settle whether zero-padded numeric fields give constant token counts.

This is the one open empirical question in spec v4. Run it on the Mac before
renting anything.

  Single distinct length per model  -> string templates suffice
  Multiple distinct lengths         -> ID-level padding is MANDATORY, not optional

Either way the engine's ID-level padding is correct; this only tells you how
much you are relying on it, and gives you a number to put in the paper instead
of an assumption.

    python scripts/tokenizer_check.py
    python scripts/tokenizer_check.py --models Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# The exact templates the engine renders. Testing bare digits would be
# misleading: BPE merges across the boundary with preceding text.
TEMPLATES = [
    "Your score: {v:>3d}",
    "Opponent score: {v:>3d}",
    "Rounds played: {v:>3d}",
    "Rounds remaining: {v:>3d}",
]

FILLER_CANDIDATES = ["\n", " ", ".", "-"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--max-value", type=int, default=200)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("pip install transformers", file=sys.stderr)
        return 2

    overall_constant = True
    checked: list[str] = []
    skipped: list[tuple[str, str]] = []

    for model in args.models:
        print(f"\n=== {model} ===")
        try:
            tok = AutoTokenizer.from_pretrained(model)
        except Exception as exc:  # noqa: BLE001 - report and continue
            reason = _diagnose(exc)
            print(f"  SKIPPED: {type(exc).__name__}")
            print(f"  FIX: {reason}")
            skipped.append((model, reason))
            continue
        checked.append(model)

        for template in TEMPLATES:
            lengths = {
                len(tok.encode(template.format(v=v), add_special_tokens=False))
                for v in range(args.max_value + 1)
            }
            constant = len(lengths) == 1
            overall_constant &= constant
            flag = "CONSTANT" if constant else "VARIABLE"
            print(f"  {flag:9} {template:28} lengths={sorted(lengths)}")

        singles = [
            repr(c)
            for c in FILLER_CANDIDATES
            if len(tok.encode(c, add_special_tokens=False)) == 1
        ]
        print(f"  single-token filler candidates: {singles or 'NONE — investigate'}")

    print("\n" + "=" * 62)
    print(f"COVERAGE: {len(checked)}/{len(args.models)} models checked")
    for model in checked:
        print(f"  checked  {model}")
    for model, reason in skipped:
        print(f"  SKIPPED  {model}")
        print(f"           -> {reason}")

    if not overall_constant:
        print("\nRESULT: FAIL — at least one tokenizer gives VARIABLE lengths.")
        print("        ID-level padding is MANDATORY. Record this in the methods")
        print("        section as a measured fact.")
        return 1

    if skipped:
        print("\nRESULT: INCOMPLETE — every checked tokenizer is constant, but")
        print(f"        {len(skipped)} model(s) were never checked. This gate is NOT passed.")
        print("        A green verdict on partial coverage is exactly the kind of")
        print("        silently-passing check this project exists to criticise.")
        return 2

    print("\nRESULT: PASS — zero-padding is constant across all target tokenizers.")
    print("        ID-level padding remains the implementation, as a guarantee.")
    return 0


def _diagnose(exc: Exception) -> str:
    """Turn library stack traces into the specific action that unblocks them."""
    text = str(exc)
    if "gated repo" in text or "restricted" in text or "403" in text:
        return (
            "Gated repo. A valid token is NOT sufficient — open the model page in a "
            "browser and click 'Request access', then wait for approval."
        )
    if "protobuf" in text:
        return "pip install protobuf sentencepiece   (SentencePiece tokenizers need both)"
    if "Connection" in text or "resolve" in text:
        return "Network or proxy issue reaching huggingface.co."
    return f"Unhandled: {text[:200]}"


if __name__ == "__main__":
    raise SystemExit(main())
