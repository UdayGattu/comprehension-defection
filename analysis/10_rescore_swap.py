#!/usr/bin/env python3
"""Re-score the label-swap probes, which were graded in the wrong label space.

THE DEFECT
    All three exp3 swap groups fail the pre-registered CPR gate at 0.200, and
    the per-probe breakdown localises it exactly:

        arm 3, vs allc      CPR    own_score  opp_last  rounds
        exp3_llama_swap    0.200     1.000     0.200     1.000
        exp3_mistral_swap  0.200     1.000     0.200     1.000
        exp3_qwen_swap     0.200     1.000     0.200     1.000

    own_score and rounds_played are perfect. Only opp_last fails, and it fails
    at exactly 0.200 - which is 1 of the 5 probed turns, i.e. turn 0 alone,
    where the answer is "none" and carries no label.

    In the swap condition the action words are inverted, so the model answers
    in the SWAPPED label space while the scorer compares against unswapped
    ground truth. Every non-turn-0 answer is marked wrong. This is a scorer
    bug; the model's comprehension was never in question.

THE DISCIPLINE THIS SCRIPT ENFORCES
    Re-scoring on the strength of that reasoning alone would be assuming the
    conclusion. So the script does two things in order:

      1. Prints the raw got x want contingency table for opp_last, from the
         stored probe text, with NO rescoring applied. If the swap hypothesis
         is right, the off-diagonal is near-complete: every "Cooperate" truth
         drew a "Defect" answer and vice versa.

      2. Only then applies the inversion and recomputes CPR.

    If the contingency is NOT a clean inversion, the rescore is not justified
    and the script says so rather than producing a number. Look at step 1
    before believing step 2.

    Nothing is written back to any database. The originals stay as they ran.

    python analysis/10_rescore_swap.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import pathname2url

RULE = "=" * 78
CPR_GATE = 0.85
SWAP = {"cooperate": "Defect", "defect": "Cooperate"}


def ro_uri(p: Path) -> str:
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def swapped(ans: str | None) -> str | None:
    """Invert an action label; leave anything else (e.g. 'none') untouched."""
    if ans is None:
        return None
    return SWAP.get(ans.strip().lower(), ans)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="SWAP_RESCORE.md")
    ap.add_argument("--glob", default="*swap*.sqlite",
                    help="Databases to re-score. Default catches exp3_*_swap "
                         "and exp2_*_labelswap.")
    args = ap.parse_args()

    dbs = sorted(Path(p) for p in glob.glob(args.glob)
                 if not Path(p).name.startswith(("smoke_", "cotsmoke_")))
    L: list[str] = []

    def w(s=""):
        L.append(s)

    w("# Label-swap probe re-scoring\n")
    w("The swap condition inverts the action words. The scorer compared the "
      "model's answer against **unswapped** ground truth, so every non-turn-0 "
      "`opponent_last` answer was marked wrong and CPR collapsed to 0.200 - "
      "exactly the turn-0-only rate.\n")
    w("**Step 1 is the evidence. Step 2 is the correction. Read them in "
      "order** - if the contingency table is not a clean inversion, the "
      "correction is not justified.\n")
    w("No database is modified.\n")

    if not dbs:
        w(f"\n_No databases matched `{args.glob}`._\n")
        Path(args.out).write_text("\n".join(L), encoding="utf-8")
        print(f"wrote {args.out} (no matching databases)")
        return 0

    for path in dbs:
        con = sqlite3.connect(ro_uri(path), uri=True)
        td = {r[1] for r in con.execute("PRAGMA table_info(turn_details)")}
        if "probe_answers" not in td:
            con.close()
            continue
        rows = con.execute("""
            SELECT arm, opponent_policy, turn, probe_answers
            FROM turn_details WHERE probe_answers IS NOT NULL""").fetchall()
        con.close()
        if not rows:
            continue

        w(f"\n{RULE}\n## `{path.stem}`\n{RULE}\n")

        # ---- step 1: raw contingency, no correction applied -------------
        cont: Counter = Counter()
        for _, _, turn, raw in rows:
            try:
                p = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            o = p.get("opponent_last") or {}
            got, want = o.get("got"), o.get("want")
            if got is None or want is None:
                continue
            cont[(str(want).strip(), str(got).strip())] += 1

        w("### 1. `opponent_last`: what was true vs what the model said\n")
        w("_No rescoring applied. This table is the evidence._\n")
        w("```")
        wants = sorted({k[0] for k in cont})
        gots = sorted({k[1] for k in cont})
        hdr = "want vs got"
        w(f"{hdr:>16}" + "".join(f"{g:>14}" for g in gots))
        for want in wants:
            w(f"{want:>16}" + "".join(f"{cont[(want, g)]:>14,}" for g in gots))
        w("```")

        diag = sum(v for (a, b), v in cont.items() if a.lower() == b.lower())
        inv = sum(v for (a, b), v in cont.items()
                  if a.lower() in SWAP and b.lower() == SWAP[a.lower()].lower())
        tot = sum(cont.values())
        w(f"\nagrees: {diag:,}/{tot:,} ({diag/tot:.3f}) · "
          f"exactly inverted: {inv:,}/{tot:,} ({inv/tot:.3f})\n")

        clean = tot and (inv / tot) > 0.5
        if not clean:
            w("\n**The pattern is NOT a clean inversion.** The swap hypothesis "
              "is not supported for this database and no rescore is reported. "
              "Investigate before assuming a scorer bug.\n")
            continue
        w("\nThe off-diagonal dominates: the model answered in the swapped "
          "label space and was graded against unswapped truth. Rescoring is "
          "justified.\n")

        # ---- step 2: rescore -------------------------------------------
        orig = defaultdict(lambda: [0, 0])
        fixed = defaultdict(lambda: [0, 0])
        for arm, opp, turn, raw in rows:
            try:
                p = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            marks_o, marks_f = [], []
            for kind in ("own_score", "opponent_last", "rounds_played"):
                d = p.get(kind) or {}
                got, want = d.get("got"), d.get("want")
                if got is None or want is None:
                    marks_o.append(0)
                    marks_f.append(0)
                    continue
                marks_o.append(int(str(got).strip() == str(want).strip()))
                g2 = swapped(got) if kind == "opponent_last" else got
                marks_f.append(int(str(g2).strip() == str(want).strip()))
            orig[(arm, opp)][0] += int(all(marks_o))
            orig[(arm, opp)][1] += 1
            fixed[(arm, opp)][0] += int(all(marks_f))
            fixed[(arm, opp)][1] += 1

        w("\n### 2. CPR before and after\n")
        w(f"_All-or-nothing over three probes. Pre-registered gate {CPR_GATE}._\n")
        w("```")
        w(f"{'arm':>6}{'opp':>7}{'n':>9}{'CPR as run':>13}{'CPR rescored':>15}"
          f"{'gate':>8}")
        for key in sorted(orig):
            a, o = key
            co, no = orig[key]
            cf, nf = fixed[key]
            g = "PASS" if (cf / nf) >= CPR_GATE else "fail"
            w(f"{a:>6}{o:>7}{no:>9,}{co/no:>13.3f}{cf/nf:>15.3f}{g:>8}")
        w("```")

    w(f"\n{RULE}\n## What to do with this\n{RULE}\n")
    w("If arm 3 rescores to >= 0.85, the swap groups satisfy the pre-registered "
      "manipulation check and their behavioural results become usable — three "
      "of exp3's nine groups.\n")
    w("Record it in `EXPERIMENTS.md` as a **scorer** defect found after the "
      "fact, alongside the zero-padding and density defects, not as a silent "
      "correction. The behavioural data never changed; only its manipulation "
      "check was graded in the wrong label space.\n")
    w("One consequence to state plainly: `exp3_qwen_swap` was one of only two "
      "SUPPORTED sign-flip verdicts in the whole project, and it was voided "
      "*because* of this failed check. If the rescore passes, that verdict "
      "returns and has to be reported — including the fact that its ATE_true "
      "of +0.456 vs ALLC is an order of magnitude beyond anything in the "
      "semantic data.\n")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())