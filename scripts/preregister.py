#!/usr/bin/env python3
"""Freeze the probe wording and analysis plan, and emit PREREGISTRATION.md.

Run once, before the first probe pass. Commit the output. The hash it produces is
written onto every database row, so any later edit to a probe question is
detectable from the data alone rather than resting on anyone's word.

    python scripts/preregister.py > PREREGISTRATION.md
    git add PREREGISTRATION.md && git commit -m "pre-register probes and analysis"
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.analysis import ALPHA, POWER, min_detectable_effect, required_n
from cdx.config import GameConfig
from cdx.optimal import summary_table
from cdx.probe import PROBE_SUITE, PROBE_SUITE_HASH


def main() -> int:
    cfg = GameConfig()
    out: list[str] = []
    a = out.append

    a("# Pre-registration")
    a("")
    a(f"Frozen: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    a(f"Probe suite hash: `{PROBE_SUITE_HASH}`")
    a("")
    a("Committed before any real inference was run. The hash is recorded on every")
    a("database row, so a later edit to any probe question is detectable from the")
    a("data itself.")
    a("")

    a("## 1. Probe wording (frozen)")
    a("")
    for spec in PROBE_SUITE:
        a(f"### `{spec.kind.value}`")
        a("")
        a(f"- **Replay question:** {spec.question}")
        a(f"- **Logit cloze stem:** {spec.cloze_stem}")
        a("")
    a("Scoring is exact match after normalisation. **CPR requires ALL components")
    a("correct** — no partial credit, because a model that recalls the round number")
    a("but not the score has not comprehended the state that drives the decision.")
    a("")
    a("Both probe methods run on every trajectory. Replay probes branch the exact")
    a("context and never re-enter the game, so they cannot perturb the action they")
    a("measure. No model ever judges another model's answer.")
    a("")

    a("## 2. Primary hypothesis")
    a("")
    a("```")
    a("ATE_true = P(defect | Arm 3) - P(defect | Arm 3b)")
    a("```")
    a("")
    a("Arm 3b is the token- and position-matched, true-but-non-diagnostic placebo.")
    a("Comparison against the baseline arm is reported as secondary only: it")
    a("confounds the estimate with prompt perturbation, measured elsewhere at up to")
    a("76 percentage points.")
    a("")

    a("## 3. Sign-flip criterion (pre-registered, four conditions)")
    a("")
    a("Directions are derived from exact dynamic programming over the payoff")
    a("matrix, not chosen after the fact:")
    a("")
    a("| opponent | optimal | ALLD | regret of ALLD | predicted direction |")
    a("|---|---|---|---|---|")
    for row in summary_table(cfg):
        a(
            f"| {row['opponent']} | {row['optimal']} | {row['alld']} | "
            f"{row['regret_of_alld']} | **{row['predicted_direction']}** |"
        )
    a("")
    a("The sign-flip holds only if ALL of:")
    a("")
    a("1. both contrasts present")
    a(f"2. both individually significant at alpha = {ALPHA}")
    a("3. opposite signs")
    a("4. each sign matches its DP-derived prediction")
    a("")
    a("Rationale: no prompt-formatting artifact, distractor effect or framing bias")
    a("can satisfy 3 and 4 simultaneously, because none of them know which opponent")
    a("is being faced.")
    a("")

    a("## 4. Power (fixed in advance)")
    a("")
    a(f"- alpha = {ALPHA}, power = {POWER}")
    a(f"- Target MDE 0.05 requires **N = {required_n(0.5, 0.05)}** per arm")
    a("")
    a("| N per arm | minimum detectable effect |")
    a("|---|---|")
    for n in (30, 100, 200, 400, 800, 1600):
        a(f"| {n} | {min_detectable_effect(n):.3f} |")
    a("")
    a("Cells below the pre-registered N are flagged by `power_audit` and **must not**")
    a("be analysed. Reporting 'no effect' from a cell with no power to detect one is")
    a("the only true failure mode in this design.")
    a("")

    a("## 5. Secondary outcomes")
    a("")
    a("- `false_belief` = P(defect | 3c) - P(defect | 3b)")
    a("- `perturbation` = P(defect | 3d) - P(defect | 1)")
    a("- Regret vs DP-optimal, stratified by CPR")
    a("- CPR decay across turn index")
    a("- CoT x comprehension: CPR(scratchpad) - CPR(logit)")
    a("- Scaffold-echo rate in Arm 3c (manipulation check)")
    a("- Off-task rate: share of turns with action mass < 0.1")
    a("- Near-tie rate: share of decisions inside the FP noise band")
    a("")

    a("## 6. Commitments")
    a("")
    a("- Results reported **per model**. Pooling is secondary and must carry a")
    a("  heterogeneity statistic: the same perturbation has been measured moving one")
    a("  model 76 pp and another 1 pp.")
    a("- Every outcome in the spec's outcome table is publishable, including the null.")
    a("- Phase 2 (LLM vs LLM) runs only if Arm 3 CPR > 85%.")
    a("- Analysis code is frozen at this commit. Any later change is a deviation and")
    a("  must be reported as one.")
    return _emit(out)


def _emit(lines: list[str]) -> int:
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
