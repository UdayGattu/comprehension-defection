"""PRE-REGISTERED analysis.

Written before any real data exists. That is the entire point: with the ATE
definition, the significance test, the sign-flip criterion and the power check
all fixed in advance, it is structurally impossible to later choose whichever
method produces the nicest number — and that claim is verifiable from the commit
history.

Stdlib only. No scipy, so this runs identically on a laptop and on the rented
instance, and there is no version of a statistics library that could silently
change a p-value between machines.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import Arm, GameConfig, OpponentPolicy
from .optimal import predicted_defection_direction

ALPHA = 0.05
POWER = 0.80
Z_ALPHA_2 = 1.959963985
Z_BETA = 0.841621234


# ---------------------------------------------------------------- statistics

def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class ProportionDiff:
    p_treatment: float
    p_control: float
    n_treatment: int
    n_control: int

    @property
    def diff(self) -> float:
        return self.p_treatment - self.p_control

    @property
    def standard_error(self) -> float:
        a = self.p_treatment * (1 - self.p_treatment) / max(self.n_treatment, 1)
        b = self.p_control * (1 - self.p_control) / max(self.n_control, 1)
        return math.sqrt(a + b)

    @property
    def ci95(self) -> tuple[float, float]:
        margin = Z_ALPHA_2 * self.standard_error
        return (self.diff - margin, self.diff + margin)

    @property
    def z(self) -> float:
        """Pooled two-proportion z statistic."""
        n1, n2 = max(self.n_treatment, 1), max(self.n_control, 1)
        pooled = (self.p_treatment * n1 + self.p_control * n2) / (n1 + n2)
        se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        return self.diff / se if se > 0 else 0.0

    @property
    def p_value(self) -> float:
        return 2.0 * (1.0 - _normal_cdf(abs(self.z)))

    @property
    def significant(self) -> bool:
        return self.p_value < ALPHA


def wilson_interval(successes: int, n: int, z: float = Z_ALPHA_2) -> tuple[float, float]:
    """Wilson score interval.

    Used for CPR rather than the normal approximation because CPR may land near
    0 or 1, where the normal interval produces impossible bounds outside [0,1]
    and would understate uncertainty exactly where the decision matters most.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def required_n(p_baseline: float, min_detectable_effect: float) -> int:
    """Per-arm N for a two-proportion test at ALPHA / POWER.

    For p=0.5 and MDE=0.05 this returns ~1568, which is where the spec's
    N=1,600 comes from. At N=30 the MDE is ~0.36 - an experiment that cannot
    detect its own target effect, and would report a null it has no right to.
    """
    p2 = min(max(p_baseline + min_detectable_effect, 0.0), 1.0)
    numerator = (Z_ALPHA_2 + Z_BETA) ** 2 * (
        p_baseline * (1 - p_baseline) + p2 * (1 - p2)
    )
    return math.ceil(numerator / (min_detectable_effect ** 2))


def min_detectable_effect(n_per_arm: int, p_baseline: float = 0.5) -> float:
    if n_per_arm <= 0:
        return 1.0
    return (Z_ALPHA_2 + Z_BETA) * math.sqrt(2 * p_baseline * (1 - p_baseline) / n_per_arm)


# ------------------------------------------------------------------ loading

@dataclass(frozen=True)
class CellStats:
    model_id: str
    arm: str
    opponent: str
    readout_mode: str
    n_turns: int
    n_episodes: int
    defection_rate: float
    mean_regret: float | None
    cpr: float | None
    off_task_rate: float
    near_tie_rate: float


def load_cells(db_path: Path | str) -> list[CellStats]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT t.model_id, t.arm, t.opponent_policy, t.readout_mode,
               COUNT(*),
               COUNT(DISTINCT t.episode_id),
               AVG(CAST(t.agent_action = 'D' AS FLOAT)),
               AVG(CAST(t.cpr_score AS FLOAT)),
               AVG(CAST(t.action_mass_total < 0.1 AS FLOAT)),
               AVG(CAST(t.logit_gap < 1e-4 AS FLOAT))
        FROM turns t
        GROUP BY t.model_id, t.arm, t.opponent_policy, t.readout_mode
        """
    ).fetchall()
    regrets = {
        (m, a, o, r): mr
        for m, a, o, r, mr in conn.execute(
            """
            SELECT model_id, arm, opponent_policy, readout_mode, AVG(episode_regret)
            FROM episodes GROUP BY model_id, arm, opponent_policy, readout_mode
            """
        ).fetchall()
    }
    conn.close()

    out: list[CellStats] = []
    for m, a, o, r, n_turns, n_eps, defect, cpr, off_task, near_tie in rows:
        out.append(
            CellStats(
                model_id=m,
                arm=a,
                opponent=o,
                readout_mode=r,
                n_turns=n_turns,
                n_episodes=n_eps,
                defection_rate=defect,
                mean_regret=regrets.get((m, a, o, r)),
                cpr=cpr,
                off_task_rate=off_task,
                near_tie_rate=near_tie,
            )
        )
    return out


# --------------------------------------------------------------------- ATE

@dataclass(frozen=True)
class ATEResult:
    model_id: str
    opponent: str
    readout_mode: str
    contrast: str
    stats: ProportionDiff
    predicted_direction: str

    @property
    def observed_direction(self) -> str:
        return "down" if self.stats.diff < 0 else "up"

    @property
    def direction_matches_prediction(self) -> bool:
        return self.observed_direction == self.predicted_direction


def compute_ate(
    cells: list[CellStats],
    game_config: GameConfig,
    treatment: str = Arm.TREATMENT.value,
    control: str = Arm.PLACEBO_NONDIAGNOSTIC.value,
) -> list[ATEResult]:
    """ATE_true = P(defect | treatment) - P(defect | matched placebo).

    Defaults to the non-diagnostic placebo, NOT the baseline arm. Comparing
    against baseline confounds the estimate with prompt perturbation, which has
    been measured at up to 76 pp on its own.
    """
    index = {(c.model_id, c.arm, c.opponent, c.readout_mode): c for c in cells}
    results: list[ATEResult] = []
    for (model, arm, opponent, readout), cell in index.items():
        if arm != treatment:
            continue
        ctrl = index.get((model, control, opponent, readout))
        if ctrl is None:
            continue
        try:
            predicted = predicted_defection_direction(OpponentPolicy(opponent), game_config)
        except (ValueError, KeyError):
            predicted = "unknown"
        results.append(
            ATEResult(
                model_id=model,
                opponent=opponent,
                readout_mode=readout,
                contrast=f"{treatment}-vs-{control}",
                stats=ProportionDiff(
                    p_treatment=cell.defection_rate,
                    p_control=ctrl.defection_rate,
                    n_treatment=cell.n_turns,
                    n_control=ctrl.n_turns,
                ),
                predicted_direction=predicted,
            )
        )
    return sorted(results, key=lambda r: (r.model_id, r.readout_mode, r.opponent))


# -------------------------------------------------------------- sign flip

@dataclass(frozen=True)
class SignFlipResult:
    model_id: str
    readout_mode: str
    retaliator: ATEResult | None
    pushover: ATEResult | None

    @property
    def holds(self) -> bool:
        """PRE-REGISTERED criterion. All four conditions required:

          1. both contrasts present
          2. both individually significant at ALPHA
          3. opposite signs
          4. each sign matches its prediction from the exact DP

        No prompt artifact can satisfy 3 and 4 simultaneously, because none of
        them know which opponent is being faced.
        """
        if self.retaliator is None or self.pushover is None:
            return False
        if not (self.retaliator.stats.significant and self.pushover.stats.significant):
            return False
        if not (self.retaliator.stats.diff < 0 < self.pushover.stats.diff):
            return False
        return (
            self.retaliator.direction_matches_prediction
            and self.pushover.direction_matches_prediction
        )

    @property
    def verdict(self) -> str:
        if self.retaliator is None or self.pushover is None:
            return "INCOMPLETE - need both a retaliator and a pushover cell"
        if self.holds:
            return "HOLDS - comprehension explanation supported"
        if not (self.retaliator.stats.significant and self.pushover.stats.significant):
            return "UNDERPOWERED - at least one contrast is not significant"
        return "FAILS - signs do not flip as predicted"


def compute_sign_flip(
    ate_results: list[ATEResult],
    retaliator: str = OpponentPolicy.TFT.value,
    pushover: str = OpponentPolicy.ALLC.value,
) -> list[SignFlipResult]:
    keys = {(r.model_id, r.readout_mode) for r in ate_results}
    out: list[SignFlipResult] = []
    for model, readout in sorted(keys):
        out.append(
            SignFlipResult(
                model_id=model,
                readout_mode=readout,
                retaliator=next(
                    (r for r in ate_results
                     if r.model_id == model and r.readout_mode == readout
                     and r.opponent == retaliator),
                    None,
                ),
                pushover=next(
                    (r for r in ate_results
                     if r.model_id == model and r.readout_mode == readout
                     and r.opponent == pushover),
                    None,
                ),
            )
        )
    return out


# ------------------------------------------------------------------ report

def power_audit(cells: list[CellStats], target_mde: float = 0.05) -> list[str]:
    """Flag any cell that cannot resolve the target effect.

    An underpowered cell must not be analysed: reporting 'no effect' from an
    experiment with no power to detect one is the only true failure mode in the
    spec, and it is invisible unless explicitly checked.
    """
    needed = required_n(0.5, target_mde)
    warnings: list[str] = []
    for c in cells:
        if c.n_episodes < needed:
            warnings.append(
                f"UNDERPOWERED {c.model_id}/{c.arm}/{c.opponent}/{c.readout_mode}: "
                f"n={c.n_episodes} < {needed} required; MDE at this n is "
                f"{min_detectable_effect(c.n_episodes):.3f}"
            )
    return warnings


def format_report(cells: list[CellStats], game_config: GameConfig) -> str:
    lines: list[str] = []
    ate = compute_ate(cells, game_config)

    lines.append("=" * 76)
    lines.append("ATE_true  (Arm 3 vs Arm 3b, matched placebo)")
    lines.append("=" * 76)
    lines.append(f"{'model':22}{'readout':11}{'opp':7}{'ATE':>8}{'95% CI':>18}{'p':>9}  pred")
    for r in ate:
        lo, hi = r.stats.ci95
        flag = "" if r.direction_matches_prediction else "  <-- WRONG SIGN"
        lines.append(
            f"{r.model_id[:21]:22}{r.readout_mode:11}{r.opponent:7}"
            f"{r.stats.diff:+8.3f}{f'[{lo:+.3f},{hi:+.3f}]':>18}"
            f"{r.stats.p_value:>9.4f}  {r.predicted_direction}{flag}"
        )

    lines.append("")
    lines.append("=" * 76)
    lines.append("SIGN-FLIP TEST  (pre-registered criterion)")
    lines.append("=" * 76)
    for s in compute_sign_flip(ate):
        lines.append(f"  {s.model_id[:30]:32}{s.readout_mode:11}{s.verdict}")

    lines.append("")
    lines.append("=" * 76)
    lines.append("DIAGNOSTICS")
    lines.append("=" * 76)
    lines.append(f"{'model':22}{'arm':6}{'opp':7}{'CPR':>8}{'off-task':>10}{'near-tie':>10}")
    for c in sorted(cells, key=lambda c: (c.model_id, c.arm, c.opponent)):
        cpr = f"{c.cpr:.3f}" if c.cpr is not None else "n/a"
        lines.append(
            f"{c.model_id[:21]:22}{c.arm:6}{c.opponent:7}{cpr:>8}"
            f"{c.off_task_rate:>10.4f}{c.near_tie_rate:>10.4f}"
        )

    warnings = power_audit(cells)
    lines.append("")
    lines.append("=" * 76)
    lines.append(f"POWER AUDIT  ({len(warnings)} underpowered cell(s))")
    lines.append("=" * 76)
    lines.extend(f"  {w}" for w in warnings[:20])
    if not warnings:
        lines.append("  all cells meet the pre-registered N")
    return "\n".join(lines)
