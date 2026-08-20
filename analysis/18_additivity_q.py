#!/usr/bin/env python3
"""Cochran's Q over exp8 configurations: is the field asymmetry constant?

WHAT THIS TESTS

  Corollary 2 (testable implication) predicts something stricter than the
  registered band: that A_lo(c) is the SAME NUMBER at every configuration c.
  That is a homogeneity question. It needs no anchor, no ratio and therefore no
  anchor guard, so all three models enter -- including the two the guard removes
  from the registered verdict.

  Rejecting constancy rejects Assumption 1 (additive nuisance). Because psi has
  already cancelled by Proposition 1, the spread that remains is a lower bound
  on the content-by-configuration interaction epsilon, not a main effect of
  configuration.

WHY IT EXISTS AS A SCRIPT

  The manuscript quotes Q, p and I^2 for six model x opponent groups. Until this
  file, no released code produced them: `exp8_logodds.json` carries the A_lo
  intervals but no Q, and nothing derived one. A quoted statistic with no
  generating code is not reproducible, whatever else the repository releases.

METHOD, exactly as the manuscript describes it

  For each model x opponent group, over that group's configurations c:

      SE_i   = (A_lo_ci_hi - A_lo_ci_lo) / (2 * z)      z = 1.959964
      w_i    = 1 / SE_i^2
      A_bar  = sum(w_i A_i) / sum(w_i)
      Q      = sum(w_i (A_i - A_bar)^2)
      df     = k - 1
      p      = P(chi^2_df >= Q)
      I^2    = max(0, (Q - df) / Q)

  THE SYMMETRY ASSUMPTION IS REAL AND IS NOT HIDDEN. Converting a percentile
  bootstrap interval to a symmetric SE assumes the interval is symmetric about
  the point estimate. Several of these are conspicuously asymmetric. `--variance
  halfwidth-max` reports the alternative convention -- SE from the LARGER half
  width -- so the sensitivity is visible rather than argued. Q moves under it;
  the verdict does not.

  No scipy. The chi-square upper tail is computed from the regularised upper
  incomplete gamma function by continued fraction, which is accurate deep into
  the 1e-50 range where these p-values live. `--check-scipy` cross-validates
  against scipy when it happens to be installed.

USAGE

    python3 analysis/18_additivity_q.py
    python3 analysis/18_additivity_q.py --json ADDITIVITY_Q.json
    python3 analysis/18_additivity_q.py --variance halfwidth-max
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

Z = 1.959964  # two-sided 95% normal quantile, the one the intervals were cut at


# ---------------------------------------------------------------- chi-square

def _gamma_q_cf(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a,x), continued fraction (x > a+1).

    Lentz's algorithm. Returns exp(-x + a ln x - ln Gamma(a)) * CF.
    """
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _gamma_p_series(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a,x), series (x < a+1)."""
    ap = a
    s = 1.0 / a
    delta = s
    for _ in range(10000):
        ap += 1.0
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * 1e-16:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(q: float, df: int) -> float:
    """P(chi^2_df >= q), no scipy."""
    if q <= 0:
        return 1.0
    a, x = df / 2.0, q / 2.0
    return _gamma_q_cf(a, x) if x > a + 1.0 else 1.0 - _gamma_p_series(a, x)


# ---------------------------------------------------------------- the test

def se_from_ci(lo: float, hi: float, point: float, convention: str) -> float:
    if convention == "halfwidth-max":
        # SE from the LARGER half width: the conservative reading of an
        # asymmetric percentile interval.
        return max(hi - point, point - lo) / Z
    # default: the symmetric convention the manuscript uses
    return (hi - lo) / (2.0 * Z)


def cochran_q(points, ses):
    w = [1.0 / (s * s) for s in ses]
    sw = sum(w)
    abar = sum(wi * ai for wi, ai in zip(w, points)) / sw
    q = sum(wi * (ai - abar) ** 2 for wi, ai in zip(w, points))
    df = len(points) - 1
    return q, df, abar, chi2_sf(q, df), max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logodds", default="exp8_logodds.json")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write a machine-readable sidecar as well")
    ap.add_argument("--variance", choices=("halfwidth", "halfwidth-max"),
                    default="halfwidth",
                    help="halfwidth: (hi-lo)/2z, the manuscript's convention. "
                         "halfwidth-max: the larger half width, a sensitivity.")
    ap.add_argument("--check-scipy", action="store_true")
    args = ap.parse_args()

    try:
        blob = json.load(open(args.logodds, encoding="utf-8"))
    except FileNotFoundError:
        print(f"ABORT: {args.logodds} not found. Run analysis/16_exp8_logodds.py "
              f"first, or pass --logodds.", file=sys.stderr)
        return 2

    cells = blob["cells"]
    groups = defaultdict(list)
    for key, c in cells.items():
        a, lo, hi = c.get("A_lo"), c.get("A_lo_ci_lo"), c.get("A_lo_ci_hi")
        if a is None or lo is None or hi is None:
            print(f"  skip {key}: no A_lo interval")
            continue
        groups[(c["model"], c["opp"])].append((c["cond"], a, lo, hi))

    print(f"Cochran's Q over exp8 configurations — {args.logodds}")
    print(f"variance convention: {args.variance}   z = {Z}")
    print()
    print(f"  {'group':<20}{'k':>3}{'df':>4}{'Q':>12}{'p':>13}{'I2 %':>9}"
          f"{'A_lo mean':>12}  verdict")
    print("  " + "-" * 84)

    out, rejected = {}, 0
    for (model, opp) in sorted(groups):
        rows = sorted(groups[(model, opp)])
        pts = [r[1] for r in rows]
        ses = [se_from_ci(r[2], r[3], r[1], args.variance) for r in rows]
        q, df, abar, p, i2 = cochran_q(pts, ses)
        rej = p < 0.05
        rejected += rej
        print(f"  {model + '|' + opp:<20}{len(pts):>3}{df:>4}{q:>12.1f}"
              f"{p:>13.2e}{i2:>9.1f}{abar:>12.4f}  "
              f"{'REJECTS constancy' if rej else 'does not reject'}")
        out[f"{model}|{opp}"] = {
            "k": len(pts), "df": df, "Q": q, "p": p, "I2_pct": i2,
            "A_lo_weighted_mean": abar,
            "conditions": [r[0] for r in rows],
            "A_lo": pts,
            "SE": ses,
        }

    qs = [v["Q"] for v in out.values()]
    i2s = [v["I2_pct"] for v in out.values()]
    ps = [v["p"] for v in out.values()]
    dfs = sorted({v["df"] for v in out.values()})
    print()
    print(f"  rejected in {rejected} of {len(out)} groups")
    print(f"  Q      {min(qs):.1f} to {max(qs):.1f}")
    print(f"  df     {' or '.join(str(d) for d in dfs)}")
    print(f"  I^2    {min(i2s):.1f}% to {max(i2s):.1f}%")
    print(f"  max p  {max(ps):.2e}")

    if args.check_scipy:
        try:
            from scipy.stats import chi2 as _c
            worst = max(abs(chi2_sf(v["Q"], v["df"]) - _c.sf(v["Q"], v["df"]))
                        for v in out.values())
            print(f"  scipy cross-check: max abs difference {worst:.3e}")
        except ImportError:
            print("  scipy not installed; internal chi-square used (this is fine)")

    if args.json_out:
        payload = {
            "_generated_by": "analysis/18_additivity_q.py",
            "_source": args.logodds,
            "_variance_convention": args.variance,
            "_z": Z,
            "_note": ("Cochran's Q over configurations within each model x "
                      "opponent group. Tests Corollary 2's prediction that "
                      "A_lo(c) is constant. Rejecting it rejects Assumption 1. "
                      "This test is NOT pre-registered."),
            "groups": out,
            "summary": {
                "rejected_groups": rejected, "n_groups": len(out),
                "Q_min": min(qs), "Q_max": max(qs),
                "I2_min_pct": min(i2s), "I2_max_pct": max(i2s),
                "p_max": max(ps), "df_values": dfs,
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(f"\n  wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
