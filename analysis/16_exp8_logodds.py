#!/usr/bin/env python3
"""ANALYSIS 16 - exp8 on the LOG-ODDS scale, the registered reporting form.

WHY THIS SCRIPT EXISTS
    PREREGISTRATION_EXP8.md section 4, "Secondary, descriptive", registers

        A_lo = logit P(D|3m) - logit P(D|3s)

    as "reported for every cell, and the ONLY form quoted for COMPRESSED cells
    (section 5.4)".  Section 5.4 defines COMPRESSED as: not FLOOR and not
    CEILING, but min P(D) over {3,3s,3m} <= 0.15 or max >= 0.85.  Every
    non-floored cell in this corpus is COMPRESSED (analysis/15 prints the
    classification; there are ZERO CLEAN cells).  So log-odds is the registered
    reporting form for every cell that votes, and until this script it had never
    been computed.  analysis/15_exp8_stability.py computes A, S and the verdict
    on the PROBABILITY scale only.

WHY A SEPARATE SCRIPT AND NOT AN EDIT TO analysis/15
    1. analysis/15 produced `exp8_stability.json`, the probability-scale
       artifact the write-up already cites.  Editing 15 would put a second set
       of numbers behind the same filename and the same provenance.
    2. Nothing is re-implemented.  This script IMPORTS analysis/15 and calls its
       `episode_rates`, `classify`, `ci`, `cpr_for`, `ro_uri` and its constants.
       There is exactly one copy of the loader, the turn-0 exclusion, the
       section-5.4 classifier and the percentile rule.
    3. It reproduces analysis/15's probability-scale bootstrap BIT FOR BIT and
       says so at the end (`--verify`).  The RNG is consumed in the same order,
       with the same calls, so draw b here is draw b there.  If that check ever
       fails, the two scales are not being computed on the same resamples and
       nothing below is comparable.

WHAT IT ADDS, PER CELL
    A_lo                     point estimate on the log-odds scale
    [A_lo_ci_lo, A_lo_ci_hi] episode-clustered percentile bootstrap CI,
                             seed 20260814, 10,000 resamples, turn 0 excluded
    S_lo  = A_lo(cond) / A_lo(anchor)
    dA_lo = A_lo(cond) - A_lo(anchor), with its bootstrap CI (paired draws)
    and the section 5.7 verdict re-evaluated on that scale.

THE ZERO PROBLEM, AND THE TREATMENT CHOSEN
    logit(0) = -inf.  Two separate facts, and they are not the same fact:

    (a) OBSERVED point estimates.  NO cell in this corpus has an observed
        P(D|3s) or P(D|3m) of exactly zero.  The "0.000" printed for mistral by
        analysis/15 is three-decimal display rounding of 3/19000 = 0.000158
        (mistral/anchor/tft) and 5/19000 = 0.000263 (mistral/anchor/allc).  The
        smallest non-rounded rate in either arm A reads is 12/19000.  So no cell
        is dropped, and none needs to be, at the point estimate.

    (b) BOOTSTRAP draws.  This is where the zeros actually are.  mistral's
        anchor 3s defections live in 3 episodes out of 1000; an episode-
        clustered resample misses all three with probability (999/1000)^1000
        ~ e^-3 ~ 5% of draws.  Those draws have p = 0 exactly and an
        uncorrected logit is -inf, which would make the CI infinite.  Any
        script that "works" on this corpus without saying what it did about
        zeros has silently made this choice somewhere.

    TREATMENT (primary): HALDANE-ANSCOMBE, h = 0.5, on the TURN counts, applied
    UNIFORMLY to every cell and every bootstrap draw.

        logit_h(p; N) = log( (p*N + h) / ((1-p)*N + h) ),  h = 0.5

    N is the number of Bernoulli turn observations behind that arm-cell after
    the turn-0 exclusion.  Every episode in this corpus is exactly 19 scored
    turns long (asserted at load time), and the registered estimator is the
    unweighted mean of episode rates, so p*N is exactly the observed defection
    count - an integer - both in the data and in every resample.  h = 0.5 is
    then literally "add half a defection and half a cooperation", the usual
    Bernoulli meaning, and not a fudge chosen to make a number behave.

    Why uniform and not zeros-only: applying a correction only where p = 0 makes
    the transform a different function in different cells, so A_lo would not be
    comparable across the cells S_lo divides.  One monotone function everywhere,
    or the ratio is not a ratio of like things.

    Why h on turns and not on episodes: h = 0.5 on 19,000 turns shifts p by
    ~2.6e-5, which is invisible anywhere except at a count of 0 or 1 - exactly
    where a correction is supposed to act and nowhere else.  The honest
    objection is that turns within an episode are correlated, so 19,000 is not
    19,000 independent observations and the correction is smaller than a
    fully-honest one would be.  That objection is answered by measurement, not
    by assertion: SENSITIVITY (i) below re-runs everything with N = the number
    of EPISODES (1,000), i.e. a correction 19x larger, and the script prints
    every cell whose band membership or CI-excludes-0 status changes.  It also
    runs SENSITIVITY (ii), dropping the offending draws instead of correcting
    them, and reports what fraction of draws that discards.

WHAT IS AND IS NOT SCALE-DEPENDENT (this decides the verdict, so it is stated
    before the numbers)
    * FLOOR / CEILING / COMPRESSED (section 5.4) are defined on P(D).  They are
      probability-scale definitions and are NOT recomputed on log-odds.
    * The anchor guard `|A(m,anchor,o)| < 0.05` (section 5.4) is written on A,
      and section 4 defines A as the probability difference.  It is applied AS
      WRITTEN, so the set of surviving models is identical on both scales.
      A log-odds-flavoured guard would be a different pre-registration; the
      script reports what such a guard would have done, clearly marked
      NOT REGISTERED, and lets nothing in the verdict depend on it.
    * F2 (sign reversal) is EXACTLY scale-invariant, and this is provable, not
      hopeful: logit_h is strictly increasing, so sign(A_lo) = sign(A) in every
      single draw; a percentile interval excludes 0 iff strictly more than 97.5%
      of draws share a sign; that count is identical draw-for-draw.  The script
      asserts the equality rather than trusting the argument.
    * F1 (magnitude, S band) IS scale-dependent - but F1 is gated on CLEAN
      cells and there are none, so F1 cannot fire on either scale.  Confirmed by
      counting, not assumed.
    * SUPPORT reads S and the factorial, both scale-dependent.  Recomputed.

USAGE
    cd <repo root>            # the globs are relative, as in analysis/15
    python3 analysis/16_exp8_logodds.py
    python3 analysis/16_exp8_logodds.py --bootstrap 500 --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import importlib.util
import json
import math
import random
import shutil
import sqlite3
import statistics
import sys
import tempfile
from pathlib import Path

# ---- import analysis/15 as a module (its name starts with a digit) ---------
_S15 = Path(__file__).with_name("15_exp8_stability.py")
_spec = importlib.util.spec_from_file_location("exp8_stability_15", _S15)
s15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s15)          # module-level only; main() is guarded

RULE = s15.RULE
SEED = s15.SEED                        # 20260814, section 4
BOOT_DEFAULT = s15.BOOT_DEFAULT        # 10,000, section 4
BAND_LO, BAND_HI = s15.BAND_LO, s15.BAND_HI      # [0.50, 2.00], section 5.6
ANCHOR_GUARD = s15.ANCHOR_GUARD        # 0.05, section 5.4
H = 0.5                                # Haldane-Anscombe


# ---------------------------------------------------------------- transform
def logit_h(p: float, n: float, h: float = H) -> float:
    """Haldane-Anscombe logit. n = effective count behind p; h added to both.

    Strictly increasing in p for every n>0 and h>0, which is what makes the
    sign of A_lo equal to the sign of A in every draw.
    """
    return math.log((p * n + h) / ((1.0 - p) * n + h))


# ------------------------------------------------------------------ loading
def open_db(path: Path):
    """Decompress-if-needed and open read-only. Mirrors analysis/15.load_group,
    which does not expose the connection; we need it for the episode-length
    assertion that licenses the turn-count form of the correction."""
    tmp = None
    if path.suffix == ".gz":
        tmp = Path(tempfile.mkdtemp()) / path.stem
        with gzip.open(path, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        path = tmp
    return sqlite3.connect(s15.ro_uri(path), uri=True), tmp


def episode_lengths(conn, arm: str, opp: str) -> list[int]:
    return [r[0] for r in conn.execute(
        "SELECT COUNT(*) FROM turns WHERE arm=? AND opponent_policy=? AND turn > 0 "
        "GROUP BY episode_id ORDER BY episode_id", (arm, opp)).fetchall()]


def load_group_ext(path: Path) -> dict:
    """arm -> opp -> (rates, turns_per_episode).  `rates` comes from
    analysis/15.episode_rates verbatim, so the estimator is not re-implemented."""
    conn, tmp = open_db(path)
    out = {}
    for arm in ("3", "3s", "3m"):
        out[arm] = {}
        for opp in ("tft", "allc"):
            rates = s15.episode_rates(conn, arm, opp)
            lens = set(episode_lengths(conn, arm, opp))
            if rates and len(lens) != 1:
                raise SystemExit(
                    f"{path.name} {arm}/{opp}: episodes have unequal scored "
                    f"lengths {sorted(lens)}. The turn-count form of the "
                    f"Haldane-Anscombe correction assumes p*N is the observed "
                    f"defection count, which needs equal-length episodes "
                    f"because the registered estimator is an unweighted mean "
                    f"of episode rates. Fix the correction before continuing.")
            out[arm][opp] = (rates, lens.pop() if lens else 0)
    conn.close()
    if tmp:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    return out


# ---------------------------------------------------------------- bootstrap
def boot_pairs(rng, m: list[float], s: list[float], boot: int):
    """Resampled arm means, episodes as clusters.

    The RNG is consumed EXACTLY as analysis/15.boot_A consumes it - choices(m)
    then choices(s), once per iteration - so A = pm - ps here is draw-for-draw
    identical to analysis/15's A_draws. Everything else is derived from these
    two lists, so the probability scale and the log-odds scale are read off the
    same resamples rather than two independent bootstraps.
    """
    nm, ns = len(m), len(s)
    pm, ps = [], []
    for _ in range(boot):
        pm.append(statistics.fmean(rng.choices(m, k=nm)))
        ps.append(statistics.fmean(rng.choices(s, k=ns)))
    return pm, ps


def alo_draws(pm, ps, nm_eff, ns_eff, h=H):
    return [logit_h(a, nm_eff, h) - logit_h(b, ns_eff, h) for a, b in zip(pm, ps)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="exp8_*_logit.sqlite*")
    ap.add_argument("--bootstrap", type=int, default=BOOT_DEFAULT)
    ap.add_argument("--out", default="exp8_logodds.json")
    ap.add_argument("--verify", default="exp8_stability.json",
                    help="probability-scale artifact to reproduce bit-for-bit")
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.glob)
                   if not p.endswith(".sqlite-wal") and not p.endswith(".sqlite-shm"))
    if not paths:
        raise SystemExit(f"no databases matched {args.glob!r} (run from the repo root)")

    import re
    rng = random.Random(SEED)
    cells: dict[tuple[str, str, str], dict] = {}

    print(f"{RULE}\nexp8 LOG-ODDS - PREREGISTRATION_EXP8.md section 4 "
          f"(registered reporting form for COMPRESSED cells)\n{RULE}")
    print(f"  seed {SEED}   bootstrap {args.bootstrap}   turn 0 EXCLUDED")
    print(f"  zero treatment: Haldane-Anscombe h={H} on turn counts, applied to "
          f"every cell and every draw\n")

    for p in paths:
        m = re.match(r"exp8_([a-z0-9]+)_([a-z0-9]+)_logit\.sqlite", p.name)
        if not m:
            continue
        model, cond = m.group(1), m.group(2)
        g = load_group_ext(p)
        for opp in ("tft", "allc"):
            (r3, L3), (r3s, L3s), (r3m, L3m) = (g["3"][opp], g["3s"][opp], g["3m"][opp])
            if not (r3 and r3s and r3m):
                continue
            p3, p3s, p3m = (statistics.fmean(r3), statistics.fmean(r3s),
                            statistics.fmean(r3m))
            Nm, Ns = len(r3m) * L3m, len(r3s) * L3s     # turn counts behind each arm
            Em, Es = float(len(r3m)), float(len(r3s))   # episode (cluster) counts

            pm, ps = boot_pairs(rng, r3m, r3s, args.bootstrap)
            A_draws = [a - b for a, b in zip(pm, ps)]
            Alo_draws = alo_draws(pm, ps, Nm, Ns)               # primary
            Alo_ep_draws = alo_draws(pm, ps, Em, Es)            # sensitivity (i)
            keep = [i for i in range(args.bootstrap)
                    if 0.0 < pm[i] < 1.0 and 0.0 < ps[i] < 1.0]
            n_zero = args.bootstrap - len(keep)                 # sensitivity (ii)
            Alo_drop = [math.log(pm[i] / (1 - pm[i])) - math.log(ps[i] / (1 - ps[i]))
                        for i in keep]

            A = p3m - p3s
            Alo = logit_h(p3m, Nm) - logit_h(p3s, Ns)
            a_ci = s15.ci(A_draws)
            alo_ci = s15.ci(Alo_draws)
            alo_ep_ci = s15.ci(Alo_ep_draws)
            alo_drop_ci = s15.ci(Alo_drop) if Alo_drop else (float("nan"),) * 2

            cells[(model, cond, opp)] = dict(
                model=model, cond=cond, opp=opp,
                p3=p3, p3s=p3s, p3m=p3m,
                n_ep_3m=len(r3m), n_ep_3s=len(r3s), N_turns_3m=Nm, N_turns_3s=Ns,
                d_3m=round(p3m * Nm), d_3s=round(p3s * Ns),
                observed_zero_3s=bool(p3s == 0.0), observed_zero_3m=bool(p3m == 0.0),
                A=A, A_ci_lo=a_ci[0], A_ci_hi=a_ci[1],
                A_excl_0=bool(a_ci[0] > 0 or a_ci[1] < 0),
                A_lo=Alo, A_lo_ci_lo=alo_ci[0], A_lo_ci_hi=alo_ci[1],
                A_lo_excl_0=bool(alo_ci[0] > 0 or alo_ci[1] < 0),
                A_lo_ep=logit_h(p3m, Em) - logit_h(p3s, Es),
                A_lo_ep_ci_lo=alo_ep_ci[0], A_lo_ep_ci_hi=alo_ep_ci[1],
                A_lo_drop_ci_lo=alo_drop_ci[0], A_lo_drop_ci_hi=alo_drop_ci[1],
                n_zero_draws=n_zero, frac_zero_draws=n_zero / args.bootstrap,
                klass=s15.classify(p3, p3s, p3m),
                cpr3=s15.cpr_for(p, opp),
                _A=A_draws, _Alo=Alo_draws, _Alo_ep=Alo_ep_draws,
            )
        print(f"  loaded {p.name}")

    # -- regression check against analysis/15 --------------------------------
    ver = Path(args.verify)
    verify_msg = f"SKIPPED ({ver} not found)"
    if ver.exists():
        ref = json.loads(ver.read_text())
        if ref.get("bootstrap") != args.bootstrap or ref.get("seed") != SEED:
            verify_msg = (f"SKIPPED (artifact has seed={ref.get('seed')} "
                          f"bootstrap={ref.get('bootstrap')}, this run has "
                          f"seed={SEED} bootstrap={args.bootstrap})")
        else:
            bad = []
            for k, c in cells.items():
                r = ref["cells"].get("|".join(k))
                if r is None:
                    bad.append(f"{k} missing from artifact")
                    continue
                for mine, theirs in (("A", "A"), ("A_ci_lo", "A_lo"), ("A_ci_hi", "A_hi")):
                    if abs(c[mine] - r[theirs]) > 1e-12:
                        bad.append(f"{k}.{theirs}: {c[mine]!r} vs {r[theirs]!r}")
            verify_msg = ("PASS - probability-scale A and its CI reproduce "
                          "analysis/15 exactly" if not bad
                          else "FAIL:\n      " + "\n      ".join(bad[:10]))
    print(f"\n  reproduction of analysis/15 (same seed, same draws): {verify_msg}")

    # -- S, S_lo, and the CIs on the differences -----------------------------
    for (model, cond, opp), c in cells.items():
        anc = cells.get((model, "anchor", opp))
        c["S"] = c["S_lo"] = None
        c["excluded_reason"] = None
        c["S_lo_unguarded"] = None
        if anc is None:
            c["excluded_reason"] = "no anchor"
            continue
        # descriptive only; NOT registered. Shows what the log-odds scale would
        # have said had the anchor guard not removed the model.
        if anc["A_lo"] != 0:
            c["S_lo_unguarded"] = c["A_lo"] / anc["A_lo"]
        # the registered guard, section 5.4, written on A (probability scale)
        if anc["klass"] in ("FLOOR", "CEILING"):
            c["excluded_reason"] = f"anchor is {anc['klass']}"
            continue
        if abs(anc["A"]) < ANCHOR_GUARD:
            c["excluded_reason"] = (f"anchor guard: |A(anchor)|="
                                    f"{abs(anc['A']):.4f} < {ANCHOR_GUARD}")
            continue
        c["S"] = c["A"] / anc["A"]
        c["S_lo"] = c["A_lo"] / anc["A_lo"]
        if cond == "anchor":
            continue
        for tag, key, pt in (("", "_A", "A"), ("_lo", "_Alo", "A_lo"),
                             ("_lo_ep", "_Alo_ep", "A_lo_ep")):
            d = [x - y for x, y in zip(c[key], anc[key])]
            lo, hi = s15.ci(d)
            c[f"dA{tag}"] = c[pt] - anc[pt]
            c[f"dA{tag}_ci_lo"], c[f"dA{tag}_ci_hi"] = lo, hi
            c[f"dA{tag}_excl_0"] = bool(lo > 0 or hi < 0)
        c["out_of_band"] = bool(c["S"] < BAND_LO or c["S"] > BAND_HI)
        c["out_of_band_lo"] = bool(c["S_lo"] < BAND_LO or c["S_lo"] > BAND_HI)
        c["S_lo_ep"] = (c["A_lo_ep"] / anc["A_lo_ep"]) if anc["A_lo_ep"] else None
        c["out_of_band_lo_ep"] = bool(c["S_lo_ep"] is not None and
                                      (c["S_lo_ep"] < BAND_LO or c["S_lo_ep"] > BAND_HI))
        c["unstable"] = bool(c["out_of_band"] and c["dA_excl_0"]
                             and c["klass"] not in ("FLOOR", "CEILING"))
        c["unstable_lo"] = bool(c["out_of_band_lo"] and c["dA_lo_excl_0"]
                                and c["klass"] not in ("FLOOR", "CEILING"))

    # -- sign invariance, asserted rather than assumed -----------------------
    sign_bad = []
    for k, c in cells.items():
        n_pos_A = sum(1 for x in c["_A"] if x > 0)
        n_pos_L = sum(1 for x in c["_Alo"] if x > 0)
        if n_pos_A != n_pos_L or c["A_excl_0"] != c["A_lo_excl_0"] \
                or ((c["A"] > 0) != (c["A_lo"] > 0)):
            sign_bad.append("|".join(k))
    print(f"  sign invariance (logit_h strictly increasing => F2 is scale-free): "
          f"{'HOLDS in all %d cells' % len(cells) if not sign_bad else 'VIOLATED in ' + ', '.join(sign_bad)}")

    n_clean = sum(1 for c in cells.values() if c["klass"] == "CLEAN")
    print(f"  CLEAN cells in corpus: {n_clean}  "
          f"-> F1 (gated on CLEAN, section 5.7) "
          f"{'CANNOT FIRE on either scale' if n_clean == 0 else 'is live'}")

    # -- zero accounting -----------------------------------------------------
    print(f"\n{RULE}\nZERO ACCOUNTING\n{RULE}")
    obs_zero = [k for k, c in cells.items()
                if c["observed_zero_3s"] or c["observed_zero_3m"]]
    print(f"  cells with an OBSERVED rate of exactly 0 in arm 3s or 3m: "
          f"{len(obs_zero)}   {[('|'.join(k)) for k in obs_zero]}")
    print(f"  smallest observed arm rates (arm, count/N):")
    for k in sorted(cells, key=lambda k: min(cells[k]["p3s"], cells[k]["p3m"]))[:4]:
        c = cells[k]
        print(f"    {'|'.join(k):28} 3s={c['d_3s']}/{c['N_turns_3s']}="
              f"{c['p3s']:.6f}   3m={c['d_3m']}/{c['N_turns_3m']}={c['p3m']:.6f}")
    aff = [(k, c) for k, c in cells.items() if c["n_zero_draws"]]
    print(f"\n  cells with at least one bootstrap draw at p=0 (where the "
          f"correction actually acts): {len(aff)} of {len(cells)}")
    for k, c in sorted(aff, key=lambda kc: -kc[1]["n_zero_draws"]):
        print(f"    {'|'.join(k):28} {c['n_zero_draws']:6d}/{len(c['_A'])} draws "
              f"({c['frac_zero_draws']*100:5.2f}%)  "
              f"CI(A_lo) HA={c['A_lo_ci_lo']:+7.3f},{c['A_lo_ci_hi']:+7.3f}  "
              f"drop-draws={c['A_lo_drop_ci_lo']:+7.3f},{c['A_lo_drop_ci_hi']:+7.3f}")
    if not aff:
        print("    none")

    # -- per-cell table ------------------------------------------------------
    print(f"\n{RULE}\nPER-CELL: A (probability) vs A_lo (log-odds, registered form)\n{RULE}")
    hdr = (f"{'model':8}{'cond':13}{'opp':5}{'P(D|3s)':>9}{'P(D|3m)':>9}"
           f"{'A':>8}{'95% CI on A':>18}{'A_lo':>8}{'95% CI on A_lo':>20}"
           f"{'S':>7}{'S_lo':>8}{'class':>12}  flags")
    print(hdr)
    for k in sorted(cells):
        c = cells[k]
        S = "    n/a" if c["S"] is None else f"{c['S']:7.3f}"
        Sl = "     n/a" if c["S_lo"] is None else f"{c['S_lo']:8.3f}"
        f = []
        if c["S"] is None and c["S_lo_unguarded"] is not None and c["cond"] != "anchor":
            f.append(f"S undefined (guard); S_lo would be {c['S_lo_unguarded']:.3f}")
        if c.get("out_of_band") is not None and c.get("out_of_band") != c.get("out_of_band_lo"):
            f.append("SCALES DISAGREE")
        elif c.get("out_of_band"):
            f.append("out of band on BOTH")
        print(f"{c['model']:8}{c['cond']:13}{c['opp']:5}{c['p3s']:9.4f}{c['p3m']:9.4f}"
              f"{c['A']:+8.3f}  [{c['A_ci_lo']:+6.3f},{c['A_ci_hi']:+6.3f}]"
              f"{c['A_lo']:+8.3f}  [{c['A_lo_ci_lo']:+7.3f},{c['A_lo_ci_hi']:+7.3f}]"
              f"{S}{Sl}{c['klass']:>12}  {'; '.join(f)}")

    # -- band membership, the direct comparison ------------------------------
    print(f"\n{RULE}\nBAND MEMBERSHIP  [0.50, 2.00]  - non-anchor cells with S defined\n{RULE}")
    print(f"{'cell':30}{'S':>8}{'  ':2}{'':6}{'S_lo':>8}{'  ':2}{'':6}"
          f"{'dA_lo':>8}{'95% CI on dA_lo':>20}{'  excl 0':>9}")
    disagree = []
    for k in sorted(cells):
        c = cells[k]
        if c["S"] is None or c["cond"] == "anchor":
            continue
        a = "OUT" if c["out_of_band"] else "in "
        b = "OUT" if c["out_of_band_lo"] else "in "
        if a != b:
            disagree.append(k)
        print(f"{'|'.join(k):30}{c['S']:8.3f}  {a:6}{c['S_lo']:8.3f}  {b:6}"
              f"{c['dA_lo']:+8.3f}  [{c['dA_lo_ci_lo']:+7.3f},{c['dA_lo_ci_hi']:+7.3f}]"
              f"{str(c['dA_lo_excl_0']):>9}"
              f"{'   <-- DISAGREE' if a != b else ''}")

    print(f"\n  cells where the two scales DISAGREE about band membership: "
          f"{len(disagree)}")
    for k in disagree:
        c = cells[k]
        print(f"    {'|'.join(k):30} S={c['S']:.3f} ({'OUT' if c['out_of_band'] else 'in'})"
              f"  ->  S_lo={c['S_lo']:.3f} ({'OUT' if c['out_of_band_lo'] else 'in'})")

    # -- sensitivity ---------------------------------------------------------
    print(f"\n{RULE}\nSENSITIVITY TO THE ZERO TREATMENT\n{RULE}")
    print("  (i) Haldane-Anscombe on EPISODE counts (N=1000) instead of turn "
          "counts (N=19000):")
    moved = []
    for k in sorted(cells):
        c = cells[k]
        if c["S_lo"] is None or c["cond"] == "anchor":
            continue
        if c["out_of_band_lo"] != c["out_of_band_lo_ep"] or \
                c["dA_lo_excl_0"] != c["dA_lo_ep_excl_0"]:
            moved.append(k)
            print(f"    {'|'.join(k):30} S_lo {c['S_lo']:.3f} -> {c['S_lo_ep']:.3f}   "
                  f"band {'OUT' if c['out_of_band_lo'] else 'in'} -> "
                  f"{'OUT' if c['out_of_band_lo_ep'] else 'in'}")
    if not moved:
        print("    no S-defined cell changes band membership or CI-excludes-0 status.")
    # The S-defined cells are exactly the ones the zeros are NOT in (mistral is
    # guarded out), so the shift must also be reported over every cell or the
    # sensitivity is being run where it cannot bite.
    for label, sel in (("all 32 cells", list(cells.values())),
                       ("cells with S defined (vote-relevant)",
                        [c for c in cells.values() if c["S_lo"] is not None])):
        ranked = sorted(sel, key=lambda c: -abs(c["A_lo_ep"] - c["A_lo"]))
        print(f"    largest |A_lo(episode-N) - A_lo(turn-N)| over {label}:")
        for c in ranked[:3]:
            print(f"      {'|'.join((c['model'], c['cond'], c['opp'])):28} "
                  f"{c['A_lo']:+8.4f} -> {c['A_lo_ep']:+8.4f}   "
                  f"(shift {c['A_lo_ep'] - c['A_lo']:+.4f}, class {c['klass']})")
    flipped = [k for k, c in cells.items()
               if c["A_lo_excl_0"] != bool(c["A_lo_ep_ci_lo"] > 0 or c["A_lo_ep_ci_hi"] < 0)]
    print(f"    cells whose 'A_lo CI excludes 0' status flips under the "
          f"episode-N correction: {len(flipped)} {['|'.join(k) for k in flipped]}")
    print("  (ii) dropping draws with a zero rate instead of correcting them:")
    print("    point estimates are unaffected (no observed rate is 0); the CIs "
          "affected are listed under ZERO ACCOUNTING above.")

    # -- factorial on A_lo (SUPPORT reads it) --------------------------------
    print(f"\n{RULE}\nFACTORIAL ON A_lo  (section 4; SUPPORT's second clause reads it)\n{RULE}")
    factorial = {}
    for mdl in sorted({c["model"] for c in cells.values()}):
        for opp in ("tft", "allc"):
            got = {c["cond"]: c for c in cells.values()
                   if c["model"] == mdl and c["opp"] == opp and c["cond"] in CODING}
            if len(got) < 4:
                continue
            full = len(got) == 8
            print(f"\n  {mdl} / {opp}   {len(got)} conditions   "
                  f"{'FULL 2^3' if full else 'HALF FRACTION - ALIASED (T=-OP, O=-TP, P=-TO)'}")
            for name, fn in TERMS.items():
                plus = [c for kk, c in got.items() if fn(*CODING[kk]) > 0]
                minus = [c for kk, c in got.items() if fn(*CODING[kk]) < 0]
                if not plus or not minus:
                    continue
                eff = (statistics.fmean([c["A_lo"] for c in plus])
                       - statistics.fmean([c["A_lo"] for c in minus]))
                draws = [statistics.fmean([c["_Alo"][b] for c in plus])
                         - statistics.fmean([c["_Alo"][b] for c in minus])
                         for b in range(args.bootstrap)]
                lo, hi = s15.ci(draws)
                star = " *" if (lo > 0 or hi < 0) else ""
                print(f"    {(name if full else name + ' + alias'):12}{eff:+8.4f}   "
                      f"[{lo:+7.4f},{hi:+7.4f}]{star}")
                factorial[f"{mdl}|{opp}|{name}"] = dict(
                    model=mdl, opp=opp, term=name, aliased=not full, scale="log-odds",
                    effect=eff, lo=lo, hi=hi, excludes_0=bool(lo > 0 or hi < 0))
    print("\n  * = 95% bootstrap CI excludes 0")

    # -- verdict on the log-odds scale, section 5.7 exactly ------------------
    print(f"\n{RULE}\nVERDICT ON THE LOG-ODDS SCALE - PREREGISTRATION_EXP8.md "
          f"section 5.7\n{RULE}")
    verdicts = {}
    for opp in ("tft", "allc"):
        print(f"\n  ---- opponent = {opp} ----")
        models = sorted({c["model"] for c in cells.values() if c["opp"] == opp})
        surviving = []
        for mdl in models:
            anc = cells.get((mdl, "anchor", opp))
            if anc is None:
                continue
            if anc["klass"] in ("FLOOR", "CEILING"):
                print(f"    {mdl:9} EXCLUDED - anchor is {anc['klass']} "
                      f"(section 5.4; the classification is on P(D) and does not "
                      f"move with the reporting scale)")
                continue
            if abs(anc["A"]) < ANCHOR_GUARD:
                print(f"    {mdl:9} EXCLUDED - anchor guard |A|={abs(anc['A']):.4f} "
                      f"< {ANCHOR_GUARD}  [A_lo(anchor)={anc['A_lo']:+.3f}, "
                      f"CI {anc['A_lo_ci_lo']:+.3f},{anc['A_lo_ci_hi']:+.3f}]")
                continue
            surviving.append(mdl)
        n = len(surviving)
        majority = n // 2 + 1
        print(f"    surviving: {surviving or '-'}   strict majority = {majority} of {n}")

        void = []
        for mdl in surviving:
            anc = cells[(mdl, "anchor", opp)]
            if anc["cpr3"] >= 0.95:
                for c in [x for x in cells.values() if x["model"] == mdl
                          and x["opp"] == opp and x["cond"] != "anchor"]:
                    if c["cpr3"] == c["cpr3"] and c["cpr3"] < 0.80:
                        void.append(f"{mdl}/{c['cond']} CPR(3)={c['cpr3']:.3f}")
        print(f"    F3 readability: "
              f"{'VOID -> ' + ', '.join(void) if void else 'all conditions readable'}")

        f1, f2, sup = [], [], []
        for mdl in surviving:
            anc = cells[(mdl, "anchor", opp)]
            conds = [c for c in cells.values() if c["model"] == mdl
                     and c["opp"] == opp and c["cond"] != "anchor"]
            clean = [c for c in conds if c["klass"] == "CLEAN"]
            cc = [c for c in conds if c["klass"] in ("CLEAN", "COMPRESSED")]
            if any(c["out_of_band_lo"] and c["dA_lo_excl_0"] for c in clean):
                f1.append(mdl)
            if any((c["A_lo"] > 0) != (anc["A_lo"] > 0) and c["A_lo_excl_0"]
                   and anc["A_lo_excl_0"] for c in cc):
                f2.append(mdl)
            elig = [c for c in conds if c["S_lo"] is not None
                    and c["klass"] not in ("FLOOR", "CEILING")]
            inband = [c for c in elig if BAND_LO <= c["S_lo"] <= BAND_HI]
            maxeff = max((abs(v["effect"]) for kk, v in factorial.items()
                          if v["model"] == mdl and v["opp"] == opp
                          and v["term"] in ("T", "O", "P")), default=float("nan"))
            sup_fac = maxeff < 0.5 * abs(anc["A_lo"])
            if elig and len(inband) == len(elig) and sup_fac:
                sup.append(mdl)
            print(f"    {mdl:9} conds={len(conds):2} CLEAN={len(clean):2} "
                  f"CLEAN|COMPRESSED={len(cc):2} in-band(S_lo)={len(inband)}/{len(elig)} "
                  f"max|dT,dO,dP|={maxeff:.3f} vs 0.5*|A_lo(anchor)|="
                  f"{0.5*abs(anc['A_lo']):.3f}")
        print(f"    F1 (magnitude, CLEAN only): {len(f1)}/{n} models"
              f"{'  - CANNOT FIRE, no CLEAN cells exist' if n_clean == 0 else ''}")
        print(f"    F2 (sign reversal):         {len(f2)}/{n} models "
              f"(scale-invariant by construction)")
        print(f"    SUPPORT (all S_lo in band + factorial): {len(sup)}/{n} models")

        if n == 0:
            v = "INCONCLUSIVE - no surviving model"
        elif n == 1:
            v = (f"INCONCLUSIVE - only {surviving[0]} survives; "
                 f"section 5.3 forbids a verdict")
        elif len(f1) >= majority:
            v = f"FALSIFIED by F1 ({len(f1)} of {n})"
        elif len(f2) >= majority:
            v = f"FALSIFIED by F2 ({len(f2)} of {n})"
        elif len(sup) >= majority:
            v = f"SUPPORTED ({len(sup)} of {n})"
        elif n == 2 and (len(f1) == 1 or len(f2) == 1 or len(sup) == 1):
            v = "SPLIT -> UNRESOLVED (section 5.3, 1 of 2, never rounded)"
        else:
            v = "PARTIAL - neither F1, F2 nor SUPPORT reached a strict majority"
        verdicts[opp] = v
        print(f"    ---> {v}")

    print(f"\n{RULE}")
    print("  REGISTERED VERDICT ON THE LOG-ODDS SCALE (TFT decides, section 5.2)")
    print(f"    TFT : {verdicts['tft']}")
    print(f"    ALLC: {verdicts['allc']}   (reported without a vote)")
    print(RULE)

    ser = {"|".join(k): {kk: vv for kk, vv in c.items() if not kk.startswith("_")}
           for k, c in cells.items()}
    Path(args.out).write_text(json.dumps(dict(
        seed=SEED, bootstrap=args.bootstrap, band=[BAND_LO, BAND_HI],
        anchor_guard=ANCHOR_GUARD, anchor_guard_scale="probability (as registered)",
        turn0_excluded=True, zero_treatment=dict(
            method="Haldane-Anscombe", h=H, applied_to="turn counts (N=n_ep*19)",
            uniform=True,
            sensitivity_1="Haldane-Anscombe on episode counts (N=n_ep)",
            sensitivity_2="drop bootstrap draws with a zero rate"),
        reproduces_analysis_15=verify_msg, n_clean_cells=n_clean,
        scales_disagree_on_band=["|".join(k) for k in disagree],
        verdicts_logodds=verdicts, factorial_logodds=factorial, cells=ser), indent=2))
    print(f"\n  json -> {args.out}")
    return 0


# section 4's design matrix, identical to analysis/15's
CODING = {
    "anchor":       (-1, -1, -1), "origp2":       (-1, -1, +1),
    "origpermp1":   (-1, +1, -1), "origpermp2":   (-1, +1, +1),
    "rewordp1":     (+1, -1, -1), "rewordp2":     (+1, -1, +1),
    "rewordpermp1": (+1, +1, -1), "rewordpermp2": (+1, +1, +1),
}
TERMS = {"T": lambda t, o, q: t, "O": lambda t, o, q: o,
         "P": lambda t, o, q: q, "TO": lambda t, o, q: t * o,
         "TP": lambda t, o, q: t * q, "OP": lambda t, o, q: o * q,
         "TOP": lambda t, o, q: t * o * q}


if __name__ == "__main__":
    raise SystemExit(main())
