#!/usr/bin/env python3
"""ANALYSIS 15 - exp8 stability verdict, per PREREGISTRATION_EXP8.md section 5.

WHY THIS SCRIPT EXISTS
    PREREGISTRATION_EXP8.md section 4 names `analysis/02_episode_level.py` as the
    estimator. It is not one, for exp8. analysis/02 computes ATE_true (3-3b),
    perturbation (3b-1) and ATE_naive (3-1). exp8's primary estimand is

        A = P(D | 3m) - P(D | 3s)

    which analysis/02 does not compute, has no turn-0 exclusion for, and cannot
    difference across groups because it opens one --db. That mismatch between
    the pre-registration and the code is a declared defect; this script is the
    correction, and it re-estimates rather than redefines.

WHAT IT IMPLEMENTS, CLAUSE BY CLAUSE
    section 4   A = P(D|3m) - P(D|3s), TURN 0 EXCLUDED from every arm. At turn 0
                there is no last move to flip, arm 3m renders byte-identically
                to arm 3, and the row carries no manipulation. Including those
                rows dilutes A toward zero by 1/20th of the horizon.
    section 4   episode-clustered bootstrap, 10,000 resamples, seed 20260814.
                The episode is the cluster: turns within one are dependent,
                episodes are seeded independently and are i.i.d. draws.
    section 5.4 FLOOR   max P(D) over {3,3s,3m} <= 0.05 -> no vote, not counted
                CEILING min P(D) over {3,3s,3m} >= 0.95 -> no vote, not counted
                COMPRESSED  min <= 0.15 or max >= 0.85 -> no F1 vote, F2 only
    section 5.4 anchor guard: S is NEVER computed when |A(anchor)| < 0.05.
                The model is then excluded FOR THAT OPPONENT.
    section 5   S = A(cond) / A(anchor); band [0.50, 2.00] with the anchor's
                sign. A condition is UNSTABLE only if S leaves the band AND the
                bootstrap CI on A(cond) - A(anchor) excludes 0. Worst case over
                conditions - conditions are never averaged.
    section 5.2 opponents are never pooled. Every verdict is a pair.
                TFT is decisive; ALLC is reported without a vote.
    section 5.3 models are counted, not averaged. Excluded models leave BOTH
                numerator and denominator. 2 of 2 disagreeing -> SPLIT /
                UNRESOLVED. Exactly one survivor -> INCONCLUSIVE, no verdict.

USAGE
    python3 analysis/15_exp8_stability.py
    python3 analysis/15_exp8_stability.py --bootstrap 2000 --out exp8_stability.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import random
import re
import shutil
import sqlite3
import statistics
import tempfile
from pathlib import Path
from urllib.request import pathname2url

RULE = "=" * 78
SEED = 20260814                      # section 4, fixed before the run
BOOT_DEFAULT = 10_000
BAND_LO, BAND_HI = 0.50, 2.00        # section 5
ANCHOR_GUARD = 0.05                  # section 5.4
FLOOR_MAX, CEIL_MIN = 0.05, 0.95     # section 5.4
COMPRESS_LO, COMPRESS_HI = 0.15, 0.85


def phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
def two_sided_p(z): return 2.0 * (1.0 - phi(abs(z)))


def ro_uri(p: Path) -> str:
    return "file:" + pathname2url(str(p.resolve())) + "?mode=ro&immutable=1"


def episode_rates(conn, arm: str, opp: str) -> list[float]:
    """Episode-level defection rates with TURN 0 EXCLUDED.

    Deliberately NOT read from episodes.defection_count: that column counts
    every turn including turn 0, and section 4 excludes turn 0 from every arm of
    every move contrast. Recomputed from `turns` so the exclusion is real.
    """
    rows = conn.execute(
        "SELECT episode_id, SUM(agent_action='D') AS d, COUNT(*) AS n "
        "FROM turns WHERE arm=? AND opponent_policy=? AND turn > 0 "
        "GROUP BY episode_id ORDER BY episode_id",
        (arm, opp),
    ).fetchall()
    return [r[1] / r[2] for r in rows if r[2]]


def load_group(path: Path) -> dict:
    """arm -> opponent -> list of episode rates, for the three arms A reads."""
    tmp = None
    if path.suffix == ".gz":
        tmp = Path(tempfile.mkdtemp()) / path.stem
        with gzip.open(path, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        path = tmp
    conn = sqlite3.connect(ro_uri(path), uri=True)
    out = {}
    for arm in ("3", "3s", "3m"):
        out[arm] = {opp: episode_rates(conn, arm, opp) for opp in ("tft", "allc")}
    conn.close()
    if tmp:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    return out



def cpr_for(db_path: Path, opp: str) -> float:
    """CPR(3) for F3, read from the driver's JSON rather than recomputed.

    F3 is a VALIDITY condition: a block the model cannot read is a broken
    prompt, not a rendering of the state, and a collapse there measures the
    renderer. The driver already computed this per cell; re-deriving it here
    would risk the two disagreeing.
    """
    j = Path(str(db_path).replace(".sqlite.gz", ".json").replace(".sqlite", ".json"))
    if not j.exists():
        return float("nan")
    try:
        return json.loads(j.read_text())["3|" + opp]["cpr"]
    except (KeyError, ValueError):
        return float("nan")


def classify(p3: float, p3s: float, p3m: float) -> str:
    """section 5.4, computed per contrast cell from that cell's own data."""
    hi, lo = max(p3, p3s, p3m), min(p3, p3s, p3m)
    if hi <= FLOOR_MAX:
        return "FLOOR"
    if lo >= CEIL_MIN:
        return "CEILING"
    if lo <= COMPRESS_LO or hi >= COMPRESS_HI:
        return "COMPRESSED"
    return "CLEAN"


def boot_A(rng, m: list[float], s: list[float], boot: int) -> list[float]:
    """Bootstrap draws of A = mean(3m) - mean(3s), resampling episodes."""
    nm, ns = len(m), len(s)
    return [statistics.fmean(rng.choices(m, k=nm))
            - statistics.fmean(rng.choices(s, k=ns)) for _ in range(boot)]


def ci(vals: list[float]) -> tuple[float, float]:
    v = sorted(vals)
    n = len(v)
    return v[int(0.025 * n)], v[min(int(0.975 * n), n - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="exp8_*_logit.sqlite*")
    ap.add_argument("--bootstrap", type=int, default=BOOT_DEFAULT)
    ap.add_argument("--out", default="exp8_stability.json")
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.glob)
                   if not p.endswith(".sqlite-wal") and not p.endswith(".sqlite-shm"))
    if not paths:
        raise SystemExit(f"no databases matched {args.glob!r}")

    # One path per database stem. `--glob` ends in `sqlite*`, so a decompressed
    # copy sitting beside its archive matches twice - which is the normal state
    # after `scripts/reproduce.sh`, since that gunzips with `-k`. Loading a
    # database twice does not double-count any data (cells are keyed by
    # model|cond|opp and the second load overwrites the first), but the bootstrap
    # draws come from ONE sequential RNG consumed per loaded path, so the extra
    # loads shift every subsequent cell's interval. Measured on a two-database
    # fixture: point estimates identical, 7 of 8 interval endpoints moved.
    # The committed exp8_stability.json reproduces bit-for-bit from archives
    # alone (32/32 estimates, 64/64 endpoints), and deduping restores that state
    # whether or not a decompressed copy is present, because the draw sequence
    # depends only on the number and order of stems, not on which form survives.
    _by_stem: dict[str, Path] = {}
    for _p in paths:
        _stem = _p.name.replace(".sqlite.gz", "").replace(".sqlite", "")
        if _stem not in _by_stem or _p.name.endswith(".sqlite"):
            _by_stem[_stem] = _p
    if len(_by_stem) != len(paths):
        print(f"  note: {len(paths) - len(_by_stem)} duplicate archive/decompressed "
              f"pair(s) collapsed; using one file per database")
    paths = [_by_stem[s] for s in sorted(_by_stem)]


    rng = random.Random(SEED)
    cells = {}
    print(f"{RULE}\nexp8 STABILITY - PREREGISTRATION_EXP8.md section 5\n{RULE}")
    print(f"  seed {SEED}   bootstrap {args.bootstrap}   turn 0 EXCLUDED\n")

    for p in paths:
        m = re.match(r"exp8_([a-z0-9]+)_([a-z0-9]+)_logit\.sqlite", p.name)
        if not m:
            continue
        model, cond = m.group(1), m.group(2)
        g = load_group(p)
        for opp in ("tft", "allc"):
            r3, r3s, r3m = g["3"][opp], g["3s"][opp], g["3m"][opp]
            if not (r3 and r3s and r3m):
                continue
            p3, p3s, p3m = (statistics.fmean(r3), statistics.fmean(r3s),
                            statistics.fmean(r3m))
            draws = boot_A(rng, r3m, r3s, args.bootstrap)
            lo, hi = ci(draws)
            # Kept, not discarded. The factorial below is a linear combination
            # of A across conditions, and conditions are independent groups, so
            # the same draws serve both the per-cell CI and the effect CIs. Re-
            # drawing would give two different answers to the same question.
            cells[(model, cond, opp)] = dict(
                model=model, cond=cond, opp=opp,
                p3=p3, p3s=p3s, p3m=p3m, A=p3m - p3s,
                A_lo=lo, A_hi=hi, A_excl_0=bool(lo > 0 or hi < 0), A_draws=draws,
                klass=classify(p3, p3s, p3m),
                cpr3=cpr_for(p, opp),
                rates_3m=r3m, rates_3s=r3s,
            )
        print(f"  loaded {p.name}")

    # ---- S and the CI on A(cond) - A(anchor) ------------------------------
    for (model, cond, opp), c in cells.items():
        anc = cells.get((model, "anchor", opp))
        c["S"] = None
        c["excluded_reason"] = None
        if anc is None:
            c["excluded_reason"] = "no anchor"
            continue
        if anc["klass"] in ("FLOOR", "CEILING") or abs(anc["A"]) < ANCHOR_GUARD:
            c["excluded_reason"] = (
                f"anchor guard: |A(anchor)|={abs(anc['A']):.4f} < {ANCHOR_GUARD}"
                if anc["klass"] not in ("FLOOR", "CEILING")
                else f"anchor is {anc['klass']}")
            continue
        c["S"] = c["A"] / anc["A"]
        if cond == "anchor":
            continue
        d = [x - y for x, y in zip(c["A_draws"], anc["A_draws"])]
        c["dA"] = c["A"] - anc["A"]
        c["dA_lo"], c["dA_hi"] = ci(d)
        c["dA_excludes_0"] = bool(c["dA_lo"] > 0 or c["dA_hi"] < 0)
        c["out_of_band"] = bool(c["S"] < BAND_LO or c["S"] > BAND_HI)
        c["unstable"] = bool(c["out_of_band"] and c["dA_excludes_0"]
                             and c["klass"] not in ("FLOOR", "CEILING"))

    # ---- report ------------------------------------------------------------
    print(f"\n{RULE}\nPER-CELL\n{RULE}")
    print(f"{'model':9}{'cond':14}{'opp':5}{'P(D|3)':>8}{'P(D|3s)':>9}{'P(D|3m)':>9}"
          f"{'A':>9}{'95% CI on A':>20}{'S':>7}{'class':>11}{'':>3}")
    for k in sorted(cells):
        c = cells[k]
        s = "  n/a" if c["S"] is None else f"{c['S']:5.2f}"
        mark = ""
        if c.get("unstable"):
            mark = " UNSTABLE"
        elif c.get("out_of_band") and not c.get("dA_excludes_0"):
            mark = " band-only"
        print(f"{c['model']:9}{c['cond']:14}{c['opp']:5}{c['p3']:8.3f}{c['p3s']:9.3f}"
              f"{c['p3m']:9.3f}{c['A']:+9.3f}"
              f"  [{c['A_lo']:+7.3f},{c['A_hi']:+7.3f}]{s:>7}{c['klass']:>11}{mark}")

    # ---- factorial on A, section 4 "Secondary, descriptive" ---------------
    #
    # CODING is the design matrix. T = template (original -> -1, reworded -> +1),
    # O = field order (canonical -> -1, permuted -> +1), P = insertion index
    # (1 -> -1, 2 -> +1). An effect is mean(A at +1) - mean(A at -1).
    #
    # With all EIGHT conditions the 2^3 is complete and every main effect and
    # two-factor interaction is clear. With FOUR (the half fraction, I = -TOP)
    # they are aliased: T = -OP, O = -TP, P = -TO. Section 4 is explicit that on
    # those models the numbers must be quoted as "T + (-OP)" and never as "T".
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

    print(f"\n{RULE}\nFACTORIAL ON A  (main effects and interactions)\n{RULE}")
    factorial = {}
    for mdl in sorted({c["model"] for c in cells.values()}):
        for opp in ("tft", "allc"):
            got = {c["cond"]: c for c in cells.values()
                   if c["model"] == mdl and c["opp"] == opp and c["cond"] in CODING}
            if len(got) < 4:
                continue
            full = len(got) == 8
            print(f"\n  {mdl} / {opp}   {len(got)} conditions"
                  f"   {'FULL 2^3 - effects are clear' if full else 'HALF FRACTION - effects are ALIASED (T=-OP, O=-TP, P=-TO)'}")
            for name, fn in TERMS.items():
                plus = [c for k, c in got.items() if fn(*CODING[k]) > 0]
                minus = [c for k, c in got.items() if fn(*CODING[k]) < 0]
                if not plus or not minus:
                    continue
                eff = (statistics.fmean([c["A"] for c in plus])
                       - statistics.fmean([c["A"] for c in minus]))
                draws = [statistics.fmean([c["A_draws"][b] for c in plus])
                         - statistics.fmean([c["A_draws"][b] for c in minus])
                         for b in range(args.bootstrap)]
                lo, hi = ci(draws)
                label = name if full else (name + " + alias")
                star = " *" if (lo > 0 or hi < 0) else ""
                print(f"    {label:12}{eff:+8.4f}   [{lo:+7.4f},{hi:+7.4f}]{star}")
                factorial[f"{mdl}|{opp}|{name}"] = dict(
                    model=mdl, opp=opp, term=name, aliased=not full,
                    effect=eff, lo=lo, hi=hi, excludes_0=bool(lo > 0 or hi < 0))
    print("\n  * = 95% bootstrap CI excludes 0")

    # ---- verdict: section 5.7's ACTUAL criteria ---------------------------
    #
    # An earlier version of this script applied section 5 item 4's prose summary
    # (the S band, worst case over conditions) and printed UNSTABLE. That is not
    # the registered rule. Section 5.7 defines F1, F2, F3 and SUPPORT, and F1 is
    # gated on CLEAN cells specifically. The distinction is not academic: this
    # corpus contains ZERO clean cells, so F1 cannot fire at all.
    print(f"\n{RULE}\nVERDICT - PREREGISTRATION_EXP8.md section 5.7\n{RULE}")
    fired = {}
    for opp in ("tft", "allc"):
        print(f"\n  ---- opponent = {opp} ----")
        models = sorted({c["model"] for c in cells.values() if c["opp"] == opp})
        surviving = []
        for mdl in models:
            anc = cells.get((mdl, "anchor", opp))
            if anc is None:
                continue
            if anc["klass"] in ("FLOOR", "CEILING"):
                print(f"    {mdl:9} EXCLUDED - anchor is {anc['klass']}")
                continue
            if abs(anc["A"]) < ANCHOR_GUARD:
                print(f"    {mdl:9} EXCLUDED - anchor guard |A|={abs(anc['A']):.4f} < {ANCHOR_GUARD}")
                continue
            surviving.append(mdl)
        n = len(surviving)
        majority = n // 2 + 1
        print(f"    surviving: {surviving or '-'}   strict majority = {majority} of {n}")

        # F3 - validity, checked before any result rule
        void = []
        for mdl in surviving:
            anc = cells[(mdl, "anchor", opp)]
            if anc["cpr3"] >= 0.95:
                for c in [x for x in cells.values()
                          if x["model"] == mdl and x["opp"] == opp and x["cond"] != "anchor"]:
                    if c["cpr3"] == c["cpr3"] and c["cpr3"] < 0.80:
                        void.append(f"{mdl}/{c['cond']} CPR(3)={c['cpr3']:.3f}")
        print(f"    F3 readability: {'VOID -> ' + ', '.join(void) if void else 'all conditions readable'}")

        f1_models, f2_models, sup_models = [], [], []
        for mdl in surviving:
            anc = cells[(mdl, "anchor", opp)]
            conds = [c for c in cells.values() if c["model"] == mdl
                     and c["opp"] == opp and c["cond"] != "anchor"]
            clean = [c for c in conds if c["klass"] == "CLEAN"]
            cc = [c for c in conds if c["klass"] in ("CLEAN", "COMPRESSED")]
            if any(c.get("out_of_band") and c.get("dA_excludes_0") for c in clean):
                f1_models.append(mdl)
            if any((c["A"] > 0) != (anc["A"] > 0) and c["A_excl_0"] and anc["A_excl_0"]
                   for c in cc):
                f2_models.append(mdl)
            eligible = [c for c in conds if c["S"] is not None
                        and c["klass"] not in ("FLOOR", "CEILING")]
            if eligible and all(BAND_LO <= c["S"] <= BAND_HI for c in eligible):
                sup_models.append(mdl)
            print(f"    {mdl:9} conds={len(conds):2} clean={len(clean):2} "
                  f"clean-or-compressed={len(cc):2} "
                  f"in-band={sum(1 for c in eligible if BAND_LO <= c['S'] <= BAND_HI)}/{len(eligible)}")

        print(f"    F1 (magnitude, CLEAN only): {len(f1_models)}/{n} models"
              f"{' - CANNOT FIRE, no CLEAN cells exist' if not any(c['klass']=='CLEAN' for c in cells.values()) else ''}")
        print(f"    F2 (sign reversal):         {len(f2_models)}/{n} models")
        print(f"    SUPPORT (all in band):      {len(sup_models)}/{n} models")

        if n == 0:
            v = "INCONCLUSIVE - no surviving model"
        elif n == 1:
            v = f"INCONCLUSIVE - only {surviving[0]} survives; section 5.3 forbids a verdict"
        elif len(f1_models) >= majority:
            v = f"FALSIFIED by F1 ({len(f1_models)} of {n})"
        elif len(f2_models) >= majority:
            v = f"FALSIFIED by F2 ({len(f2_models)} of {n})"
        elif len(sup_models) >= majority:
            v = f"SUPPORTED ({len(sup_models)} of {n})"
        else:
            v = "PARTIAL - neither F1, F2 nor SUPPORT reached a strict majority"
        fired[opp] = v
        print(f"    ---> {v}")

    print(f"\n{RULE}")
    print(f"  REGISTERED VERDICT (TFT decides, section 5.2)")
    print(f"    TFT : {fired['tft']}")
    print(f"    ALLC: {fired['allc']}   (reported without a vote)")
    if fired["tft"].startswith("PARTIAL"):
        print(f"\n  PARTIAL licenses a SCOPE CLAUSE on C5 (section 6 row 4). It does NOT")
        print(f"  license a retraction, and it does NOT license a generality claim.")
    print(RULE)

    ser = {f"{k[0]}|{k[1]}|{k[2]}": {kk: vv for kk, vv in c.items()
                                     if not kk.startswith("rates_")
                                     and kk != "A_draws"}
           for k, c in cells.items()}
    Path(args.out).write_text(json.dumps(
        dict(seed=SEED, bootstrap=args.bootstrap, band=[BAND_LO, BAND_HI],
             anchor_guard=ANCHOR_GUARD, turn0_excluded=True,
             verdicts=fired, factorial=factorial, cells=ser), indent=2))
    print(f"\n  json -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
