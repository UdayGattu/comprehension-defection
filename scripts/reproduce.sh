#!/usr/bin/env bash
# Reproduce every quoted number from the released databases. CPU only.
#
#     pip install -r requirements.txt
#     bash scripts/reproduce.sh
#
# No GPU, no model weights, no HF token, no network. Nothing here re-runs
# inference; every database is opened read-only (mode=ro&immutable=1) by the
# analysis scripts, and the frozen archives are never touched.
#
# ---------------------------------------------------------------------------
# THE DEPENDENCY ORDER, AND WHY IT IS THIS ORDER
# ---------------------------------------------------------------------------
#   gunzip                     the archives are what is committed; the .sqlite
#                              files are gitignored and regenerable.
#   02_episode_level.py        the ONLY source of intervals in this project.
#                              Everything downstream reads its output rather
#                              than recomputing, so a number cannot disagree
#                              with itself between documents. Must run per
#                              database, before 06.
#   06_evidence.py             point estimates by SELECT over every *.sqlite in
#                              the CURRENT DIRECTORY, plus the intervals 02
#                              wrote. Emits EVIDENCE.md and EVIDENCE_cells.csv.
#   07_cross_experiment.py     joins EVIDENCE_cells.csv across databases into
#                              the cross-file tables (readout ladder, stack
#                              drift, lexical test). Needs 06's CSV to exist.
#   13_exp6_fields.py          exp6 field decomposition - the headline result.
#                              Independent of 06/07, needs numpy, and needs the
#                              exp6_*.sqlite files unzipped.
#
# ---------------------------------------------------------------------------
# THE NAMING TRAP  --  read this before "fixing" the --out flags below
# ---------------------------------------------------------------------------
# analysis/06_evidence.py looks for a per-database interval file named
# EXACTLY `ep_<stem>.json`, where <stem> is the database filename without
# `.sqlite` (see `load_ci()`, analysis/06_evidence.py). If it is not there, 06
# prints a remediation hint:
#
#     _No `ep_<stem>.json`; run `analysis/02_episode_level.py --db <stem>.sqlite`
#      for intervals._
#
# That hint is WRONG, and wrong in a way that is self-concealing. It omits
# `--out`, and `--out` defaults to `episode_level.json` - a single, stem-less
# filename. So following the hint literally:
#
#     * writes `episode_level.json`, overwriting it once per database, so only
#       the last database's intervals survive;
#     * leaves `ep_<stem>.json` still absent;
#     * makes 06 print the same "no intervals" line on the next run.
#
# The operator concludes the databases are broken. They are not. The fix is one
# flag, and this script always passes it. Do not remove `--out` below.
#
# (The hint lives in analysis/06_evidence.py, which is frozen against the
# databases' git_commit provenance and is deliberately not edited here.)

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY=${PYTHON:-python3}
BOOTSTRAP=${BOOTSTRAP:-10000}   # 10,000 is what every quoted interval used
RULE="=============================================================="

banner() { echo; echo "$RULE"; echo "  $*"; echo "$RULE"; }

# ---------------------------------------------------------------- preflight

banner "PREFLIGHT"

command -v "$PY" >/dev/null || { echo "ABORT: $PY not found."; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "ABORT: Python 3.10+ required (got $("$PY" -V 2>&1))."; exit 1; }

# numpy is a hard requirement of analysis/13 (it calls sys.exit at import) and a
# 50x speedup for 08/11. Checked here so the failure is at second zero rather
# than after an hour of bootstrapping.
"$PY" -c 'import numpy' 2>/dev/null || {
    echo "ABORT: numpy missing. analysis/13_exp6_fields.py exits at import"
    echo "       without it.  pip install -r requirements.txt"; exit 1; }

command -v gunzip >/dev/null || { echo "ABORT: gunzip not found."; exit 1; }

echo "  python      $("$PY" -V 2>&1)"
echo "  numpy       $("$PY" -c 'import numpy; print(numpy.__version__)')"
echo "  cwd         $(pwd)"
echo "  bootstrap   ${BOOTSTRAP} resamples"

# ---------------------------------------------------------------- 0. gunzip

banner "STEP 0  decompress"

shopt -s nullglob
archives=(*.sqlite.gz)
if [ ${#archives[@]} -eq 0 ]; then
    echo "ABORT: no *.sqlite.gz in $(pwd)."
    echo "       The databases are committed to the repository root. If this is"
    echo "       a shallow or partial clone, or git-lfs has not fetched them,"
    echo "       nothing below can run."
    exit 1
fi

for gz in "${archives[@]}"; do
    db="${gz%.gz}"
    if [ -f "$db" ] && [ "$db" -nt "$gz" ]; then
        echo "  keep    $db (newer than archive)"
    else
        gunzip -kf "$gz"
        echo "  gunzip  $db"
    fi
done

# The uncompressed copies are gitignored (*.sqlite) and are safe to delete
# afterwards; the archives beside them are the artefact of record.

# ------------------------------------------------- 1. episode-level intervals

banner "STEP 1  episode-level re-estimation  (analysis/02)"

echo "  Episode-level bootstrap. This is the ONLY source of intervals in the"
echo "  project; turn-level intervals misstate width by 0.46x-4.07x across"
echo "  exp2-exp5 and are never quoted (CLAIMS.md B3)."
echo

failed_02=()
for db in *.sqlite; do
    stem="${db%.sqlite}"
    case "$stem" in
        smoke_*|cotsmoke_*)
            echo "  skip    $db  (N=4 instrument check, not a measurement)"
            continue ;;
    esac

    # --out is NOT optional. See THE NAMING TRAP at the top of this file.
    if "$PY" analysis/02_episode_level.py \
            --db "$db" \
            --out "ep_${stem}.json" \
            --bootstrap "$BOOTSTRAP" > "ep_${stem}.log" 2>&1; then
        echo "  ok      ep_${stem}.json"
    else
        echo "  FAILED  $db  (see ep_${stem}.log)"
        failed_02+=("$db")
    fi
done

if [ ${#failed_02[@]} -gt 0 ]; then
    echo
    echo "  ${#failed_02[@]} database(s) failed step 1: ${failed_02[*]}"
    echo "  Continuing - 06 marks any database with no ep_*.json rather than"
    echo "  crashing, so the run still produces a usable evidence file. Read"
    echo "  the .log files before quoting anything from those databases."
fi

# ------------------------------------------------------------- 2. evidence

banner "STEP 2  evidence file  (analysis/06)"

"$PY" analysis/06_evidence.py --out EVIDENCE.md --csv EVIDENCE_cells.csv
echo "  wrote  EVIDENCE.md, EVIDENCE_cells.csv"

# --------------------------------------------------------- 3. cross-experiment

banner "STEP 3  cross-experiment joins  (analysis/07)"

[ -f EVIDENCE_cells.csv ] || {
    echo "ABORT: EVIDENCE_cells.csv missing; step 2 did not complete."; exit 1; }

"$PY" analysis/07_cross_experiment.py \
    --csv EVIDENCE_cells.csv --out CROSS_EXPERIMENT.md
echo "  wrote  CROSS_EXPERIMENT.md"

# ------------------------------------------------------- 4. exp6 field split

banner "STEP 4  exp6 field decomposition  (analysis/13)"

exp6=(exp6_*.sqlite)
if [ ${#exp6[@]} -eq 0 ]; then
    echo "  SKIPPED: no exp6_*.sqlite present."
    echo "  This is the headline result (CLAIMS.md C5, C6, C7, C8). If the exp6"
    echo "  archives are in the repository, step 0 failed to decompress them."
else
    "$PY" analysis/13_exp6_fields.py \
        --glob 'exp6_*.sqlite' \
        --bootstrap "$BOOTSTRAP" \
        --out EXP6_FIELDS.json | tee EXP6_FIELDS.txt
    echo "  wrote  EXP6_FIELDS.json, EXP6_FIELDS.txt"
fi

# ------------------------------------------------------------------ step 5

banner "STEP 5  exp8 cross-configuration stability  (analysis/15, analysis/16)"

# These two produce the paper's second result -- the cross-configuration
# stability study -- and were previously absent from this script, so a reader
# following it could not regenerate Table 5 or Table 6.
#
# They read `.sqlite.gz` transparently, so they need none of the gunzip above.
# That matters: this script gunzips with `-k`, leaving a decompressed copy
# beside every archive, and their `--glob` ends in `sqlite*`, which matches
# both. Both scripts now collapse each archive/decompressed pair to one path.
# Without that, the duplicate consumes draws from the single sequential RNG and
# every interval shifts -- measured on a two-database fixture: point estimates
# identical, 7 of 8 interval endpoints moved.
#
# analysis/16 re-derives analysis/15's probability-scale draws and aborts if
# they do not reproduce bit-for-bit, so running them in this order is a check,
# not just a sequence.

if ls exp8_*_logit.sqlite.gz >/dev/null 2>&1; then
    "$PY" analysis/15_exp8_stability.py --out exp8_stability.json \
        | tee exp8_stability.txt
    echo "  wrote  exp8_stability.json, exp8_stability.txt"
    "$PY" analysis/16_exp8_logodds.py --out exp8_logodds.json \
        --verify exp8_stability.json | tee exp8_logodds.txt
    echo "  wrote  exp8_logodds.json, exp8_logodds.txt"
else
    echo "  SKIP: no exp8_*_logit.sqlite.gz in the working directory"
fi

# ------------------------------------------------------------------ summary

banner "COMPLETE"

cat <<'EOF'
  Produced:
    ep_<stem>.json      episode-level bootstrap intervals, one per database
    ep_<stem>.log       the full printed output of each analysis/02 run
    EVIDENCE.md         every point estimate, by SELECT, with provenance
    EVIDENCE_cells.csv  the same, machine-readable; input to analysis/07
    CROSS_EXPERIMENT.md readout ladder, stack drift, lexical test
    EXP6_FIELDS.json    exp6 field decomposition (score vs last move)
    exp8_stability.json exp8 cross-configuration stability, probability scale
    exp8_logodds.json   the same on the registered log-odds scale (Tables 5, 6)
    EXP6_FIELDS.txt     its printed form

  CLAIM_MAP.md maps every claim id in CLAIMS.md to the file, the command and
  the expected value. Check the numbers against it; a disagreement is a
  finding, not a rounding difference.

  Optional, not on the critical path:
    analysis/03_scorer_audit.py     --db sweep.sqlite       exp1 scorer defect
    analysis/04_donor_echo.py       --db <exp3 db>          block-reading (C1)
    analysis/08_decomposition_ci.py                         content/schema split
    analysis/09_dose_response.py                            lie size vs effect
    analysis/10_rescore_swap.py                             swap-label rescore
    analysis/11_stratified_donor.py                         09's confound removed
    analysis/12_exp6_prerequisites.py                       the exp6 design gates
    analysis/14_reviewer_responses.py                       eight referee
                                                            objections, answered
                                                            from committed data.
                                                            Read check A first:
                                                            along a cooperative
                                                            trajectory TFT and
                                                            ALLC are
                                                            observationally
                                                            identical, which
                                                            bears directly on
                                                            how the
                                                            pre-registered
                                                            rejection reads.

  The *.sqlite files are gitignored and regenerable; delete them when done:
    rm -f *.sqlite
EOF
