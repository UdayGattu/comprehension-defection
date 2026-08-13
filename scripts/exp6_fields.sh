#!/usr/bin/env bash
# EXPERIMENT 6 - which FIELD of the state block carries the effect?
#
# THE QUESTION
#   Arm 3c replaces the whole [STATE] block with another episode's, so when it
#   produced a 24pp swing in qwen-vs-TFT - replicated at -0.2407 in exp2 and
#   -0.2375 in exp3 - the field responsible could not be identified.
#
#   Two measurements since then say it was not the score:
#
#     analysis/12   the score falsification arm 3c actually delivers is tiny.
#                   In exp3_qwen_sem vs TFT, sd(d) = 3.30 and only 0.1% of rows
#                   carry |d| >= 15. At the measured ~0.01 defection per point
#                   that predicts a 3pp shift. The observed effect is 24pp.
#
#     analysis/08   the content effect is larger vs TFT than vs ALLC in 8 of 9
#                   cells. Under ALLC the opponent's last move is ALWAYS
#                   Cooperate, so a donor from the same cell shows Cooperate too
#                   and that field CANNOT be falsified. Under TFT it mirrors the
#                   agent and is falsified constantly.
#
#   Both point at "Opponent's last move". This run tests it directly.
#
# WHY THIS AND NOT A NEW OPPONENT
#   Three opponent designs were rejected first. Threshold-defector: exhaustive
#   search shows "cooperate once then defect forever" optimal at every X.
#   Lead-guard: 75,582 sequences tie for the optimum and a policy reading only
#   the round index attains it exactly. Both die to the same theorem - a
#   DETERMINISTIC opponent from a fixed start makes the open-loop optimum equal
#   the closed-loop one, so state-tracking is never payoff-necessary. A fourth
#   scripted opponent would fail identically.
#
#   The last-move field sidesteps the objection those designs existed to answer.
#   The standing critique is that cumulative score is a SUNK variable against
#   TFT and ALLC, so falsifying it should change nothing. That is true of the
#   score and false of the last move, which is the entire input to optimal play
#   against TFT.
#
# THE ARMS
#   1    no block
#   3b   placebo, token- and density-matched
#   3    true state
#   3s   ONLY "Your score" wrong, by +/-15          <- new
#   3m   ONLY "Opponent's last move" flipped        <- new
#   3c   both wrong (donor), the existing anchor
#
#   Falsification is set DELIBERATELY, not sampled from donors, because
#   analysis/12 showed donor sampling delivers +/-3 in the cell where the effect
#   is largest - never a magnitude capable of moving a decision.
#
#   PREDICTION: 3m carries the effect, 3s is null. If it comes out the other way
#   the last-move account is wrong and that is the finding.
#
# ARM 3m vs ALLC IS THE CONDITION THE CORPUS CANNOT PRODUCE
#   Arm 3c cannot falsify the move there - every donor also shows Cooperate. 3m
#   flips it deliberately, so the block asserts a betrayal while the [HISTORY]
#   section immediately below lists an unbroken run of cooperation. The lie and
#   its refutation sit in the same context window. That is the cleanest
#   context-poisoning test available here, and the one that generalises past
#   game theory: when a state summary contradicts the raw log, which wins?
#
# TWO PHASES
#   LOGIT       all six arms. Every content effect measured so far is a LOGIT
#               effect, so this is where the prediction lives.
#   SCRATCHPAD  arms 1, 3b, 3, 3m only, minimal prompt. Not for the number - for
#               the REASONING. Under 3m-vs-ALLC the traces show whether the
#               model notices the contradiction and which source it trusts.
#               Arms chosen so perturbation and the 3m contrast both still
#               compute.
#
#   MODE=logit bash scripts/exp6_fields.sh     # ~35 min
#   MODE=both  bash scripts/exp6_fields.sh     # ~1.5 h

set -euo pipefail

DB_FS=$(df -T . 2>/dev/null | tail -1 | awk '{print $2}')
case "$DB_FS" in
    fuse*|nfs*|*moose*|*mfs*|9p)
        echo "ABORT: working directory is on '$DB_FS'. SQLite WAL is"
        echo "       unsupported there. Run from local disk (/root)."
        exit 1 ;;
esac

MODE=${MODE:-both}
EPISODES=${EPISODES:-1000}
LOGIT_BUDGET=${LOGIT_BUDGET:-30}
PAD_BUDGET=${PAD_BUDGET:-90}
SCRATCH_TOKENS=${SCRATCH_TOKENS:-128}
TAG=exp6
LOG=exp6_session.log
RULE="=============================================================="

LOGIT_ARMS="1 3b 3 3s 3m 3c"
PAD_ARMS="1 3b 3 3m"

MODELS=(
  "llama:meta-llama/Llama-3.1-8B-Instruct:models--meta-llama--Llama-3.1-8B-Instruct"
  "qwen:Qwen/Qwen2.5-7B-Instruct:models--Qwen--Qwen2.5-7B-Instruct"
  "mistral:mistralai/Mistral-7B-Instruct-v0.3:models--mistralai--Mistral-7B-Instruct-v0.3"
)

banner() { echo | tee -a "$LOG"; echo "$RULE" | tee -a "$LOG"; \
           echo "  $*" | tee -a "$LOG"; echo "$RULE" | tee -a "$LOG"; }

banner "PREFLIGHT  mode=$MODE  episodes=$EPISODES"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
python -c "import vllm; print('vllm', vllm.__version__)" | tee -a "$LOG"

[ -n "${HF_TOKEN:-}" ] || { echo "  ABORT: HF_TOKEN unset."; exit 1; }
[ -n "${HF_HOME:-}" ]  || { echo "  ABORT: HF_HOME unset."; exit 1; }

python -m pytest tests/ -q 2>&1 | tail -3 | tee -a "$LOG"
git diff --quiet -- . ':!*_session.log' || {
    echo "  ABORT: uncommitted changes; run_meta records git_commit."; exit 1; }

# A fresh pod has no git identity. Without this check the failure is SILENT and
# costs the whole run: `git commit || true` swallows the error, `git push` has
# nothing to send, and the end-of-run guard below sees HEAD == origin/main and
# reports "all artefacts pushed" while every database exists only on a machine
# that is about to be terminated.
git config user.email >/dev/null || {
    echo "  ABORT: git identity unset. The driver's commits would fail silently"
    echo "         and the completion check would report a false negative."
    echo "         Run:  git config user.email you@example.com"
    echo "               git config user.name  'Your Name'"
    exit 1; }
git remote get-url origin >/dev/null 2>&1 || {
    echo "  ABORT: no 'origin' remote. Nothing written here would survive."; exit 1; }
echo "  code at $(git rev-parse --short HEAD)" | tee -a "$LOG"

run_group () {
    local name=$1 model=$2 readout=$3 arms=$4 budget=$5; shift 5
    local tag="${TAG}_${name}_sem_${readout}"

    if [ -f "${tag}.done" ]; then
        banner "$tag - already complete, skipping"
        return
    fi

    local free_gb
    free_gb=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc '0-9')
    [ "$free_gb" -ge 18 ] || {
        echo "  ABORT: ${free_gb}G free in HF_HOME, need ~18G for $name." | tee -a "$LOG"
        echo "         rm -rf \$HF_HOME/hub/models--*" | tee -a "$LOG"
        exit 1; }

    banner "$tag  ($model)  arms: $arms  readout: $readout"
    python scripts/gpu_run.py \
        --model "$model" \
        --episodes "$EPISODES" \
        --arms $arms \
        --opponents tft allc \
        --readout "$readout" \
        --max-scratchpad-tokens "$SCRATCH_TOKENS" \
        --run-id "$tag" \
        --db "$tag.sqlite" \
        --out "$tag.json" \
        --budget-minutes "$budget" \
        "$@" 2>&1 \
        | stdbuf -oL tr '\r' '\n' \
        | stdbuf -oL sed -E '/[0-9](it\/s|s\/it)[],]/d' \
        | tee -a "$LOG"

    local rows falsified
    rows=$(python3 -c "
import sqlite3
try:
    print(sqlite3.connect('$tag.sqlite').execute('SELECT COUNT(*) FROM turns').fetchone()[0])
except Exception:
    print(0)")
    [ "$rows" -ge 1 ] || { echo "  ABORT: $tag wrote 0 turns." | tee -a "$LOG"; exit 1; }

    # The manipulation check for this experiment. If displayed_opponent_last is
    # never populated the falsifying arms silently rendered the truth and the
    # whole run is a null by construction - the class of failure that cost this
    # project three smoke runs before anyone looked at a stored prompt.
    falsified=$(python3 -c "
import sqlite3
c = sqlite3.connect('$tag.sqlite')
print(c.execute(\"SELECT COUNT(*) FROM turns WHERE arm IN ('3m','3s','3c') \"
                \"AND displayed_opponent_last IS NOT NULL\").fetchone()[0])")
    echo "  $tag: $rows turns, $falsified rows carry a recorded falsification" | tee -a "$LOG"
    if echo "$arms" | grep -q "3m"; then
        [ "$falsified" -ge 1 ] || {
            echo "  ABORT: no falsification recorded - the lying arms told the truth." \
                | tee -a "$LOG"; exit 1; }
    fi

    gzip -kf "$tag.sqlite"
    touch "${tag}.done"
    git add -f "$tag.sqlite.gz" "$tag.json" "$LOG"
    git commit -m "exp6: $tag" || true
    git push || echo "  PUSH FAILED - do not terminate before this succeeds" | tee -a "$LOG"
}

for entry in "${MODELS[@]}"; do
    IFS=':' read -r name hf_id cache_dir <<< "$entry"
    run_group "$name" "$hf_id" logit "$LOGIT_ARMS" "$LOGIT_BUDGET"
    if [ "$MODE" = "both" ]; then
        run_group "$name" "$hf_id" scratchpad "$PAD_ARMS" "$PAD_BUDGET" \
                  --scratchpad-prompt minimal
    fi
    # One model resident at a time. The volume quota is ~45G against ~46G of
    # weights for three models - this is the EDQUOT that killed two exp4 smokes.
    rm -rf "${HF_HOME:?}/hub/${cache_dir}" 2>/dev/null || true
    rm -f "${TAG}_${name}"_*.sqlite
    echo "  evicted $name  (cache now $(du -sh "$HF_HOME" 2>/dev/null | cut -f1))" | tee -a "$LOG"
done

banner "COMPLETE  ($MODE)"

python3 - <<'EOF' 2>&1 | tee -a "$LOG"
import json, glob, os
print("\nField decomposition. content(3s) isolates the score, content(3m) the")
print("last move. PREDICTION: 3m carries the effect, 3s is null.\n")
print(f"{'group':28}{'opp':6}{'P(D|3)':>9}{'P(D|3s)':>9}{'P(D|3m)':>9}"
      f"{'P(D|3c)':>9}{'3-3s':>9}{'3-3m':>9}{'3-3c':>9}{'off':>7}")
for f in sorted(glob.glob('exp6_*_logit.json')):
    d = json.load(open(f))
    for opp in ('allc', 'tft'):
        try:
            g = {a: d[f'{a}|{opp}'] for a in ('3', '3s', '3m', '3c')}
        except KeyError:
            continue
        off = max(v['off_task_rate'] for v in g.values())
        r = {a: v['defect_rate'] for a, v in g.items()}
        print(f"{f[5:-5]:28}{opp:6}{r['3']:>9.3f}{r['3s']:>9.3f}{r['3m']:>9.3f}"
              f"{r['3c']:>9.3f}{r['3']-r['3s']:>+9.3f}{r['3']-r['3m']:>+9.3f}"
              f"{r['3']-r['3c']:>+9.3f}{off:>7.3f}")
print("""
Read off-task first; above 0.10 the row is prose, not decisions.

Turn 0 rows in arm 3m carry no falsification (no last move to flip) and are
marked donor_degenerate=1. Exclude them before quoting 3-3m, exactly as arm 3c
requires. The rates above do NOT exclude them, so treat this table as a first
look and re-estimate with analysis/02_episode_level.py.

The cell to look at first is 3m vs ALLC. There the block asserts a betrayal
while [HISTORY] shows unbroken cooperation - a contradiction arm 3c cannot
create, because every ALLC donor also shows Cooperate.""")
EOF

git add -f "$LOG"; git commit -m "exp6: session log" || true; git push || true

# Two independent conditions, because "HEAD == origin/main" alone is satisfied
# by the case where NOTHING was ever committed. Checking that every archive on
# disk is also tracked catches the silent-commit-failure path directly.
MISSING=$(comm -23 <(ls "${TAG}"_*.sqlite.gz 2>/dev/null | sort) \
                   <(git ls-files "${TAG}_*.sqlite.gz" | sort) | wc -l | tr -d ' ')
UNPUSHED=$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo unknown)
if [ "$UNPUSHED" != "0" ] || [ "$MISSING" != "0" ]; then
    banner "DATA AT RISK: $UNPUSHED unpushed commit(s), $MISSING untracked archive(s)"
    echo "  DO NOT TERMINATE THE POD. Retry until this succeeds:" | tee -a "$LOG"
    echo "    cd $(pwd)" | tee -a "$LOG"
    echo "    git add -f '${TAG}_*.sqlite.gz' '${TAG}_*.json' '$LOG'" | tee -a "$LOG"
    echo "    git commit -m 'exp6: rescue' && git push" | tee -a "$LOG"
    exit 2
fi
echo "  all artefacts pushed ($(git ls-files "${TAG}_*.sqlite.gz" | wc -l | tr -d ' ') archives tracked)" \
    | tee -a "$LOG"