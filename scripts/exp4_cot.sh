#!/usr/bin/env bash
# EXPERIMENT 4 — chain-of-thought ablation.
#
# THE OBJECTION THIS ANSWERS
#   Every number in exp1-exp3 comes from LOGIT readout: the action is read from
#   the next-token distribution with no reasoning space. The literature this
#   work corrects lets models reason in text before acting. A reviewer will say
#   the lexical effect is an artefact of denying the model a scratchpad, and
#   that the paper therefore says nothing about modern agent setups.
#
#   They would be right to ask. This answers it.
#
# THE TEST
#   3 models x 2 framings x 2 READOUTS x 3 arms x 2 opponents = 72 cells.
#
#   BOTH READOUTS RUN HERE, ON ONE STACK. exp3's environment cannot be
#   reproduced on a CUDA 12.8 driver - it ran vLLM 0.27.1 / torch 2.13+cu130 /
#   transformers 5.15.0 against driver 580.159.04, and this pod has 0.11.0 /
#   2.8.0+cu128 / 4.57.6 against 570.195.03. Comparing exp4's SCRATCHPAD cells
#   to exp3's LOGIT cells would confound readout with four version changes.
#
#   So the LOGIT condition is re-run here. The CoT-vs-LOGIT contrast is then
#   entirely within one environment, and the exp3-vs-exp4 LOGIT comparison
#   becomes a free measurement of how much the stack itself moves the numbers -
#   at full N rather than a spot check.
#
#   LOGIT cells cost ~1/100th of a scratchpad cell, so this adds ~25 minutes.
#
#   The contrast that matters is the perturbation (arm 1 -> 3b):
#
#     exp3, LOGIT       semantic -0.181 / -0.192    abstract +0.007 / +0.029
#     exp4, LOGIT         ?  (should reproduce the above if the stack is inert)
#     exp4, SCRATCHPAD    ?  (the actual question)
#
#   If the semantic-vs-abstract gap survives reasoning, the lexical account
#   reaches CoT agents and the paper is much stronger. If it closes, the effect
#   is specific to constrained readout - a real boundary condition, and one you
#   would rather publish than have a reviewer discover.
#
#   Arms 3c and 3d are omitted. This is an ablation of a known contrast, not a
#   second factorial; adding them doubles cost for a question already answered
#   under LOGIT.
#
#   MISTRAL ABSTRACT IS INCLUDED DELIBERATELY. Under LOGIT it was 100% off-task
#   and the group was discarded: the model would not emit X or Y at all. Given
#   room to reason first, it may comply. If it does, the condition is recovered
#   AND the earlier failure is localised to constrained readout rather than to
#   abstract labels - a better result than either half alone.
#
# N = 800, NOT 1600
#   This is an ablation of an 18-19pp effect. MDE at 800 is ~7pp, nearly three
#   times inside it. At 128 reasoning tokens per decision, 1600 would double a
#   ~3.5h run to ~7h for intervals nobody will scrutinise. If a cell comes back
#   ambiguous, extend that cell.
#
# THE FAILURE MODE TO WATCH
#   The action is read AFTER the generated reasoning. If the continuation
#   scatters probability across prose the way abstract framing did for Mistral
#   (off-task 1.000, action mass ~0), the cells are unreadable no matter what
#   the defection rate says. STEP 2 prints a warning; the smoke mode exists to
#   catch it in fifteen minutes rather than three hours.
#
# COST
#   Runtime is roughly linear in the reasoning budget. At 128 tokens, 12 cells
#   x N=1600 is ~3h on an A100 and ~1.5-2h on an H100. Compare exp3's 10 cells
#   in 25 minutes: generating 128 tokens per decision instead of 1 is the whole
#   difference.
#
#   MODE=smoke bash scripts/exp4_cot.sh    # 4 episodes, ~15 min
#   MODE=prod  bash scripts/exp4_cot.sh    # 1600 episodes

set -euo pipefail

DB_FS=$(df -T . 2>/dev/null | tail -1 | awk '{print $2}')
case "$DB_FS" in
    fuse*|nfs*|*moose*|*mfs*|9p)
        echo "ABORT: working directory is on '$DB_FS'. SQLite WAL is"
        echo "       unsupported there. Run from local disk (/root)."
        exit 1 ;;
esac

MODE=${MODE:-smoke}
case "$MODE" in
    smoke) DEFAULT_EPISODES=4;   DEFAULT_BUDGET=15;  TAG=cotsmoke ;;
    prod)  DEFAULT_EPISODES=800; DEFAULT_BUDGET=150; TAG=exp4     ;;
    *) echo "MODE must be smoke or prod"; exit 1 ;;
esac

# Overridable from the command line:
#   EPISODES=1000 MODE=prod bash scripts/exp4_cot.sh
#
# BUDGET scales with EPISODES. Generating 128 tokens per decision instead of 1
# is ~100x the work of an exp3 cell, so a budget sized for LOGIT will truncate
# groups here. 150 min covers 6 cells at N=1000 on an A100 with headroom; the
# guard exists to stop a runaway, not to pace a healthy run.
EPISODES=${EPISODES:-$DEFAULT_EPISODES}
BUDGET=${BUDGET:-$DEFAULT_BUDGET}

# LOGIT groups emit one token per decision and finish in minutes; the shared
# budget would be absurd for them and is only a runaway guard anyway.
LOGIT_BUDGET=${LOGIT_BUDGET:-25}

# name : hf id : cache dir prefix
MODELS=(
  "llama:meta-llama/Llama-3.1-8B-Instruct:models--meta-llama--Llama-3.1-8B-Instruct"
  "qwen:Qwen/Qwen2.5-7B-Instruct:models--Qwen--Qwen2.5-7B-Instruct"
  "mistral:mistralai/Mistral-7B-Instruct-v0.3:models--mistralai--Mistral-7B-Instruct-v0.3"
)

ARMS="1 3b 3"
OPPONENTS="tft allc"
SCRATCH_TOKENS=${SCRATCH_TOKENS:-128}
LOG=${TAG}_session.log
RULE="=============================================================="

banner() { echo | tee -a "$LOG"; echo "$RULE" | tee -a "$LOG"; \
           echo "  $*" | tee -a "$LOG"; echo "$RULE" | tee -a "$LOG"; }

banner "PREFLIGHT  mode=$MODE  episodes=$EPISODES  scratchpad=$SCRATCH_TOKENS"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
python -c "import vllm; print('vllm', vllm.__version__)" | tee -a "$LOG"

FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
NEED_GB=8
if [ "$(df --output=source "$HF_HOME" | tail -1)" = "$(df --output=source . | tail -1)" ]; then
    NEED_GB=30
fi
echo "  disk free: ${FREE_GB}G  (need ${NEED_GB}G)" | tee -a "$LOG"
[ "$FREE_GB" -ge "$NEED_GB" ] || { echo "  ABORT: insufficient space."; exit 1; }

[ -n "${HF_TOKEN:-}" ] || { echo "  ABORT: HF_TOKEN unset."; exit 1; }
[ -n "${HF_HOME:-}" ]  || { echo "  ABORT: HF_HOME unset."; exit 1; }

python -m pytest tests/ -q 2>&1 | tail -3 | tee -a "$LOG"
git diff --quiet -- . ':!*_session.log' || {
    echo "  ABORT: uncommitted changes; run_meta records git_commit."; exit 1; }
echo "  code at $(git rev-parse --short HEAD)" | tee -a "$LOG"

run_group () {
    local name=$1 model=$2 cond=$3 readout=$4; shift 4
    local tag="${TAG}_${name}_${cond}_${readout}"

    if [ -f "${tag}.done" ]; then
        banner "$tag — already complete, skipping"
        return
    fi

    # LOGIT emits one token per decision; SCRATCHPAD emits up to SCRATCH_TOKENS.
    # A budget sized for one starves the other. Written as an if rather than
    # `[ x ] && y=z`, whose exit status is 1 when the test fails - harmless
    # mid-function under set -e, but not worth relying on.
    local budget=$BUDGET
    if [ "$readout" = "logit" ]; then budget=$LOGIT_BUDGET; fi

    banner "$tag  ($model)  arms: $ARMS  readout: $readout"
    python scripts/gpu_run.py \
        --model "$model" \
        --episodes "$EPISODES" \
        --arms $ARMS \
        --opponents $OPPONENTS \
        --readout "$readout" \
        --max-scratchpad-tokens "$SCRATCH_TOKENS" \
        --run-id "$tag" \
        --db "$tag.sqlite" \
        --out "$tag.json" \
        --budget-minutes "$budget" \
        "$@" 2>&1 | stdbuf -oL tr '\r' '\n' \
                  | stdbuf -oL sed -E '/[0-9](it\/s|s\/it)[],]/d' \
                  | tee -a "$LOG"

    local rows
    rows=$(python3 -c "
import sqlite3
try:
    print(sqlite3.connect('$tag.sqlite').execute('SELECT COUNT(*) FROM turns').fetchone()[0])
except Exception:
    print(0)")
    [ "$rows" -ge 1 ] || { echo "  ABORT: $tag wrote 0 turns." | tee -a "$LOG"; exit 1; }
    echo "  $tag wrote $rows turns" | tee -a "$LOG"

    # Scratchpad groups must contain reasoning, or the readout degenerated to
    # reading a continuation of nothing. LOGIT groups legitimately have none.
    if [ "$readout" = "scratchpad" ]; then
        local pads
        pads=$(python3 -c "
import sqlite3
c = sqlite3.connect('$tag.sqlite')
print(c.execute(\"SELECT COUNT(*) FROM turn_details WHERE scratchpad IS NOT NULL AND LENGTH(scratchpad) > 20\").fetchone()[0])")
        echo "  $tag stored $pads non-trivial scratchpads" | tee -a "$LOG"
        [ "$pads" -ge 1 ] || { echo "  ABORT: no reasoning generated." | tee -a "$LOG"; exit 1; }
    fi

    gzip -kf "$tag.sqlite"
    touch "${tag}.done"
    git add -f "$tag.sqlite.gz" "$tag.json" "$LOG"
    git commit -m "$TAG: $tag" || true
    git push || echo "  PUSH FAILED — do not terminate before this succeeds" | tee -a "$LOG"
}

for entry in "${MODELS[@]}"; do
    IFS=':' read -r name hf_id cache_dir <<< "$entry"

    # LOGIT first: it is ~100x cheaper and establishes this stack's baseline
    # before the expensive cells run. If a LOGIT group disagrees with exp3, the
    # stack matters and you learn it in four minutes rather than after three
    # hours of generation.
    run_group "$name" "$hf_id" sem logit
    run_group "$name" "$hf_id" abs logit --framing abstract

    run_group "$name" "$hf_id" sem scratchpad
    run_group "$name" "$hf_id" abs scratchpad --framing abstract

    # One model resident at a time, in BOTH modes. The volume quota is ~45G
    # against ~46G of weights for three models, so smoke runs hit EDQUOT on the
    # third model exactly like prod does - and did, twice.
    rm -rf "${HF_HOME:?}/hub/${cache_dir}" 2>/dev/null || true
    [ "$MODE" = "prod" ] && rm -f "${TAG}_${name}"_*.sqlite
    echo "  evicted $name  (cache now $(du -sh "$HF_HOME" 2>/dev/null | cut -f1))" | tee -a "$LOG"
done

banner "COMPLETE  ($MODE)"

python3 - <<EOF 2>&1 | tee -a "$LOG"
import json, glob
for f in sorted(glob.glob('${TAG}_*.json')):
    d = json.load(open(f))
    print(f'\n{f}')
    for k, v in sorted(d.items()):
        if isinstance(v, dict) and 'defect_rate' in v:
            print(f"  {k:10}{v['defect_rate']:>8.3f}  off-task {v['off_task_rate']:>6.3f}")
EOF

git add -f "$LOG"; git commit -m "$TAG: session log" || true; git push || true

# THE ONLY COPY THAT SURVIVES THE POD IS THE ONE ON THE REMOTE.
#
# A failed push is not fatal mid-run: the commit is already in local history,
# so the next group's push carries it. The .gz files are also never deleted -
# `gzip -kf` keeps the original and the eviction glob only matches *.sqlite.
#
# What IS fatal is a failed push on the LAST group. The script would print
# COMPLETE, the run would look green, and terminating the pod would destroy
# three hours of GPU time that exists nowhere else. Warnings scroll past in a
# log this size; an explicit exit code does not.
#
# Copying the archives elsewhere ON THIS POD is not a mitigation - same disk,
# same lifetime. Only the remote counts.
UNPUSHED=$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo unknown)
if [ "$UNPUSHED" != "0" ]; then
    banner "DATA AT RISK: $UNPUSHED commit(s) exist only on this pod"
    echo "  Retry until this succeeds, THEN terminate:" | tee -a "$LOG"
    echo "      cd $(pwd) && git push" | tee -a "$LOG"
    ls -la ./"${TAG}"_*.gz 2>/dev/null | tee -a "$LOG"
    exit 2
fi
echo "  all artefacts pushed to $(git rev-parse --abbrev-ref '@{u}')" | tee -a "$LOG"

echo
echo "  Read off-task FIRST. Any cell above ~0.10 is measuring a tail, not a"
echo "  decision, and its defection rate means nothing."
echo "  Then compare perturbation (arm 1 -> 3b) against exp3's LOGIT values:"
echo "      semantic  -0.181 / -0.192      abstract  +0.007 / +0.029"