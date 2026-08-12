#!/usr/bin/env bash
# EXPERIMENT 5 - minimal-CoT control for exp4's horizon confound.
#
# THE CONFOUND
#   exp4's SCRATCHPAD instruction reads:
#
#     "Before choosing, reason step by step about the current state, the
#      opponent's behaviour so far, and how many rounds remain."
#
#   That last clause names the finite horizon. Finite horizons induce backward
#   induction, and backward induction is the textbook argument for defecting
#   from round one. exp4's most consistent result is a large level shift in
#   defection between readouts, in all three models, from three different
#   floors:
#
#     semantic, P(D | arm 3b)     LOGIT  ->  SCRATCHPAD
#       llama                     0.102  ->  0.579
#       qwen                      0.042  ->  0.597
#       mistral                   0.000  ->  0.335
#
#   And in llama, the lexical container effect collapses:
#
#     perturbation (1 -> 3b)      LOGIT -0.222 / -0.207
#                                 SCRATCHPAD +0.022 / +0.016
#
#   Each has two explanations and exp4 cannot separate them:
#
#     "reasoning shatters the cooperative prior"  vs  "naming the horizon does"
#     "reasoning kills the placebo effect"        vs  "'reason about the
#                                                      current state' does, by
#                                                      replacing passive lexical
#                                                      priming with active
#                                                      attention"
#
#   exp4's WITHIN-readout contrasts are untouched - the confound is applied
#   identically to every arm - so every ATE_true and every perturbation figure
#   in exp4 stands. Only the cross-readout claim is unsafe.
#
# THE CONTROL
#   Identical to exp4's semantic scratchpad groups in every respect except the
#   instruction, which becomes:
#
#     "Before choosing, think step by step."
#
#   No state, no opponent, no horizon, no action, no output format. Reasoning is
#   requested; nothing is pointed at.
#
# ALL THREE MODELS - and this is not scope creep
#   The perturbation collapse is a LLAMA question: qwen's semantic LOGIT
#   perturbation was +0.033 and mistral's was 0.000, so neither has a container
#   effect to collapse.
#
#   But the LEVEL SHIFT appears in all three, and it is the paper's most
#   consistent cross-architecture finding. Controlling it in one model licenses
#   a claim about one model. Re-running the other two later means a different
#   pod - and exp3 -> exp4 already measured a ~4pp shift from the stack alone,
#   which would land on top of the effect being measured. Same session or the
#   comparison is worth nothing.
#
# SEMANTIC ONLY
#   llama's abstract perturbation was +0.011 / +0.043 - nothing to collapse -
#   and qwen_abs_scratchpad is excluded at off-task 0.201. Adding the abstract
#   condition doubles the cost to answer a question nobody asked.
#
# READING IT - TWO INDEPENDENT ANSWERS PER MODEL, NOT ONE
#   The level shift and the perturbation collapse can dissociate. Read both:
#
#     level high, perturb ~0      both effects are reasoning. exp4 stands.
#     level low,  perturb ~0      level was horizon salience; collapse is real.
#     level high, perturb ~-0.20  collapse was attention; level is reasoning.
#     level low,  perturb ~-0.20  both were instruction artefacts.
#
#   Every one of those four is publishable. Only running it is optional, and
#   only until a reviewer opens the prompt appendix.
#
# WHAT THIS CANNOT TELL YOU
#   The rules section already states the game lasts 20 rounds, so the horizon
#   was never hidden - the exp4 instruction only made it SALIENT. exp5 is a
#   salience control, not a horizon control. Answering "does KNOWING the horizon
#   cause defection?" needs HorizonMode.STOCHASTIC, where no last round exists
#   and backward induction cannot apply. That machinery is in the codebase; it
#   is not this run. Describe exp5 as a salience control in the writeup.
#
# COST
#   3 groups x 6 cells x N=1000 at 128 reasoning tokens ~ 25 min each, plus two
#   model downloads after eviction. ~1.5 h, ~$6.
#
#   EPISODES=1000 bash scripts/exp5_minimal_cot.sh

set -euo pipefail

DB_FS=$(df -T . 2>/dev/null | tail -1 | awk '{print $2}')
case "$DB_FS" in
    fuse*|nfs*|*moose*|*mfs*|9p)
        echo "ABORT: working directory is on '$DB_FS'. SQLite WAL is"
        echo "       unsupported there. Run from local disk (/root)."
        exit 1 ;;
esac

EPISODES=${EPISODES:-1000}
BUDGET=${BUDGET:-60}
SCRATCH_TOKENS=${SCRATCH_TOKENS:-128}
TAG=exp5
LOG=exp5_session.log
RULE="=============================================================="

# name : hf id : cache dir prefix
MODELS=(
  "llama:meta-llama/Llama-3.1-8B-Instruct:models--meta-llama--Llama-3.1-8B-Instruct"
  "qwen:Qwen/Qwen2.5-7B-Instruct:models--Qwen--Qwen2.5-7B-Instruct"
  "mistral:mistralai/Mistral-7B-Instruct-v0.3:models--mistralai--Mistral-7B-Instruct-v0.3"
)

banner() { echo | tee -a "$LOG"; echo "$RULE" | tee -a "$LOG"; \
           echo "  $*" | tee -a "$LOG"; echo "$RULE" | tee -a "$LOG"; }

banner "PREFLIGHT  episodes=$EPISODES  scratchpad=$SCRATCH_TOKENS  prompt=minimal"

# exp4 must already exist ON THIS POD: this control is meaningless against
# numbers from a different stack, and finding that out after 90 minutes is
# worse than finding it out now.
for m in llama qwen mistral; do
    for suffix in sem_scratchpad sem_logit; do
        [ -f "exp4_${m}_${suffix}.json" ] || {
            echo "  ABORT: exp4_${m}_${suffix}.json missing."
            echo "         Run exp4 on this pod first - a cross-pod comparison"
            echo "         carries the ~4pp stack offset measured in exp3->exp4."
            exit 1; }
    done
done
echo "  exp4 reference data present for all three models" | tee -a "$LOG"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
python -c "import vllm; print('vllm', vllm.__version__)" | tee -a "$LOG"

[ -n "${HF_TOKEN:-}" ] || { echo "  ABORT: HF_TOKEN unset."; exit 1; }
[ -n "${HF_HOME:-}" ]  || { echo "  ABORT: HF_HOME unset."; exit 1; }

python -m pytest tests/ -q 2>&1 | tail -3 | tee -a "$LOG"
git diff --quiet -- . ':!*_session.log' || {
    echo "  ABORT: uncommitted changes; run_meta records git_commit."; exit 1; }
echo "  code at $(git rev-parse --short HEAD)" | tee -a "$LOG"

run_group () {
    local name=$1 model=$2
    local tag="${TAG}_${name}_sem_minimal"

    if [ -f "${tag}.done" ]; then
        banner "$tag - already complete, skipping"
        return
    fi

    # exp4 evicts weights after each model, so every model here is a fresh
    # download. Surface a quota failure before the engine starts rather than
    # inside vLLM's loader, where it appears as an opaque EngineCore crash.
    local free_gb
    free_gb=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc '0-9')
    [ "$free_gb" -ge 18 ] || {
        echo "  ABORT: ${free_gb}G free in HF_HOME, need ~18G for $name." | tee -a "$LOG"
        echo "         rm -rf \$HF_HOME/hub/models--*" | tee -a "$LOG"
        exit 1; }

    banner "$tag  ($model)  arms: 1 3b 3  opponents: tft allc  prompt: MINIMAL"
    python scripts/gpu_run.py \
        --model "$model" \
        --episodes "$EPISODES" \
        --arms 1 3b 3 \
        --opponents tft allc \
        --readout scratchpad \
        --scratchpad-prompt minimal \
        --max-scratchpad-tokens "$SCRATCH_TOKENS" \
        --run-id "$tag" \
        --db "$tag.sqlite" \
        --out "$tag.json" \
        --budget-minutes "$BUDGET" 2>&1 \
        | stdbuf -oL tr '\r' '\n' \
        | stdbuf -oL sed -E '/[0-9](it\/s|s\/it)[],]/d' \
        | tee -a "$LOG"

    local rows pads
    rows=$(python3 -c "
import sqlite3
try:
    print(sqlite3.connect('$tag.sqlite').execute('SELECT COUNT(*) FROM turns').fetchone()[0])
except Exception:
    print(0)")
    [ "$rows" -ge 1 ] || { echo "  ABORT: $tag wrote 0 turns." | tee -a "$LOG"; exit 1; }

    pads=$(python3 -c "
import sqlite3
c = sqlite3.connect('$tag.sqlite')
print(c.execute(\"SELECT COUNT(*) FROM turn_details WHERE scratchpad IS NOT NULL AND LENGTH(scratchpad) > 20\").fetchone()[0])")
    echo "  $tag: $rows turns, $pads non-trivial scratchpads" | tee -a "$LOG"
    [ "$pads" -ge 1 ] || { echo "  ABORT: no reasoning generated." | tee -a "$LOG"; exit 1; }

    gzip -kf "$tag.sqlite"
    touch "${tag}.done"
    git add -f "$tag.sqlite.gz" "$tag.json" "$LOG"
    git commit -m "exp5: $tag" || true
    git push || echo "  PUSH FAILED - do not terminate before this succeeds" | tee -a "$LOG"
}

for entry in "${MODELS[@]}"; do
    IFS=':' read -r name hf_id cache_dir <<< "$entry"
    run_group "$name" "$hf_id"
    # One model resident at a time. The volume quota is ~45G against ~46G of
    # weights for three models; this is the EDQUOT that killed two exp4 smokes.
    rm -rf "${HF_HOME:?}/hub/${cache_dir}" 2>/dev/null || true
    rm -f "${TAG}_${name}"_*.sqlite
    echo "  evicted $name  (cache now $(du -sh "$HF_HOME" 2>/dev/null | cut -f1))" | tee -a "$LOG"
done

banner "COMPARISON  guided vs minimal, semantic scratchpad, all models"

python3 - <<'EOF' 2>&1 | tee -a "$LOG"
import json, os

MODELS = ("llama", "qwen", "mistral")
ROWS = (("LOGIT        ", "exp4_%s_sem_logit"),
        ("CoT guided   ", "exp4_%s_sem_scratchpad"),
        ("CoT minimal  ", "exp5_%s_sem_minimal"))

for m in MODELS:
    print(f"\n=== {m} ===")
    print(f"{'condition':14} {'opp':5} {'P(D|1)':>8} {'P(D|3b)':>8} {'P(D|3)':>8}"
          f" {'perturb':>9} {'ATE_true':>9} {'off':>6}")
    for label, pattern in ROWS:
        path = pattern % m + ".json"
        if not os.path.exists(path):
            print(f"{label:14} (missing {path})")
            continue
        d = json.load(open(path))
        for opp in ("allc", "tft"):
            try:
                b, p, t = d[f"1|{opp}"], d[f"3b|{opp}"], d[f"3|{opp}"]
            except KeyError:
                print(f"{label:14} {opp:5} (incomplete)")
                continue
            print(f"{label:14} {opp:5} {b['defect_rate']:8.3f} {p['defect_rate']:8.3f}"
                  f" {t['defect_rate']:8.3f} {p['defect_rate']-b['defect_rate']:9.3f}"
                  f" {t['defect_rate']-p['defect_rate']:9.3f}"
                  f" {max(b['off_task_rate'],p['off_task_rate'],t['off_task_rate']):6.3f}")

print("""
READ TWO THINGS PER MODEL, NOT ONE.

  LEVEL     P(D) under minimal vs guided. If minimal stays high, reasoning
            drives defection. If it falls back toward the LOGIT level, exp4's
            level shift was the horizon clause, not the reasoning.

  PERTURB   1 -> 3b under minimal. Only LLAMA can answer this - qwen and
            mistral had no container effect under LOGIT to collapse. If llama's
            stays near zero, the container effect really does die under
            reasoning. If it returns to about -0.20, the collapse was caused by
            telling the model to attend to the state.

Any combination is a result. None is a reason to bury this.
Off-task above ~0.10 in any row voids that row regardless of what it says.
And exp5 controls SALIENCE, not availability: the rules already state the
horizon. Say so in the writeup.""")
EOF

UNPUSHED=$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo unknown)
if [ "$UNPUSHED" != "0" ]; then
    banner "DATA AT RISK: $UNPUSHED commit(s) exist only on this pod"
    echo "  Retry until this succeeds, THEN terminate:  cd $(pwd) && git push"
    exit 2
fi
git add -f "$LOG"; git commit -m "exp5: session log" || true; git push || true
echo "  all artefacts pushed" | tee -a "$LOG"