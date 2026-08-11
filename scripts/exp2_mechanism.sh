#!/usr/bin/env bash
# EXPERIMENT 2 — mechanism decomposition and cross-model replication.
#
# EXPLORATORY. Experiment 1 (tag exp1-frozen) tested the pre-registered
# sign-flip hypothesis and rejected it at p<1e-4. Nothing here revisits that,
# and no p-value from this run may be presented as if it were pre-registered.
#
# Run IDs are exp2_* to match the exp1-frozen tag. They are written into
# run_meta and onto every row, so they end up in the released dataset
# permanently - "s1_llama" would tell a reader nothing in two years.
#
# WHAT IT ANSWERS
#
#   1. WHY does an information-free block cut defection by 26pp?
#      Arm 3d is structure with no language, at matched density. If 3d ~ 3b the
#      effect is structural; if 3d < 3b, language content carries part of it.
#
#   2. Does WRONG state differ from IRRELEVANT state?
#      Arm 3c renders the treatment template from another episode's game.
#      BEHAVIOURAL contrast only - the scaffold-echo instrument check is NOT
#      wired (see cdx/donor.py). Nothing writes the scaffold_echo column, so
#      this run cannot show whether the model read the block.
#
#   3. Is any of this Llama-specific?
#      Three models, identical protocol. This is what turns "Llama-3.1-8B does
#      X" into "small open-weight models do X", and it is the single most
#      predictable reviewer objection to Experiment 1.
#
#   4. Is behaviour a strategy or a position bias?
#      The label-swap cells invert which word means which action. If choices
#      follow the label rather than the meaning, the behavioural claims are
#      about token position and not about play.
#
# WATCH THE DONOR DEGENERACY GATE. Episodes advance in lockstep, so at turn 0
# every episode has score 0 and no last move and no distinct donor can exist.
# A cell reporting high degeneracy did not test stale state on those turns and
# its estimate is diluted accordingly. The gate prints this rather than letting
# it pass unnoticed.
#
#   bash scripts/exp2_mechanism.sh
#
# Roughly 95 minutes and ~$2.60 on one A100-80GB.

set -euo pipefail

EPISODES=${EPISODES:-1600}
FULL_ARMS="1 3 3b 3c 3d"
SWAP_ARMS="1 3b"
OPPONENTS="tft allc"
RULE="=============================================================="

LLAMA="meta-llama/Llama-3.1-8B-Instruct"
QWEN="Qwen/Qwen2.5-7B-Instruct"
MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"

banner() { echo; echo "$RULE"; echo "  $*"; echo "$RULE"; }

# --- preflight: everything that can fail for free, fails now ---------------

banner "PREFLIGHT"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# GNU coreutils syntax. This script runs on the Linux pod, never on macOS.
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "  disk free: ${FREE_GB}G"
if [ "$FREE_GB" -lt 60 ]; then
    echo "  ABORT: three 7-8B models need ~50G of weights plus overhead."
    echo "         A volume cannot be resized after launch; redeploy with 100G."
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "  ABORT: HF_TOKEN unset. Llama-3.1 is gated and would 401 forty"
    echo "         minutes in, after Qwen and Mistral had already been paid for."
    echo "         export HF_TOKEN=...   (never commit it)"
    exit 1
fi

python -m pytest tests/ -q
git diff --quiet || { echo "  ABORT: uncommitted changes. run_meta records"; \
                      echo "         git_commit, which would then be a lie."; exit 1; }
echo "  code at $(git rev-parse --short HEAD)"

banner "STEP 0  instrument verification (~3 min)"

# ALL FIVE ARMS, not the default three. The donor wiring in run_cell and the
# degeneracy gate in report() are covered by unit tests only in isolation - the
# integration has never executed. Exercising Arm 3c here costs about ten cents
# and three minutes; discovering a typo in it costs twenty-five minutes and a
# dead cell, eight cells into the paid run.
#
# gpu_run.py --verify hardcodes its output to verify.json, which is a COMMITTED
# EXPERIMENT 1 ARTEFACT. Left alone it would be silently overwritten and then
# swept up by the first `git add`. Move the new one aside; restore exp1's.
python scripts/gpu_run.py --model "$LLAMA" --verify \
    --arms $FULL_ARMS --opponents tft
[ -f verify.json ] || { echo "  ABORT: verify produced no output"; exit 1; }
mv verify.json exp2_verify.json
git checkout -- verify.json
echo "  exp1 verify.json restored; this run's is exp2_verify.json"

# --- one block per model ---------------------------------------------------

run_block () {
    local tag=$1 model=$2 budget=$3 arms=$4; shift 4
    banner "$tag  ($model)  arms: $arms"
    python scripts/gpu_run.py \
        --model "$model" \
        --episodes "$EPISODES" \
        --arms $arms \
        --opponents $OPPONENTS \
        --run-id "$tag" \
        --db "$tag.sqlite" \
        --out "$tag.json" \
        --budget-minutes "$budget" \
        "$@"

    # Commit immediately. A preempted pod then loses the current block only.
    # Experiment 1 was committed once at the very end, which was luck.
    gzip -kf "$tag.sqlite"
    git add -f "$tag.sqlite.gz" "$tag.json"
    git commit -m "exp2: $tag" || true
    git push || echo "  PUSH FAILED — do not terminate the pod until it succeeds"
}

# Ordered by value: a budget overrun costs the least important block.
run_block exp2_llama   "$LLAMA"   35 "$FULL_ARMS"
run_block exp2_qwen    "$QWEN"    35 "$FULL_ARMS"
run_block exp2_mistral "$MISTRAL" 35 "$FULL_ARMS"

# Baseline and placebo only - the contrast is what matters, not the full set.
run_block exp2_llama_labelswap "$LLAMA" 15 "$SWAP_ARMS" --swap-labels

# --- done ------------------------------------------------------------------

banner "COMPLETE"
git log --oneline -6
echo
echo "  Confirm every block is pushed, THEN Terminate the pod (not Stop)."
echo "  Stop keeps billing for storage."
echo
echo "  Record per model, from each block's STEP 2 output — both differ by"
echo "  tokeniser and belong in the methods section:"
echo "      parity target  (treatment_block_tokens)"
echo "      filler token   (Mistral has no single-token newline)"
echo
echo "  Analyse locally:"
echo "    for f in exp2_*.sqlite; do"
echo "      python analysis/02_episode_level.py --db \$f --out ep_\${f%.sqlite}.json"
echo "    done"
echo "    python analysis/01_diagnose_arm3.py --db exp2_llama.sqlite"