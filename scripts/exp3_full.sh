#!/usr/bin/env bash
# EXPERIMENT 3 — the complete factorial.
#
#   3 models x 5 arms x 2 opponents x 3 conditions = 90 cells
#
# CONDITIONS
#   sem   semantic labels (Cooperate / Defect), normal mapping   — the main result
#   swap  semantic labels, INVERTED mapping                      — is the effect lexical?
#   abs   abstract labels (X / Y), normal mapping                — FALSIFICATION test
#
# WHY THOSE THREE
#   exp2 found that a [STATE] block moves behaviour while its CONTENT does not,
#   and that the direction flips by model (Llama -21pp, Qwen +5.8pp). The
#   label-swap cells then suggested the effect is on WHICH WORD the model emits
#   rather than on how it plays: the same block raises p(emit "Cooperate")
#   whether or not that word means cooperating.
#
#   If that is right, ABSTRACT framing should shrink or kill the effect, because
#   X and Y carry no cooperative connotation. If the effect is just as large
#   under X/Y, the lexical story is WRONG. That is the cheapest available way to
#   be proven wrong, which is why it is in the design rather than the appendix.
#
#   swap x abs is deliberately omitted: X and Y are already arbitrary, so
#   swapping them tests nothing.
#
# TWO MODES
#   MODE=smoke   4 episodes/cell. Exercises all 90 cells and every code path in
#                ~20 min for ~$0.60. Numbers are meaningless; the point is that
#                nothing raises and every cell writes rows.
#   MODE=prod    1600 episodes/cell. ~4 hours, ~$7.
#
#   MODE=smoke bash scripts/exp3_full.sh
#   MODE=prod  bash scripts/exp3_full.sh
#
# WHY 1600 AND NOT 3000
#   MDE at 1600 is 5pp. The smallest effect that matters is ATE_true at ~1.2pp,
#   and exp2 already detected it at p=0.0003 with episode-level standard errors.
#   Doubling N narrows intervals ~27% and doubles cost; the same money buys the
#   swap and abstract conditions, which answer questions no amount of N can.
#
# EVERYTHING IS LOGGED
#   Full console output goes to exp3_session.log and is committed after every
#   block, so a dead pod cannot take the run's history with it - which is
#   exactly what happened in exp2.

# pipefail is NOT optional here. Every gpu_run.py invocation is piped into tee,
# and without it a pipeline's status is tee's - which always succeeds. A failed
# run would then fall through to gzip, .done and git push, marking a group
# complete and shipping an EMPTY database. Observed exactly that: sqlite3
# creates the file before the failing PRAGMA, so gzip found something to
# compress and the script cheerfully continued to the next model.
set -euo pipefail

# WAL journaling requires shared memory and does not work on a network
# filesystem. RunPod mounts /workspace over MooseFS, where `PRAGMA
# journal_mode=WAL` fails with "disk I/O error", so the repo and its databases
# must live on local disk (/root, the overlay). Model weights stay on the
# volume: they are read-only and never touch WAL.
DB_FS=$(df -T . 2>/dev/null | tail -1 | awk '{print $2}')
case "$DB_FS" in
    fuse*|nfs*|*moose*|*mfs*|9p)
        echo "ABORT: working directory is on '$DB_FS', a network filesystem."
        echo "       SQLite WAL is unsupported there. Clone to /root and run"
        echo "       from that copy; keep HF_HOME on the volume."
        exit 1 ;;
esac

MODE=${MODE:-smoke}
case "$MODE" in
    smoke) EPISODES=4;    BUDGET=10; TAG=smoke ;;
    prod)  EPISODES=1600; BUDGET=40; TAG=exp3  ;;
    *) echo "MODE must be smoke or prod"; exit 1 ;;
esac

ARMS="1 3 3b 3c 3d"
OPPONENTS="tft allc"
LOG=${TAG}_session.log
RULE="=============================================================="

# name : hf id : cache dir prefix
MODELS=(
  "llama:meta-llama/Llama-3.1-8B-Instruct:models--meta-llama--Llama-3.1-8B-Instruct"
  "qwen:Qwen/Qwen2.5-7B-Instruct:models--Qwen--Qwen2.5-7B-Instruct"
  "mistral:mistralai/Mistral-7B-Instruct-v0.3:models--mistralai--Mistral-7B-Instruct-v0.3"
)

banner() { echo | tee -a "$LOG"; echo "$RULE" | tee -a "$LOG"; \
           echo "  $*" | tee -a "$LOG"; echo "$RULE" | tee -a "$LOG"; }

# --- preflight: everything that can fail for free, fails now ---------------

banner "PREFLIGHT  mode=$MODE  episodes=$EPISODES"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"

python -c "import vllm; print('vllm', vllm.__version__)" | tee -a "$LOG"

# One model resident at a time, so ~20G is the requirement, not ~60G.
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "  disk free: ${FREE_GB}G" | tee -a "$LOG"
if [ "$FREE_GB" -lt 8 ]; then
    echo "  ABORT: need ~20G for one model plus room for databases." | tee -a "$LOG"
    exit 1
fi

[ -n "${HF_TOKEN:-}" ] || { echo "  ABORT: HF_TOKEN unset; Llama is gated and"; \
                            echo "         would 401 after the others were paid for."; exit 1; }
[ -n "${HF_HOME:-}" ]  || { echo "  ABORT: HF_HOME unset. Weights would land on the"; \
                            echo "         small container disk, not the volume."; exit 1; }

python -m pytest tests/ -q 2>&1 | tail -3 | tee -a "$LOG"
git diff --quiet || { echo "  ABORT: uncommitted changes; run_meta records git_commit."; exit 1; }
echo "  code at $(git rev-parse --short HEAD)" | tee -a "$LOG"

# --- one cell group ---------------------------------------------------------

run_group () {
    local name=$1 model=$2 cond=$3; shift 3
    local tag="${TAG}_${name}_${cond}"

    if [ -f "${tag}.done" ]; then
        banner "$tag — already complete, skipping"
        return
    fi

    banner "$tag  ($model)  arms: $ARMS"
    python scripts/gpu_run.py \
        --model "$model" \
        --episodes "$EPISODES" \
        --arms $ARMS \
        --opponents $OPPONENTS \
        --run-id "$tag" \
        --db "$tag.sqlite" \
        --out "$tag.json" \
        --budget-minutes "$BUDGET" \
        "$@" 2>&1 | tee -a "$LOG"

    # A group is complete only if it actually wrote turns. sqlite3 creates the
    # file before it can fail, so "the file exists" proves nothing - and a
    # .done marker on an empty database is worse than a crash, because resume
    # would skip it forever.
    local rows
    rows=$(python3 -c "
import sqlite3, sys
try:
    print(sqlite3.connect('$tag.sqlite').execute('SELECT COUNT(*) FROM turns').fetchone()[0])
except Exception:
    print(0)")
    if [ "$rows" -lt 1 ]; then
        echo "  ABORT: $tag wrote 0 turns. Not marking done." | tee -a "$LOG"
        exit 1
    fi
    echo "  $tag wrote $rows turns" | tee -a "$LOG"

    # Commit per group. A preempted pod loses one group, not the session.
    gzip -kf "$tag.sqlite"
    touch "${tag}.done"
    git add -f "$tag.sqlite.gz" "$tag.json" "$LOG"
    git commit -m "$TAG: $tag" || true
    git push || echo "  PUSH FAILED — do not terminate before this succeeds" | tee -a "$LOG"
}

# --- all conditions for one model, then evict its weights -------------------

for entry in "${MODELS[@]}"; do
    IFS=':' read -r name hf_id cache_dir <<< "$entry"

    run_group "$name" "$hf_id" sem
    run_group "$name" "$hf_id" swap --swap-labels
    run_group "$name" "$hf_id" abs  --framing abstract

    # Evict before the next model. One model resident at a time is what keeps
    # this inside a 100G volume; exp2 died of EDQUOT trying to hold three.
    if [ "$MODE" = "prod" ]; then
        rm -rf "${HF_HOME:?}/${cache_dir}" 2>/dev/null || true
        rm -f "${TAG}_${name}"_*.sqlite
        echo "  evicted $name  (cache now $(du -sh "$HF_HOME" | cut -f1))" | tee -a "$LOG"
    fi
done

# --- done -------------------------------------------------------------------

banner "COMPLETE  ($MODE)"
python3 - <<'EOF' 2>&1 | tee -a "$LOG"
import glob, sqlite3, os
tot = 0
for f in sorted(glob.glob('*_*.sqlite.gz')):
    print(f'  {f:44} {os.path.getsize(f)/1e6:7.1f} MB')
print(f'  {len(glob.glob("*.done"))} groups complete')
EOF

git add -f "$LOG"
git commit -m "$TAG: session log" || true
git push

echo
echo "  Confirm every group is pushed, THEN Terminate (not Stop)."
echo "  Record per model from each STEP 2 block, for the methods section:"
echo "      parity target (treatment_block_tokens) and filler token"