#!/usr/bin/env bash
# EXPERIMENT 7 - the two confounds that decide whether exp6 means anything.
#
# THE FINDING UNDER TEST
#   exp6: falsifying ONLY "Opponent's last move" moves defection up to -0.4049
#   (qwen vs TFT), 2.8x-31x more than falsifying only the score, over the 4
#   of 6 cells where the score contrast excludes zero (59x and 267x divide by
#   a null),
#   while 30,000/30,000 probes reproduced the falsified score perfectly and
#   behaviour moved 0.5-3.2pp. Read as: the model conditions on the last-move
#   FIELD, not on the state block in general.
#
#   Three independent reviewers converged on two objections. Neither is
#   answerable from any database in this repository, because neither factor was
#   ever varied. Both are cheap.
#
# ---------------------------------------------------------------------------
# OBJECTION 1 - LEXICAL PRIMING              [phase: abs]
# ---------------------------------------------------------------------------
#   exp6 ran under SEMANTIC framing only. Arm 3m injects the literal token
#   "Defect" into the context. This project's own exp3 shows labels dominating
#   every other manipulation: Llama's baseline defection moves 0.28-0.31 under
#   Cooperate/Defect to 0.71-0.74 under X/Y, and Qwen's container effect
#   REVERSES SIGN between framings (CLAIMS.md D2). So "the model conditions on
#   the last-move field" is currently indistinguishable from "the string
#   'Defect' raises P(Defect)".
#
#   THE TEST. Arms {1, 3b, 3, 3s, 3m} under Framing.ABSTRACT. X/Y preserves the
#   field's INFORMATIONAL role exactly - "Opponent's last move: Y" is as
#   decision-relevant against TFT as "Opponent's last move: Defect" - and strips
#   its LEXICAL force, because Y carries no connotation to prime with.
#
#     effect survives under X/Y   -> state conditioning. C5 stands, and the
#                                    claim gets the falsification test D2 got.
#     effect vanishes under X/Y   -> lexical priming. C5 must be rewritten as a
#                                    result about tokens, converging with D2.
#
#   3c is dropped from this phase: exp6 showed it is a weak instrument (0.0000
#   falsification rate vs ALLC, 0.14-0.38 vs TFT, CLAIMS.md C8) and the
#   deliberate arms measure the same fields at a known dose.
#
#   VIABILITY. mistral_abs is EXCLUDED BEFORE THE RUN, not after: off-task 1.000
#   in exp3 and 1.000 again in exp4_mistral_abs_logit - two independent stacks
#   agreeing that Mistral does not emit X or Y at all when a state block is
#   present. Paying for it a third time buys a third exclusion. llama_abs and
#   qwen_abs are clean under LOGIT in both runs. qwen_abs_scratchpad is excluded
#   for the same reason at 0.201, which is why this phase is LOGIT-only - and
#   LOGIT is where the effect under test lives anyway.
#
# ---------------------------------------------------------------------------
# OBJECTION 2 - THE BLOCK IS REDUNDANT WITH [HISTORY]      [phase: nohist]
# ---------------------------------------------------------------------------
#   PromptAssembler.assemble has ALWAYS rendered [HISTORY] with every round in
#   it, one section below the block. So arms 3c/3s/3m were never false-state
#   manipulations - they were CONTRADICTION manipulations. The truth sat in the
#   same context window: trivially checkable for the last move (read the final
#   line) and arithmetically expensive for the score (sum twenty payoffs).
#
#   That admits a rival account of the ENTIRE exp6 pattern: "models discount a
#   locally contradicted claim, and discount it more when the contradiction is
#   cheap to verify." It predicts the score/move asymmetry, the perfect probe
#   reproduction of the score lie (reproducing is not believing), and the
#   ALLC/TFT difference. Nothing in six experiments separates it from the
#   last-move account.
#
#   THE TEST. Identical arms, --no-history, so the [STATE] block is the only
#   source of state and the lie is unrefuted.
#
#     behaviour now tracks the block -> the finding is "conflict resolution
#                                       favours raw history over injected
#                                       summaries" - sharper, more useful, and
#                                       a claim about context engineering rather
#                                       than about game theory.
#     behaviour still ignores it     -> the dissociation is finally earned. The
#                                       block is not discounted for being
#                                       contradicted; it is simply not used.
#
#   BONUS, and it is not small: with history removed arm 1 becomes a genuine
#   state-DEPRIVATION condition rather than "same state, no summary", and arm-3
#   CPR stops being a copy task - the model can no longer answer the probe by
#   reading the log.
#
# WHAT WOULD FALSIFY THE HEADLINE
#   3-3m at ~zero under X/Y in BOTH llama and qwen. C5 would then be a lexical
#   result and must be retracted in place, as F retracts the others.
#
# ---------------------------------------------------------------------------
#   MODE=nohist bash scripts/exp7_confounds.sh    # objection 2   ~35 min  ~$1.8
#   MODE=abs    bash scripts/exp7_confounds.sh    # objection 1   ~25 min  ~$1.3
#   MODE=swap   bash scripts/exp7_confounds.sh    # ceiling-proof lexical test
#   MODE=both   bash scripts/exp7_confounds.sh    # abs + nohist    ~55 min  ~$3
#   MODE=cross  bash scripts/exp7_confounds.sh    # + swap + abs x nohist   ~$5
#   MODE=pad    bash scripts/exp7_confounds.sh    # nohist under minimal CoT
#
# RECOMMENDED SCOPE: MODE=cross in one session, ~1.5 h, ~$5 on an H100 at $3/h.
# Hold MODE=pad (~$3.6) in reserve for the case where the LOGIT result is
# ambiguous or a reviewer insists on the CoT regime the literature uses.
#
# RUN ORDER IF ONLY ONE IS AFFORDABLE: nohist. Reasoning in the closing banner.

set -euo pipefail

DB_FS=$(df -T . 2>/dev/null | tail -1 | awk '{print $2}')
case "$DB_FS" in
    fuse*|nfs*|*moose*|*mfs*|9p)
        echo "ABORT: working directory is on '$DB_FS'. SQLite WAL is"
        echo "       unsupported there. Run from local disk (/root)."
        exit 1 ;;
esac

# abs | nohist | both | cross | pad. Separate phases because they have different
# model sets (mistral cannot be measured under X/Y), different N requirements,
# and different failure modes. A single mode would mean a failure in one phase
# costs the other, which is exactly what the pad/logit split bought in exp6.
MODE=${MODE:-both}
case "$MODE" in
    abs|swap|nohist|both|cross|pad) ;;
    *) echo "ABORT: MODE must be abs, swap, nohist, both, cross or pad (got '$MODE')"
       exit 1 ;;
esac
# Set per phase and read by the history gate inside run_group. Declared here so
# `set -u` cannot turn a missing assignment into an unchecked run.
NOHIST=0
EPISODES=${EPISODES:-1000}
LOGIT_BUDGET=${LOGIT_BUDGET:-30}
PAD_BUDGET=${PAD_BUDGET:-90}
SCRATCH_TOKENS=${SCRATCH_TOKENS:-128}
TAG=exp7
LOG=exp7_session.log
RULE="=============================================================="

# Same five arms in every phase, so the abstract and no-history results are
# directly comparable to each other AND to exp6's semantic-with-history cells.
# 3c is omitted: measured at 0.0000 dose vs ALLC in exp6, and the deliberate
# arms measure its two fields at a dose that is known rather than sampled.
ARMS="1 3b 3 3s 3m"
PAD_ARMS="1 3b 3 3m"

# One model list, one residency. Weights are ~18G each against a ~45G quota and
# a several-minute download, so a model is loaded once and every condition it is
# eligible for runs before it is evicted. Phase-major ordering would download
# llama and qwen twice in MODE=both for no scientific gain.
MODELS=(
  "llama:meta-llama/Llama-3.1-8B-Instruct:models--meta-llama--Llama-3.1-8B-Instruct"
  "qwen:Qwen/Qwen2.5-7B-Instruct:models--Qwen--Qwen2.5-7B-Instruct"
  "mistral:mistralai/Mistral-7B-Instruct-v0.3:models--mistralai--Mistral-7B-Instruct-v0.3"
)

# WHICH MODELS CAN BE MEASURED UNDER X/Y AT ALL.
#
# mistral is excluded from every abstract condition BEFORE the run, not after:
# off-task 1.000 in exp3_mistral_abs and 1.000 again in exp4_mistral_abs_logit.
# Two independent inference stacks agree that Mistral emits prose rather than X
# or Y whenever a state block is present, so those cells contain no measurement.
# Paying for it a third time buys a third exclusion.
#
# The semantic no-history phase keeps all three: mistral is clean under
# Cooperate/Defect. It sits at the floor there (99.8% of episodes never defect)
# and will inform little, but a floor is an answer when the question is whether
# removing the refutation moves anything at all.
ABS_VIABLE="llama qwen"
abs_viable () { case " $ABS_VIABLE " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

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
# nothing to send, and the end-of-run guard sees HEAD == origin/main and reports
# "all artefacts pushed" while every database exists only on a machine that is
# about to be terminated.
git config user.email >/dev/null || {
    echo "  ABORT: git identity unset. The driver's commits would fail silently"
    echo "         and the completion check would report a false negative."
    echo "         Run:  git config user.email you@example.com"
    echo "               git config user.name  'Your Name'"
    exit 1; }
git remote get-url origin >/dev/null 2>&1 || {
    echo "  ABORT: no 'origin' remote. Nothing written here would survive."; exit 1; }
echo "  code at $(git rev-parse --short HEAD)" | tee -a "$LOG"

# The no-history condition is a PROMPT-RENDERING claim, and this repository has
# already shipped one experiment whose manipulation silently did not apply. So
# assert the flag's semantics on the CPU before renting anything: exactly one
# section removed, block byte-identical, exp1-exp6 untouched with it off.
python -m pytest tests/test_no_history.py tests/test_abstract_falsification.py \
    -q 2>&1 | tail -3 | tee -a "$LOG"

run_group () {
    local name=$1 model=$2 cond=$3 readout=$4 arms=$5 budget=$6; shift 6
    local tag="${TAG}_${name}_${cond}_${readout}"

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

    banner "$tag  ($model)  arms: $arms  readout: $readout  cond: $cond"
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

    # GATE 1 - falsification, inherited from exp6 unchanged. If
    # displayed_opponent_last is never populated the lying arms silently
    # rendered the truth and the whole run is a null by construction.
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

    # GATE 2 - THE NEW ONE. The history condition, checked against the stored
    # prompts rather than against the flag we passed.
    #
    # Two independent readings, because each covers what the other cannot:
    #
    #   prompt_full   the literal header. Direct and unambiguous, but only the
    #                 first 3 episodes of each cell store it.
    #   prompt_tokens turn-invariance. [HISTORY] is the ONLY section that grows
    #                 with the turn index - rules are constant, the block is
    #                 padded to the parity target, the instruction is fixed - so
    #                 "one distinct prompt length per (arm,opponent)" is
    #                 equivalent to "no history", and it covers 100% of rows.
    #
    # A run that passes one and fails the other is not interpretable either way;
    # both must agree.
    python3 - "$tag.sqlite" "${NOHIST:-0}" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sqlite3, sys
db, want_nohist = sys.argv[1], sys.argv[2] == "1"
c = sqlite3.connect(db)
seen = c.execute(
    "SELECT COUNT(*) FROM turn_details "
    "WHERE prompt_full IS NOT NULL AND prompt_full LIKE '%[HISTORY]%'"
).fetchone()[0]
stored = c.execute(
    "SELECT COUNT(*) FROM turn_details WHERE prompt_full IS NOT NULL"
).fetchone()[0]
widths = c.execute(
    "SELECT arm, opponent_policy, COUNT(DISTINCT prompt_tokens), COUNT(*) "
    "FROM turns GROUP BY arm, opponent_policy"
).fetchall()
bad = [f"{a}|{o}: {n} distinct prompt widths over {t} turns"
       for a, o, n, t in widths if (n != 1) is want_nohist]
if stored == 0:
    sys.exit("  ABORT: no prompt_full rows stored; the history gate cannot run.\n"
             "         --full-prompt-episodes must be > 0.")
if want_nohist and seen:
    sys.exit(f"  ABORT: --no-history requested but [HISTORY] appears in {seen} "
             f"of {stored} stored prompts.")
if not want_nohist and seen != stored:
    sys.exit(f"  ABORT: history expected but [HISTORY] is in only {seen} of "
             f"{stored} stored prompts. The condition applied by accident.")
if bad:
    sys.exit("  ABORT: prompt-width check disagrees with the history flag:\n    "
             + "\n    ".join(bad))
print(f"  history gate OK: [HISTORY] in {seen}/{stored} stored prompts, "
      f"prompt width {'constant' if want_nohist else 'grows'} across turns")
PYEOF

    gzip -kf "$tag.sqlite"
    touch "${tag}.done"
    git add -f "$tag.sqlite.gz" "$tag.json" "$LOG"
    git commit -m "exp7: $tag" || true
    git push || echo "  PUSH FAILED - do not terminate before this succeeds" | tee -a "$LOG"
}

for entry in "${MODELS[@]}"; do
    IFS=':' read -r name hf_id cache_dir <<< "$entry"

    # PHASE 1 - abstract framing, history intact.
    # Everything except the labels matches exp6's LOGIT cells, so
    # exp6_${name}_sem_logit IS the control for this phase. Do not re-run it.
    if [ "$MODE" = "abs" ] || [ "$MODE" = "both" ] || [ "$MODE" = "cross" ]; then
        if abs_viable "$name"; then
            NOHIST=0
            run_group "$name" "$hf_id" abs logit "$ARMS" "$LOGIT_BUDGET" \
                      --framing abstract
        else
            banner "$name: abstract conditions SKIPPED (off-task 1.000 in exp3 and exp4)"
        fi
    fi

    # PHASE 1b - swap, and it is not optional if phase 1 comes back ambiguous.
    #
    # THE CEILING PROBLEM WITH THE ABSTRACT CELL. Under X/Y both viable models
    # already defect near the top of the range with a block present: exp3 has
    # qwen_abs arm 3 at 0.771 (allc) / 0.770 (tft) with not one episode in 2,000
    # avoiding defection, and llama's abstract baselines at 0.71-0.74. 3-3m is
    # NEGATIVE - arm 3m defects MORE than arm 3 - so the largest effect
    # observable there is about 0.23, against a semantic effect of -0.4049. A
    # compressed estimate would look exactly like the lexical account winning.
    #
    # SWAP HAS NO SUCH PROBLEM, because the two accounts predict OPPOSITE SIGNS
    # rather than different magnitudes. With the mapping inverted, the word
    # "Cooperate" MEANS defect. Arm 3m flips the ACTION, so when the opponent
    # truly cooperated the block asserts a betrayal and renders it as the word
    # "Cooperate":
    #
    #   lexical priming  -> the cooperative token lowers the recorded defection
    #                       rate. 3-3m changes SIGN, from negative to positive.
    #   state conditioning -> the asserted betrayal still provokes retaliation.
    #                       3-3m keeps its sign.
    #
    # A sign test survives any monotone compression of the scale, which is what
    # a ceiling is. Ten cells, ~7 minutes, and it is the cheapest insurance in
    # this design against an uninterpretable phase 1.
    if [ "$MODE" = "swap" ] || [ "$MODE" = "cross" ]; then
        if abs_viable "$name"; then
            NOHIST=0
            run_group "$name" "$hf_id" swap logit "$ARMS" "$LOGIT_BUDGET" \
                      --swap-labels
        fi
    fi

    # PHASE 2 - no history, semantic labels.
    # exp6_${name}_sem_logit is again the control: same labels, same arms, same
    # N, same readout. The only difference is whether the raw log is in the
    # window to refute the block.
    if [ "$MODE" = "nohist" ] || [ "$MODE" = "both" ] || [ "$MODE" = "cross" ]; then
        NOHIST=1
        run_group "$name" "$hf_id" nohist logit "$ARMS" "$LOGIT_BUDGET" \
                  --no-history
    fi

    # PHASE 3 (cross) - abstract AND no history.
    # The only cell that can show an INTERACTION. If the last-move effect
    # survives X/Y while the history is present but dies once it is removed, the
    # two accounts are not additive and neither single-factor phase would show
    # it. Two models, LOGIT, ~13 min total.
    if [ "$MODE" = "cross" ] && abs_viable "$name"; then
        NOHIST=1
        run_group "$name" "$hf_id" absnohist logit "$ARMS" "$LOGIT_BUDGET" \
                  --framing abstract --no-history
    fi

    # PHASE 4 (pad) - no history under minimal CoT.
    # C7 records that the last-move effect does not survive reasoning in llama.
    # If the effect is really "the model checks the block against the log", then
    # reasoning is precisely when the check happens - so removing the log should
    # RESTORE it under CoT. That is a directional prediction no LOGIT phase can
    # make, and it is the strongest form of objection 2 that can be tested.
    # llama and qwen only: mistral's CoT effect grew rather than shrank, so it
    # has no collapse to explain.
    if [ "$MODE" = "pad" ] && abs_viable "$name"; then
        NOHIST=1
        run_group "$name" "$hf_id" nohist scratchpad "$PAD_ARMS" "$PAD_BUDGET" \
                  --no-history --scratchpad-prompt minimal
    fi

    # One model resident at a time. The volume quota is ~45G against ~46G of
    # weights for three models - this is the EDQUOT that killed two exp4 smokes.
    rm -rf "${HF_HOME:?}/hub/${cache_dir}" 2>/dev/null || true
    rm -f "${TAG}_${name}"_*.sqlite
    echo "  evicted $name  (cache now $(du -sh "$HF_HOME" 2>/dev/null | cut -f1))" | tee -a "$LOG"
done

banner "COMPLETE  ($MODE)"

python3 - <<'EOF' 2>&1 | tee -a "$LOG"
import json, glob

print("""
The two contrasts, side by side with exp6. 3-3m is the effect under test;
3-3s is the field that exp6 says does almost nothing.

  cond=abs        objection 1. If 3-3m collapses toward 3-3s here, the exp6
                  result was the token "Defect", not the last-move field.
  cond=nohist     objection 2. If 3-3m GROWS here, the block was being
                  discounted because [HISTORY] refuted it, and the finding is
                  about conflict resolution rather than about state use.
""")
print(f"{'group':30}{'opp':6}{'P(D|1)':>9}{'P(D|3b)':>9}{'P(D|3)':>9}"
      f"{'P(D|3s)':>9}{'P(D|3m)':>9}{'3-3s':>9}{'3-3m':>9}{'off':>7}{'cpr3':>7}")
for f in sorted(glob.glob('exp7_*_logit.json')) + sorted(glob.glob('exp6_*_logit.json')):
    d = json.load(open(f))
    for opp in ('allc', 'tft'):
        try:
            g = {a: d[f'{a}|{opp}'] for a in ('1', '3b', '3', '3s', '3m')}
        except KeyError:
            continue
        r = {a: v['defect_rate'] for a, v in g.items()}
        off = max(v['off_task_rate'] for v in g.values())
        print(f"{f[:-5]:30}{opp:6}{r['1']:>9.3f}{r['3b']:>9.3f}{r['3']:>9.3f}"
              f"{r['3s']:>9.3f}{r['3m']:>9.3f}{r['3']-r['3s']:>+9.3f}"
              f"{r['3']-r['3m']:>+9.3f}{off:>7.3f}{g['3']['cpr']:>7.3f}")
print("""
READ IN THIS ORDER.

1. off-task. Above 0.10 the row is prose, not decisions - and under X/Y that is
   the expected failure, not a surprise. mistral_abs was excluded twice for it
   already and is not in this run.

2. cpr3. In the nohist rows this is no longer a copy task: the model cannot read
   the answer off the log, so arm-3 CPR is the first real readability
   measurement in the project. If it stays at 1.000, readability was never
   carried by the history.

3. 3-3m in cond=abs against exp6's semantic value. The comparison is
   between-run only in labels - same arms, same N, same readout, same code.

4. 3-3m in cond=nohist against the same. GROWS = discount account. FLAT = the
   block simply is not used. SHRINKS = the effect needed the contradiction,
   which is the sharpest possible version of objection 2 landing.

Turn 0 rows in arm 3m carry no falsification (no last move to flip) and are
marked donor_degenerate=1. The rates above do NOT exclude them; re-estimate
with analysis/02_episode_level.py before quoting anything.""")
EOF

git add -f "$LOG"; git commit -m "exp7: session log" || true; git push || true

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
    echo "    git commit -m 'exp7: rescue' && git push" | tee -a "$LOG"
    exit 2
fi
echo "  all artefacts pushed ($(git ls-files "${TAG}_*.sqlite.gz" | wc -l | tr -d ' ') archives tracked)" \
    | tee -a "$LOG"

# IF ONLY ONE PHASE CAN BE AFFORDED, RUN nohist.
#
# Objection 1 can lose and the paper survives. If the effect turns out lexical,
# C5 becomes a claim about token priming - which is a real finding this project
# is already equipped to make, because D2 established the same thing for the
# container effect from the same data. The paper reframes.
#
# Objection 2 cannot lose that way. It says the manipulation was never the
# manipulation described: with [HISTORY] one section below, arms 3c/3s/3m
# falsify nothing, they CONTRADICT. If that is right, every field-level sentence
# in C3, C5, C6 and C8 is mis-stated, and the ACL reviewer's alternative account
# explains the whole corpus including the asymmetry that is the headline.
# It is also the objection that two of the three reviewers raised as a single
# required change, and the AC's W1.
#
# It is cheaper (one flag, no framing risk), it covers all three models rather
# than two, it cannot be lost to an off-task exclusion, and it is the one whose
# failure mode is unrecoverable. Run nohist first.
