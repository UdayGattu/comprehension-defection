#!/usr/bin/env bash
# EXPERIMENT 8 - does the instrument survive being pointed somewhere else?
#
# THE OBJECTION, RAISED INDEPENDENTLY BY TWO REFEREES AND NOW THE PAPER'S SPINE
#   The paper has been reframed as a METHODS contribution: "prompt ablations are
#   uninterpretable without a token- and density-matched placebo; here is the
#   instrument." A method that has only ever been run on ONE prompt template is
#   not an instrument. It is one observation.
#
#   Seven experiments, ~300,000 episodes, and every single one of them used:
#     - one [STATE] template  (Your score / Opponent score / Opponent's last
#       move / Rounds played)
#     - one field order       (that one)
#     - one insertion position (insertion_index=1, after the rules)
#
#   The repository already says why that is not safe. cdx/config.py's own note
#   on insertion_index: "lost-in-the-middle effects produce >30% swings from
#   position alone". No driver in exp1-exp7 ever set it to anything but 1; the
#   only non-default use anywhere is a single test. And CLAIMS.md D2 measured a
#   purely LEXICAL change REVERSING the sign of the container effect in Qwen -
#   from this project's own data. So the field-level asymmetry that is now the
#   headline (C5: the last-move field dominates, the score does not) has been
#   measured at exactly one point in a three-dimensional space where this
#   project has already found sign reversals.
#
# ---------------------------------------------------------------------------
# THE THREE FACTORS
# ---------------------------------------------------------------------------
#   TEMPLATE   original  vs  reworded
#              Your score / Opponent score / Opponent's last move / Rounds
#              played   ->   Your cumulative points / Their cumulative points /
#              Their previous choice / Rounds elapsed.
#              Same four fields, same four values, no shared content word.
#              Each template carries its OWN placebo bodies and its OWN parity
#              target - see the note on _derive_block_tokens. The original's
#              target does not move; that is pinned by
#              tests/test_template_family.py and it is the reason exp1-exp7
#              still reproduce from HEAD.
#
#   ORDER      canonical  vs  permuted
#              (score, opp score, last move, rounds) -> (last move, rounds,
#              opp score, score). Both falsifiable fields move as far as a
#              four-field block allows: the score first->last, the last move
#              third->first. This is the only permutation that separates "the
#              last-move FIELD dominates" from "the THIRD LINE dominates" - a
#              serial-position account nothing in exp1-exp7 can rule out.
#
#   POSITION   insertion_index=1  vs  2
#              After the rules and before [HISTORY], vs after [HISTORY] and
#              before the instruction. Position 2 is the one the config file
#              warns about and no experiment has ever run.
#
# ---------------------------------------------------------------------------
# THE FRACTION, AND WHY IT IS A FRACTION
# ---------------------------------------------------------------------------
#   Full crossing is 2x2x2 = 8 conditions x 10 cells x 3 models = 240 cells.
#   That is ~3h15 and ~$10 on an H100 - it is NOT dollar-limited at a $40
#   budget, and the pre-registration says so rather than pretending otherwise.
#   The fraction is chosen for two other reasons and both are stated:
#
#     1. MULTIPLICITY. 8 conditions x 3 models x 2 opponents = 48 stability
#        verdicts on one claim. At 48 independent looks, a 5% test yields ~2.4
#        spurious "instabilities" by construction, and the falsification rule
#        in PREREGISTRATION_EXP8.md is a WORST-CASE rule over conditions - it
#        fires if ANY condition breaks the band. A worst-case rule over 8
#        conditions is a strictly harsher test than the same rule over 4, for
#        reasons that have nothing to do with the instrument.
#
#     2. THE SPARE BUDGET BUYS MORE ELSEWHERE. The freed ~$15 is pre-committed
#        to MODE=pad (a SCRATCHPAD replication, the regime the prior literature
#        actually runs in and the regime C7 says attenuates the effect), which
#        no amount of extra LOGIT conditions can substitute for.
#
#   THE DESIGN. A 2^(3-1) half fraction, replicated on all three models, plus
#   its FOLDOVER on the primary model only.
#
#     Half A, defining relation I = -TOP:
#         anchor         original           canonical  pos 1     (-,-,-)
#         origpermp2     original           permuted   pos 2     (-,+,+)
#         rewordp2       reworded           canonical  pos 2     (+,-,+)
#         rewordpermp1   reworded           permuted   pos 1     (+,+,-)
#
#     Half B (the foldover), I = +TOP:
#         origp2         original           canonical  pos 2     (-,-,+)
#         origpermp1     original           permuted   pos 1     (-,+,-)
#         rewordp1       reworded           canonical  pos 1     (+,-,-)
#         rewordpermp2   reworded           permuted   pos 2     (+,+,+)
#
#   WHY HALF A IS THE HALF WE KEEP. It contains (-,-,-), which is exp6's
#   condition exactly - same arms, same opponents, same N, same readout, same
#   labels. That makes the anchor a WITHIN-SESSION control rather than a
#   between-run comparison to a database from a different vLLM image. B5
#   measured perturbation contrasts moving 4pp across stacks and the
#   block-vs-no-block contrast is the stack-fragile one, so the anchor is
#   re-run, not inherited. It costs 3 groups and it removes the only
#   between-run difference in the whole design.
#
#   WHAT THE HALF COSTS, STATED PLAINLY. In a 2^(3-1) resolution-III design
#   each main effect is aliased with the complementary two-factor interaction:
#       T = -OP      O = -TP      P = -TO
#   So on llama and mistral, "the template moved the asymmetry" and "order and
#   position interact" are the same number. The foldover on qwen breaks every
#   one of those aliases (a foldover of a resolution-III design always does),
#   giving the full 2^3 on the model with the largest and best-measured effect
#   - E_move = -0.4049 vs TFT, and the model where D2 already found a lexical
#   sign reversal, i.e. the model most likely to be fragile. The licensing
#   assumption is stated in the pre-registration and it is the standard one:
#   the three-factor interaction TOP is negligible. If qwen's foldover shows
#   ANY two-factor interaction outside the stability band, MODE=full runs the
#   foldover on llama and mistral too and the aliases are broken everywhere.
#
# ---------------------------------------------------------------------------
# THE ESTIMAND
# ---------------------------------------------------------------------------
#   A = P(defect | 3m) - P(defect | 3s)
#
#   The field asymmetry, as a single number, with arm 3 cancelled out of it.
#   E_move - E_score = [P(3)-P(3m)] - [P(3)-P(3s)] = P(3s) - P(3m), so A is the
#   negated version and A > 0 means "falsifying the move moves defection more
#   than falsifying the score, in the direction exp6 found".
#
#   Why this form and not the pair (3-3m, 3-3s): 3s and 3m are BYTE-IDENTICAL
#   except for one line, padded to the same target, at the same position, in the
#   same template. Every between-condition nuisance - block width, template
#   verbosity, position - is common to both and cancels exactly. A cross-
#   template comparison of A is therefore legitimate even though the two
#   templates have different parity targets and their prompts are different
#   widths. A cross-template comparison of a RAW defect rate would not be, and
#   the pre-registration says so.
#
# ---------------------------------------------------------------------------
#   Calibration: LOGIT 0.667 min/cell (8 min/model for 12 cells at N=1000 on an
#   H100), 10 cells per group, +2 min group overhead, +4 min first download per
#   model, $3/h.
#
#   MODE=anchor  bash scripts/exp8_templates.sh   #  3 groups   30 cells ~38m  $1.90
#   MODE=half    bash scripts/exp8_templates.sh   # 12 groups  120 cells ~1h56 $5.80
#   MODE=fold    bash scripts/exp8_templates.sh   #  4 groups   40 cells ~39m  $1.95
#   MODE=screen  bash scripts/exp8_templates.sh   # 16 groups  160 cells ~2h31 $7.55
#   MODE=full    bash scripts/exp8_templates.sh   # 24 groups  240 cells ~3h40 $11.00
#   MODE=pad     bash scripts/exp8_templates.sh   #  6 groups   60 cells ~4h09 $12.45
#
# RECOMMENDED SCOPE: MODE=screen in one session, ~2h31, $7.55 ($9.45 at +25%).
# Hold MODE=full (+$3.45 over screen) for the case where qwen's foldover flags a
# two-factor interaction, and MODE=pad ($12.45) for the CoT regime. The whole
# programme is $23.45 point / $29.30 at +25%, inside a $40 budget with ~$10 of
# headroom. Arithmetic in PREREGISTRATION_EXP8.md section 8.
#
# RUN ORDER IF ONLY ONE IS AFFORDABLE: half. Reasoning in the closing banner.

set -euo pipefail

DB_FS=$(df -T . 2>/dev/null | tail -1 | awk '{print $2}')
case "$DB_FS" in
    fuse*|nfs*|*moose*|*mfs*|9p)
        echo "ABORT: working directory is on '$DB_FS'. SQLite WAL is"
        echo "       unsupported there. Run from local disk (/root)."
        exit 1 ;;
esac

MODE=${MODE:-screen}
case "$MODE" in
    anchor|half|fold|screen|full|pad) ;;
    *) echo "ABORT: MODE must be anchor, half, fold, screen, full or pad (got '$MODE')"
       exit 1 ;;
esac

EPISODES=${EPISODES:-1000}
LOGIT_BUDGET=${LOGIT_BUDGET:-30}
PAD_BUDGET=${PAD_BUDGET:-90}
SCRATCH_TOKENS=${SCRATCH_TOKENS:-128}
TAG=exp8
LOG=exp8_session.log
RULE="=============================================================="

# Same five arms in every condition, so every cell is directly comparable to
# every other AND to exp6's semantic LOGIT cells. 3c stays dropped for exp7's
# reason: measured at 0.0000 dose vs ALLC, a weak instrument whose two fields
# the deliberate arms measure at a known dose.
ARMS="1 3b 3 3s 3m"
PAD_ARMS="1 3b 3 3s 3m"

# cond : state-template : insertion-index
# Half A - I = -TOP. Contains the anchor, which is exp6's condition exactly.
HALF_A=(
  "anchor:original:1"
  "origpermp2:original_permuted:2"
  "rewordp2:reworded:2"
  "rewordpermp1:reworded_permuted:1"
)
# Half B - the foldover, I = +TOP. Breaks every main-effect/2fi alias when run
# alongside Half A on the same model.
HALF_B=(
  "origp2:original:2"
  "origpermp1:original_permuted:1"
  "rewordp1:reworded:1"
  "rewordpermp2:reworded_permuted:2"
)
# The two design points furthest apart in the factor space: the anchor, and
# every factor flipped. The CoT phase runs only these, because SCRATCHPAD is
# 5.6x the cost per cell and a two-point contrast is what it can afford.
PAD_CONDS=("anchor:original:1" "rewordpermp2:reworded_permuted:2")

MODELS=(
  "llama:meta-llama/Llama-3.1-8B-Instruct:models--meta-llama--Llama-3.1-8B-Instruct"
  "qwen:Qwen/Qwen2.5-7B-Instruct:models--Qwen--Qwen2.5-7B-Instruct"
  "mistral:mistralai/Mistral-7B-Instruct-v0.3:models--mistralai--Mistral-7B-Instruct-v0.3"
)

# THE PRIMARY MODEL, FIXED BEFORE THE RUN.
#
# qwen carries the foldover in MODE=screen. Chosen on three measured facts, all
# pre-existing: the largest last-move effect in the corpus (E_move = -0.4049 vs
# TFT), the cleanest off-task record under semantic LOGIT, and - decisively -
# it is the model in which D2 found a purely lexical change REVERSING the sign
# of the container effect. If any model's field asymmetry is going to move
# under a reworded template, the prior says it is this one. Putting the full
# 2^3 anywhere else would be spending the de-aliasing budget on the model least
# likely to need it.
PRIMARY=qwen

# exp8 runs SEMANTIC labels with [HISTORY] PRESENT throughout. That is not a
# limitation, it is the point: the anchor has to be exp6's condition or the
# whole design has two differences in it. Abstract framing and --no-history are
# exp7's factors and they are orthogonal to these three; crossing them here
# would be a 32-cell design answering neither question well.
NOHIST=0

banner() { echo | tee -a "$LOG"; echo "$RULE" | tee -a "$LOG"; \
           echo "  $*" | tee -a "$LOG"; echo "$RULE" | tee -a "$LOG"; }

banner "PREFLIGHT  mode=$MODE  episodes=$EPISODES  primary=$PRIMARY"

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

# exp8 is a PROMPT-RENDERING experiment end to end, and this repository has
# already shipped one experiment whose manipulation silently did not apply.
# Assert the whole family on the CPU before renting anything: exp1-exp7
# byte-identical under the default, each template at parity within itself,
# one-line falsification under BOTH templates, position 2 where intended.
python -m pytest tests/test_template_family.py \
    tests/test_exp1_to_exp5_unchanged.py tests/test_field_falsification.py \
    tests/test_no_history.py -q 2>&1 | tail -3 | tee -a "$LOG"

# And print the targets the stub derives, so the log carries the numbers a
# reader would otherwise have to trust. These are CharTokenizer values; the
# real per-model targets are printed by gpu_run.py at STEP 2 of every group.
python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys
sys.path.insert(0, ".")
from cdx.config import ScaffoldConfig
from cdx.scaffold import STATE_TEMPLATES, ScaffoldBuilder
class Char:
    def encode(self, t, add_special_tokens=False): return [ord(c) for c in t]
    def decode(self, i): return "".join(chr(c) for c in i)
print("  template family (CharTokenizer parity targets - per template, never shared):")
for name in sorted(STATE_TEMPLATES):
    b = ScaffoldBuilder(Char(), ScaffoldConfig(), state_template=name)
    print(f"    {name:20} target={b.block_tokens:4}  "
          f"{' | '.join(STATE_TEMPLATES[name].labels)}")
assert ScaffoldBuilder(Char(), ScaffoldConfig()).block_tokens == 119, \
    "the original template's parity target moved - exp1-exp7 no longer reproduce"
print("  original target unchanged at 119 -> exp1-exp7 reproduce from HEAD")
PYEOF

run_group () {
    local name=$1 model=$2 cond=$3 template=$4 index=$5 readout=$6 arms=$7 budget=$8
    shift 8
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

    banner "$tag  ($model)  template: $template  index: $index  arms: $arms  readout: $readout"
    python scripts/gpu_run.py \
        --model "$model" \
        --episodes "$EPISODES" \
        --arms $arms \
        --opponents tft allc \
        --readout "$readout" \
        --framing semantic \
        --state-template "$template" \
        --insertion-index "$index" \
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

    # GATE 0 - COMPLETENESS. Runs FIRST because every gate below it passes on
    # partial data.
    #
    # --budget-minutes is CAUGHT inside gpu_run.py, not raised: on overrun it
    # prints STOPPED, writes whatever cells finished, and exits 0. Without this
    # gate the row check passes, the falsification check passes, the
    # manipulation gate passes, `.done` is written, the truncated database is
    # committed and pushed, and every later run skips the group as complete.
    # What goes missing is whole arm x opponent cells: lose 3m|tft or 3s|tft and
    # the primary estimand A = P(D|3m) - P(D|3s) has no value there, with
    # nothing downstream able to tell it was ever meant to exist.
    #
    # The expected count is DERIVED from the arms actually passed rather than
    # hardcoded, because MODE=pad runs its own ladder and this gate has to hold
    # there too. x2 for --opponents tft allc, which is fixed in run_group.
    local want_cells got_cells
    want_cells=$(( $(echo $arms | wc -w) * 2 ))
    got_cells=$(python3 -c "
import json
try:
    print(len(json.load(open('$tag.json'))))
except Exception:
    print(0)")
    [ "$got_cells" -eq "$want_cells" ] || {
        echo "  ABORT: $tag wrote $got_cells/$want_cells cells." | tee -a "$LOG"
        echo "         Budget truncation is the likely cause (budget was $budget min)." | tee -a "$LOG"
        echo "         Nothing was marked .done. Re-run the SAME command: completed" | tee -a "$LOG"
        echo "         groups skip on .done and this group resumes at the missing" | tee -a "$LOG"
        echo "         cells from $tag.json - do not delete that file." | tee -a "$LOG"
        echo "         If it stops here twice, raise the budget instead of retrying:" | tee -a "$LOG"
        echo "           LOGIT_BUDGET=60 MODE=$MODE bash scripts/exp8_templates.sh" | tee -a "$LOG"
        exit 1; }
    echo "  $tag: $got_cells/$want_cells cells present" | tee -a "$LOG"

    # GATE 1 - falsification, inherited from exp6/exp7 unchanged. If
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

    # GATE 2 - THE MANIPULATION GATE. Template, order and position, all three
    # read back off the STORED PROMPTS rather than off the flags we passed.
    #
    # This is the gate exp8 exists to have. A run that inherited the default
    # template while its run_id said 'reworded' is a clean replication of exp6
    # mislabelled as a generalisation test - it produces a beautiful, wrong
    # answer and nothing downstream can tell. Same for a permutation tuple that
    # never reached the renderer, and same for an insertion index that landed
    # the block somewhere the config file says is worth >30% on its own.
    #
    # Four independent readings, and they must all agree:
    #   template  every declared label present, no foreign label present
    #   order     label byte offsets inside [STATE] are monotonically increasing
    #   position  [STATE] before/after [HISTORY] as insertion_index demands
    #   parity    ONE distinct scaffold_tokens across every block arm, and one
    #             distinct prompt width per (opponent, turn) across block arms -
    #             on 100% of rows, not on the three that store prompt_full
    python3 - "$tag.sqlite" "$template" "$index" "${NOHIST:-0}" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sqlite3, sys
sys.path.insert(0, ".")
from cdx.scaffold import HISTORY_HEADER, STATE_HEADER, STATE_TEMPLATES

db, tname, index, want_nohist = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
tmpl = STATE_TEMPLATES[tname]
own = list(tmpl.labels)
foreign = sorted({l for t in STATE_TEMPLATES.values() for l in t.labels} - set(own))

# Arms that RENDER THE STATE TEMPLATE, and therefore must carry its labels.
#
# 3b (non-diagnostic) and 3d (syntactic) are placebo bodies: 3b renders
# "Round parity: ...", "Interaction type: repeated"; 3d renders "<node attr />".
# NEITHER contains a single template label, by construction - that is what makes
# them placebos. An earlier version of this gate demanded every label in every
# prompt containing [STATE], which arm 3b can never satisfy, and it aborted a
# run whose data was entirely correct. Label and ORDER are checked only on the
# arms that render the template; POSITION and FOREIGN-LABEL are checked on every
# block arm, because those must hold for placebos too.
STATE_ARMS = {"3", "3s", "3m", "3c"}

c = sqlite3.connect(db)
rows = c.execute(
    "SELECT arm, prompt_full FROM turn_details WHERE prompt_full IS NOT NULL"
).fetchall()
if not rows:
    sys.exit("  ABORT: no prompt_full rows stored; the manipulation gate cannot "
             "run. --full-prompt-episodes must be > 0.")

n_labelled = 0
for arm, p in rows:
    if STATE_HEADER not in p:
        continue                      # arm 1 injects no block
    at = p.index(STATE_HEADER)
    # Foreign labels must not appear in ANY block arm, placebo included.
    for lab in foreign:
        if f"{lab}:" in p:
            sys.exit(f"  ABORT: template {tname!r} is active but foreign label "
                     f"{lab!r} appears in a stored prompt (arm {arm}).")
    if arm in STATE_ARMS:
        for lab in own:
            if f"{lab}:" not in p:
                sys.exit(f"  ABORT: template {tname!r} declares {lab!r} but it "
                         f"is absent from a stored prompt for arm {arm}. The "
                         f"template did not apply.")
        offsets = [p.index(f"{lab}:", at) for lab in own]
        if offsets != sorted(offsets):
            sys.exit(f"  ABORT: field order {tmpl.field_order} declared but the "
                     f"stored prompt for arm {arm} renders {own} at offsets "
                     f"{offsets}.")
        n_labelled += 1
    if not want_nohist:
        if HISTORY_HEADER not in p:
            sys.exit("  ABORT: history expected but [HISTORY] is missing from a "
                     "stored prompt.")
        before = p.index(STATE_HEADER) < p.index(HISTORY_HEADER)
        if index == 1 and not before:
            sys.exit("  ABORT: insertion_index=1 but [STATE] renders AFTER "
                     "[HISTORY] in a stored prompt.")
        if index == 2 and before:
            sys.exit("  ABORT: insertion_index=2 but [STATE] renders BEFORE "
                     "[HISTORY] in a stored prompt.")

widths = c.execute(
    "SELECT COUNT(DISTINCT scaffold_tokens) FROM turns WHERE arm != '1'"
).fetchone()[0]
if widths != 1:
    sys.exit(f"  ABORT: {widths} distinct scaffold_tokens across the block arms. "
             f"Token parity - the single property every causal claim in this "
             f"project rests on - did not hold.")

# TURN 0 ONLY, and the restriction is load-bearing.
#
# At turn 0 [HISTORY] is empty, so every block arm must render to the same
# width and any difference IS a parity violation. From turn 1 the agent's own
# past actions enter the history; once behaviour diverges across arms - which
# is the entire point of the experiment - the histories differ in text, and
# wherever the tokeniser gives "Cooperate" and "Defect" different lengths they
# differ in token count too.
#
# MEASURED, not argued. Run against the committed exp6 databases, the
# unrestricted form fails 36 of 40 (opponent, turn) groups on mistral - it
# rejects the very data C5 is built on - while passing 0 of 40 on qwen, whose
# action labels happen to tokenise to equal length. A gate whose verdict
# depends on the vocabulary of the model under test is not a gate. Block parity
# is already established above by scaffold_tokens, which holds on 100% of rows
# in both. Restricted to turn 0 this check still catches a real break: injecting
# +7 tokens into arm 3b at turn 0 in exp6 qwen is detected.
bad = [f"{o}|turn {t}: {n} distinct prompt widths across block arms at turn 0"
       for o, t, n in c.execute(
           "SELECT opponent_policy, turn, COUNT(DISTINCT prompt_tokens) "
           "FROM turns WHERE arm != '1' AND turn = 0 "
           "GROUP BY opponent_policy, turn")
       if n != 1]
if bad:
    sys.exit("  ABORT: prompts differ in width across arms at the same turn:\n    "
             + "\n    ".join(bad[:10]))

grew = c.execute(
    "SELECT arm, opponent_policy, COUNT(DISTINCT prompt_tokens) FROM turns "
    "GROUP BY arm, opponent_policy").fetchall()
flat = [f"{a}|{o}" for a, o, n in grew if (n == 1) is not want_nohist]
if flat:
    sys.exit(f"  ABORT: prompt width does not track the turn index in {flat[:6]}, "
             f"but history is ON. [HISTORY] is not rendering rounds.")

# Without this the gate passes vacuously if no state-rendering arm ever stored a
# prompt - the label and order readings would simply never execute.
if n_labelled == 0:
    sys.exit(f"  ABORT: no stored prompt came from a state-rendering arm "
             f"({sorted(STATE_ARMS)}); the template and order checks never ran.")

print(f"  manipulation gate OK: template={tname} order={'->'.join(own)} "
      f"index={index}, {len(rows)} stored prompts ({n_labelled} label-checked), "
      f"one block width across all arms and all turns")
PYEOF

    gzip -kf "$tag.sqlite"
    touch "${tag}.done"
    git add -f "$tag.sqlite.gz" "$tag.json" "$LOG"
    git commit -m "exp8: $tag" || true
    git push || echo "  PUSH FAILED - do not terminate before this succeeds" | tee -a "$LOG"
}

run_conditions () {
    # $1 = model short name, $2 = hf id, $3.. = "cond:template:index" entries
    local name=$1 hf_id=$2; shift 2
    local entry cond template index
    for entry in "$@"; do
        IFS=':' read -r cond template index <<< "$entry"
        run_group "$name" "$hf_id" "$cond" "$template" "$index" \
                  logit "$ARMS" "$LOGIT_BUDGET"
    done
}

for entry in "${MODELS[@]}"; do
    IFS=':' read -r name hf_id cache_dir <<< "$entry"

    # PHASE A - the half fraction, on every model. Includes the anchor, which is
    # exp6's condition re-run in-session so that no contrast in the design is
    # between-stack (B5: 4pp of movement across images).
    if [ "$MODE" = "half" ] || [ "$MODE" = "screen" ] || [ "$MODE" = "full" ]; then
        run_conditions "$name" "$hf_id" "${HALF_A[@]}"
    fi

    # MODE=anchor - the anchor alone. Cheapest possible answer to "has the stack
    # drifted since exp6", and the thing to run first if a session is going to
    # be interrupted.
    if [ "$MODE" = "anchor" ]; then
        run_conditions "$name" "$hf_id" "${HALF_A[0]}"
    fi

    # PHASE B - the foldover, on the primary model only in MODE=screen. Adding
    # it to Half A gives the FULL 2^3 on qwen: every main effect and every
    # two-factor interaction clear of aliases. On llama and mistral the half
    # stands alone at resolution III (T = -OP, O = -TP, P = -TO), which is
    # sufficient for the stability verdict the pre-registration actually asks
    # for and insufficient for attributing an instability to one factor. That
    # trade is the fraction, stated.
    if [ "$MODE" = "fold" ] || [ "$MODE" = "screen" ]; then
        if [ "$name" = "$PRIMARY" ]; then
            run_conditions "$name" "$hf_id" "${HALF_B[@]}"
        fi
    fi

    # PHASE C - the foldover everywhere. Pre-committed, CONDITIONAL: run this
    # only if qwen's 2^3 shows a two-factor interaction on A outside the
    # stability band, i.e. only if the aliases on llama and mistral are known to
    # matter. Running it unconditionally would be the full factorial and the
    # fraction would have bought nothing.
    if [ "$MODE" = "full" ]; then
        run_conditions "$name" "$hf_id" "${HALF_B[@]}"
    fi

    # PHASE D - SCRATCHPAD, two conditions, the two furthest apart in the factor
    # space. C7 records that the last-move effect does not survive reasoning in
    # llama and shrinks 50-87% in qwen, so the CoT regime is where this
    # instrument is WEAKEST - and it is the regime the prior literature uses. If
    # the asymmetry is going to be template-dependent anywhere it is here.
    # mistral is included: it sits at the floor under semantic labels, and the
    # pre-registration's floor rule says a floored cell casts no vote, so its
    # inclusion costs a group and cannot corrupt a verdict.
    if [ "$MODE" = "pad" ]; then
        for pentry in "${PAD_CONDS[@]}"; do
            IFS=':' read -r cond template index <<< "$pentry"
            run_group "$name" "$hf_id" "$cond" "$template" "$index" \
                      scratchpad "$PAD_ARMS" "$PAD_BUDGET" \
                      --scratchpad-prompt minimal
        done
    fi

    # One model resident at a time. The volume quota is ~45G against ~46G of
    # weights for three models - this is the EDQUOT that killed two exp4 smokes.
    rm -rf "${HF_HOME:?}/hub/${cache_dir}" 2>/dev/null || true
    rm -f "${TAG}_${name}"_*.sqlite
    echo "  evicted $name  (cache now $(du -sh "$HF_HOME" 2>/dev/null | cut -f1))" \
        | tee -a "$LOG"
done

banner "COMPLETE  ($MODE)"

python3 - <<'EOF' 2>&1 | tee -a "$LOG"
import json, glob, re

print("""
A = P(D|3m) - P(D|3s), the field asymmetry with arm 3 cancelled out of it.
3s and 3m are byte-identical except one line, at the same width and the same
position, so every between-condition nuisance is common to both and cancels.
S = A(cond) / A(anchor) within the same model and opponent.

The anchor is exp6's condition, re-run in this session. If A(anchor) does not
reproduce exp6 to within its CI, the stack has drifted and NOTHING below is a
statement about templates - see B5.
""")
print(f"{'group':38}{'opp':6}{'P(D|3)':>9}{'P(D|3s)':>9}{'P(D|3m)':>9}"
      f"{'A':>9}{'S':>8}{'off':>7}{'cpr3':>7}")

rows = {}
for f in sorted(glob.glob('exp8_*_logit.json')) + sorted(glob.glob('exp8_*_scratchpad.json')):
    d = json.load(open(f))
    m = re.match(r'exp8_([a-z]+)_([a-z0-9]+)_([a-z]+)\.json$', f)
    model, cond = (m.group(1), m.group(2)) if m else (f, f)
    for opp in ('allc', 'tft'):
        try:
            g = {a: d[f'{a}|{opp}'] for a in ('1', '3b', '3', '3s', '3m')}
        except KeyError:
            continue
        r = {a: v['defect_rate'] for a, v in g.items()}
        A = r['3m'] - r['3s']
        rows[(model, cond, opp)] = A
        anchor = rows.get((model, 'anchor', opp))
        if anchor is None:
            S = "  n/a"
        elif abs(anchor) < 0.05:
            S = " FLOOR"          # never divide by a near-zero anchor
        else:
            S = f"{A / anchor:6.2f}"
        off = max(v['off_task_rate'] for v in g.values())
        print(f"{f[:-5]:38}{opp:6}{r['3']:>9.3f}{r['3s']:>9.3f}{r['3m']:>9.3f}"
              f"{A:>+9.3f}{S:>8}{off:>7.3f}{g['3']['cpr']:>7.3f}")

print("""
READ IN THIS ORDER, AND READ TFT FIRST.

0. THE AGGREGATION RULE IS BINDING AND IT IS IN PREREGISTRATION_EXP8.md S5.
   Opponents are NEVER pooled and never averaged. Every verdict is a pair
   (TFT, ALLC). Where one verdict is required, TFT decides and ALLC is reported
   without a vote - under ALLC the opponent cooperates unconditionally, so the
   true last move is constant and the field is not diagnostic for optimal play.
   Counting ALLC as an independent vote lets a mechanically attenuated cell
   outvote the cell the claim is about. exp7 left this unstated and the two
   readings of its condition (b) disagree; that is a declared defect and it is
   not repeated here.

1. THE ANCHOR, against exp6_${model}_sem_logit. Same arms, same N, same
   readout, same labels, same code path. If it has moved, stop.

2. off-task. Above 0.10 in arms 1, 3b or 3 the group is prose, not decisions,
   and it is excluded from causal claims (B4) - which changes the DENOMINATOR
   of the "k of 3 models" rule, per the pre-registration's exclusion clause.

3. FLOOR / CEILING. A contrast cell with max P(D) over {3,3s,3m} <= 0.05 is at
   a FLOOR: reported in full, casts NO vote in either direction, and never
   appears in a denominator. mistral under semantic labels is the expected
   case (99.8% of episodes never defect in exp7). If the ANCHOR is floored, S
   is undefined for that model+opponent and the model is excluded for it. S is
   NEVER computed when |A(anchor)| < 0.05.

4. S per condition. Band is [0.50, 2.00] with the anchor's sign. Worst case
   over conditions: a model is UNSTABLE if ANY of its non-anchor, non-floored
   conditions leaves the band with a CI on A(cond) - A(anchor) excluding 0.
   Conditions are not averaged - an average would let six stable conditions
   hide one that reverses, which is exactly the failure D2 found in Qwen.

5. THE FACTORIAL, on qwen only in MODE=screen. Main effects on A:
       dT = mean A over reworded - mean A over original
       dO, dP likewise.
   On qwen these are clear. On llama and mistral they are aliased with the
   complementary two-factor interaction (T = -OP, O = -TP, P = -TO) and must be
   quoted as "T + (-OP)", never as "T".

Turn 0 rows in arm 3m carry no falsification (no last move to flip) and are
marked donor_degenerate=1. The rates above do NOT exclude them; re-estimate
with analysis/02_episode_level.py before quoting anything.""")
EOF

git add -f "$LOG"; git commit -m "exp8: session log" || true; git push || true

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
    echo "    git commit -m 'exp8: rescue' && git push" | tee -a "$LOG"
    exit 2
fi
echo "  all artefacts pushed ($(git ls-files "${TAG}_*.sqlite.gz" | wc -l | tr -d ' ') archives tracked)" \
    | tee -a "$LOG"

# IF ONLY ONE PHASE CAN BE AFFORDED, RUN half.
#
# MODE=half is the only mode that produces a verdict on all three models, and
# the claim under test is a claim about the INSTRUMENT, not about qwen. A
# foldover on one model with no half on the others answers "which factor moved
# it" for a question nobody has yet established has an answer.
#
# It also contains the anchor. The anchor is the only cell in the design that
# can invalidate every other cell: if exp6's condition does not reproduce in
# this session, the differences below are stack drift wearing a template
# costume, and MODE=fold would spend 28 minutes measuring it without ever
# noticing.
#
# MODE=fold is worth running second and only second. Its entire value is
# de-aliasing, and there is nothing to de-alias until the half fraction has
# shown something moved.
