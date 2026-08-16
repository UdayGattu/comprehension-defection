# Experiment registry

Every run that produced data, what it tested, what changed since the previous
one, and what it found. Runs are **append-only**: nothing here is edited to
match a later result, and no run is deleted because a better one exists.

`PREREGISTRATION.md` is Experiment 1's and is never modified.

| run | tag / commit | status | models | cells |
|---|---|---|---|---|
| exp1 | `exp1-frozen` · data `823313e` · code `8aad06f` | **pre-registered, closed** | Llama | 6 |
| exp2 | `f880b9a` `2574f3c` `c282959` · code `29e3aa7`, `2192dd9` | exploratory | Llama, Qwen | 28 |
| exp3 | `91acf27` | exploratory, confirmatory for exp2 | Llama, Qwen, Mistral | 90 |
| exp4 | `249c486` | exploratory — chain-of-thought ablation | Llama, Qwen, Mistral | 72 |
| exp5 | `ecd813a` | control for exp4's instruction confound | Llama, Qwen, Mistral | 18 |
| exp6 | driver `scripts/exp6_fields.sh` · code TODO | exploratory — field-level falsification; **carries the headline** | Llama, Qwen, Mistral | 60 |

exp6 was missing from this table while having a full section below. It is the
run that turns exp3's strongest mechanism from an inference into a measurement
(`turns.displayed_opponent_last`, closing known defect 4) and it supplies
`CLAIMS.md` C3, C5, C6, C7 and C8 — the last-move-dominates result, the
100%-echo-of-a-false-score result, and the CoT limitation. A registry that omits
the run carrying the headline is the registry failing at its one job.

> TODO (author): fill in exp6's commit hash. Every other row cites one; this row
> cannot, because the code-only checkout this table was audited from carries no
> git history. Read it from the data itself:
> `SELECT DISTINCT git_commit FROM run_meta;` on any `exp6_*.sqlite`.

**exp7 has code and a pre-registration but no row, deliberately.**
`PREREGISTRATION_EXP7.md` is frozen before data, on the unchanged probe suite;
`scripts/exp7_confounds.sh`,
`tests/test_no_history.py`, `tests/test_abstract_falsification.py`,
`analysis/14_reviewer_responses.py` and `PromptAssembler.assemble
(include_history=...)` are all present and their gates pass. No row appears above
because this table is append-only over **runs that produced data**, and exp7 has
not run. Adding a row for a planned run is how a registry starts describing
intentions instead of history.

exp7 tests two confounds on exp6's last-move result: lexical priming (arm 3m
injects the literal token `Defect`, and D2 shows labels dominating every other
manipulation) and redundancy with `[HISTORY]` (which has always rendered every
round one section below the block, making arms 3c/3s/3m *contradiction*
manipulations rather than false-state ones).

> TODO (author): when exp7 runs, add its row and section here first, then its
> claims to `CLAIMS.md`, then its rows to `CLAIM_MAP.md`. Note also that
> `--no-history` is not captured by `config_fingerprint` — it is an argument to
> `assemble`, not a config field — so exp7's two phases will be distinguishable
> only by `run_id`, `run_meta.argv` and `prompt_full`. That is the same class as
> known defect 2 below and should be recorded there once it is in the data.

---

## Experiment 1 — pre-registered

**Tag** `exp1-frozen` · **data** `sweep.sqlite.gz` · **probe hash** `12c9a10d970099c1…`

| | |
|---|---|
| Design | arms 1, 3, 3b × TFT, ALLC · N=1,600 · semantic framing · known horizon |
| Model | Llama-3.1-8B-Instruct, bf16, A100-80GB |
| Hypothesis | Comprehension repair moves play toward opponent-conditional optimum: defection **down** vs TFT, **up** vs ALLC |
| Result | **Rejected.** ATE_true +0.052 (tft) and +0.042 (allc), both p < 1e-4 — same sign, wrong direction |
| Manipulation check | CPR(3) − CPR(3b) = +0.244 / +0.307. Arm 3 CPR 0.24–0.31, **below the pre-registered 0.85 gate** |
| Phase 2 | Not run — gate not met |

### Defects discovered afterwards

Both were found by diagnostics on the frozen data, not by re-running until the
answer changed. `analysis/out_03.txt` is the record.

1. **Zero-padded score field.** `Your score: 012` — the model read the leading
   zero. 49.7% of all treatment score probes (6,366 of 12,800; 70.1% of failures alone) are attributable to that one
   format spec. Measured comprehension was therefore ~30% when the model could
   in fact read the block perfectly.
2. **Placebo density mismatch.** Arm 3b was two content lines padded with ~19
   blank ones — 44% content against a treatment that was 94% text. Token parity
   held; the stimuli were not comparable.

Consequence: **exp1's ATE_true of +0.042 (p < 1e-4) is a confounded estimate.**
exp2 measures, for the same model and the same contrast with a density-matched
placebo, **−0.0116 (p = 0.0040) vs ALLC and −0.0051 (p = 0.3195) vs TFT**
(`ep_exp2_llama.json`).

*This line previously read "−0.012 (p = 0.24)". **That p-value does not exist in
any run.*** The four exp2 ATE_true p-values are 0.0040 / 0.3195 (llama
allc / tft) and 0.0003 / 0.4481 (qwen allc / tft). Corrected here rather than
deleted, and the correction changes what the claim says:

- **vs TFT** the exp1 estimate collapsed to non-significance: +0.052 (p < 1e-4)
  → −0.0051 (p = 0.3195).
- **vs ALLC** it did not merely vanish — it **reversed sign and stayed
  significant**: +0.042 (p < 1e-4) → −0.0116 (p = 0.0040).

The accurate statement is therefore *"density matching reversed the sign of the
exp1 estimate"*, not *"the effect went away"*. Anyone writing this sentence must
name the opponent, because the two cells behave differently.

*This line previously read "is a false positive". That was wrong and is
corrected here rather than deleted.* A false positive is a true null rejected by
chance. This was a real, highly significant effect of a real manipulation — the
placebo differed from the treatment in **density as well as content**, so the
contrast estimated both at once. The correct phrasing is "the exp1 estimate did
not survive density matching", and the confounded-instrument reading is the more
interesting result of the two.

That comparison is reported as a **result**, not hidden as an erratum. Token
parity is necessary and not sufficient; density must match too, and this is a
demonstration of it with matched data on both sides.

---

## Experiment 2 — mechanism and second model

**Data** `exp2_llama*`, `exp2_qwen*` · same probe hash

| | |
|---|---|
| Design | arms 1, 3, 3b, 3c, 3d × TFT, ALLC · N=1,600 · Llama + Qwen |
| Plus | label swap on arms 1, 3b, both models |
| Engine | vLLM 0.27.0, `LOGPROBS_TOP_K = 20` |

### Changes from exp1

| change | reason |
|---|---|
| Numbers render naturally (`12`, not `012`) | the leading-zero defect |
| Parity enforced on token IDs, target auto-derived per tokenizer | character padding cannot control token count under BPE; `" 12"` and `"100"` differ |
| 3b and 3d templates densified (44% → 85%, 32% → 79% content) | the density defect |
| `Rounds elapsed` removed from 3b | it restated a treatment field; rounds remaining is decision-relevant, so the control was carrying usable state |
| `LOGPROBS_TOP_K` 60 → 20 | vLLM 0.27 hard-caps at 20 |

`agent_action` in exp1 and exp2 is the enum value `C`/`D`, not the long form.

### Findings

| | Llama tft | Llama allc | Qwen tft | Qwen allc |
|---|---|---|---|---|
| arm 1 | 0.322 | 0.307 | 0.010 | 0.009 |
| arm 3d | 0.238 | 0.197 | 0.037 | 0.034 |
| arm 3b | 0.110 | 0.091 | 0.068 | 0.061 |
| arm 3c | 0.122 | 0.091 | **0.306** | 0.048 |
| arm 3 | 0.104 | 0.080 | 0.065 | 0.049 |
| perturbation (3b−1) | −0.213 | −0.215 | **+0.058** | **+0.053** |
| ATE_true (3−3b) | −0.005 | **−0.012** | −0.003 | **−0.012** |

Episode-level bootstrap, 10,000 resamples. ATE_true p-values: 0.32, **0.004**,
0.45, **0.0003**.

1. **Comprehension is not the bottleneck.** Arm 3 CPR = **1.000**, verified
   `got == want` on all 12,800 probes, both models. Defection unchanged
   (0.080 → 0.080 vs exp1). The pre-registered 0.85 gate now passes.
2. **The container effect flips sign by model.** Llama −21pp, Qwen +5.8pp.
   Baselines spanning 36× collapse into a ~2× band once a block is present.
3. **Content contributes ~nothing, and what it does is wrong-signed.** ATE_true
   ≈ −0.01 in all four cells; significant vs ALLC in both models, in the
   direction *opposite* to prediction. Optimal play vs ALLC requires more
   defection; correct state produced less.
4. **False state does what true state does not.** Qwen 3c vs TFT: 0.306, with
   **0 of 1,600 episodes** avoiding defection. Same arm vs ALLC: 0.048.
5. **The container effect appears lexical.** Under label swap the block raises
   `p(emit "Cooperate")` regardless of meaning: Llama 0.678→0.890 normal,
   0.518→0.746 swapped; Qwen 0.073→0.969 swapped. Only arms 1 and 3b were
   tested — exp3 covers the full ladder.
6. Standard-error inflation from episode-level clustering is **1.4×–2.8×**.
   Turn-level intervals are roughly half the correct width; never quote them.

### Retracted

An earlier reading held that a block containing the correct answer made the
model *worse* at reporting turn-0 state (1.000 → 0.190). That was the
`.ljust()` padding on the last-move field. With natural rendering, turn-0
`opponent_last` is 1.000 in every arm. **Withdrawn.** The effect survives only
in arm 3d for Llama (`opp_last` 0.800/0.776), where there is no padding to
blame.

### Known gaps

- Mistral never ran — an `EDQUOT` disk-quota failure, not a model or auth problem
- Label swap covers arms 1 and 3b only
- `scaffold_echo` column exists but nothing wrote to it; block-reading was
  inferred indirectly from Qwen's 3c `own_score` dropping 0.400 → 0.200.
  **Superseded** — exp3 added `turns.donor_agent_score` and
  `analysis/04_donor_echo.py` measures it directly at 0.94–1.00. The inference
  was correct and badly understated. See exp3, *FINDING: the block is read*.

---

## Experiment 3 — full factorial

**Driver** `scripts/exp3_full.sh` · **90 cells** = 3 models × 5 arms × 2
opponents × 3 conditions

| condition | labels | tests |
|---|---|---|
| `sem` | Cooperate / Defect, normal | the main result; independent replication of exp2 |
| `swap` | Cooperate / Defect, inverted | is the container effect lexical, across the whole ladder |
| `abs` | X / Y | **falsification.** If the effect is about the word "Cooperate", abstract labels should shrink it. If it is unchanged under X/Y, the lexical account in exp2 finding 5 is wrong |

`swap × abs` is omitted: X and Y are already arbitrary.

### New instrumentation

| field | answers |
|---|---|
| `turns.donor_agent_score` | the number the 3c block displayed — a probe answer matching it proves the model read the block, directly rather than by inference |
| `turns.donor_degenerate` | which turns had no distinct donor, per row rather than aggregated |
| `turn_details.prompt_full` | complete prompt, first 3 episodes per cell; `prompt_preview` truncates the middle where `[STATE]` sits |
| `exp3_session.log` | full console output, committed after every group |

Database inserts now name their columns. They were positional with hand-counted
placeholders; a miscount shifts every value one position left without raising.
`tests/test_db_columns.py` gives every field a distinct value and reads all of
them back.

### Relationship to exp2

`exp3_*_sem` uses the same arms, templates and probes as exp2 but a different
`run_id`, hence **different seeds**. It is an independent replication with
non-overlapping randomness, not a re-run. Both are reported.

### N = 1,600, not 3,000

MDE at 1,600 is 5pp. The smallest effect that matters is ATE_true at ~1.2pp,
already detected at p = 0.0003 with episode-level standard errors. Doubling N
narrows intervals ~27%; the same cost buys the `swap` and `abs` conditions,
which answer questions no amount of N can.

### Parity constants (measured)

Both differ by tokenizer and belong in the methods section. Mistral has **no
single-token newline**, so `filler_candidates` fell through to a space — the
reason that setting is a list rather than a constant.

| model | parity target | filler token |
|---|---|---|
| Llama-3.1-8B-Instruct | 34 | `'\n'` (id 198) |
| Qwen2.5-7B-Instruct | 39 | `'\n'` (id 198) |
| Mistral-7B-Instruct-v0.3 | 45 | `' '` (id 29473) |

### Completed

9 groups, 90 cells, 180,000 episodes, 3.6M decisions. 21:43 → 01:14, ~3.5 h.
All estimates below are episode-level with 10,000 bootstrap resamples.
The episode/turn SE ratio spans 0.46x-4.07x across cells (0.61x-4.07x excluding `exp3_mistral_abs`).

### EXCLUDED: `exp3_mistral_abs`

Off-task rate by cell: `3|tft 1.000`, `3|allc 1.000`, `3c|allc 1.000`,
`3c|tft 0.999`, `3b|tft 0.977`, `3b|allc 0.833`, `1|tft 0.556`, `1|allc 0.442`.

Under abstract labels with a state block, Mistral does not emit X or Y at all —
it begins prose ("Given…", "To…", "Based on…"). Action tokens hold ~0% of the
mass, so those cells contain no measurement. Llama and Qwen show 0% off-task on
the same prompts, so this is a model-specific instruction-following failure, not
an instrument fault. Report it as a compliance result; exclude it from analysis.

**The group nonetheless printed `SIGN-FLIP: SUPPORTED`** with ATE_true(tft) =
-0.2266, p < 1e-4, computed entirely from noise. A worked example of why the
off-task gate exists, and of why a significant p-value from an uninspected cell
is worth nothing.

### The pre-registered condition rejects for a third time

`exp3_llama_sem`, independent seeds, N=2,000:

| | ATE_true | 95% CI | p |
|---|---|---|---|
| vs ALLC | **-0.0135** | [-0.0213, -0.0061] | 0.0005 |
| vs TFT | **-0.0207** | [-0.0299, -0.0115] | <1e-4 |

Both significant, both negative. Predicted up vs ALLC. **REJECTED**, consistent
with exp1 and exp2. Three runs, three rejections, non-overlapping randomness.

The full ladder replicates exp2 to within ~0.01 on every cell.

### FINDING: the container effect is lexical

Perturbation (arm 1 -> 3b), by framing:

| model | semantic | abstract |
|---|---|---|
| Llama vs ALLC | **-0.181** [-0.192, -0.169] | +0.007 [-0.003, +0.016] **ns** |
| Llama vs TFT | **-0.192** [-0.205, -0.179] | +0.029 [+0.019, +0.040] |
| Qwen vs ALLC | +0.039 [+0.033, +0.044] | **-0.055** [-0.064, -0.046] |
| Qwen vs TFT | +0.038 [+0.032, +0.044] | **-0.065** [-0.076, -0.053] |

Identical blocks, identical positions, identical token parity. The only
difference is whether the action labels carry meaning. In Llama the effect
disappears; in Qwen it **reverses sign**.

Baselines move too: Llama defects 0.28-0.31 under Cooperate/Defect and
0.71-0.74 under X/Y. The labels drive more behaviour than any intervention
tested.

This was a pre-specified falsification test of the lexical account suggested by
exp2's label swap. The account survived it.

### FINDING: numeric fields, not their content

`exp3_qwen_abs` (off-task clean):

| arm | vs ALLC | vs TFT | episodes never defecting |
|---|---|---|---|
| 1 | 0.081 | 0.098 | 66% / 69% |
| 3b | 0.026 | 0.033 | 90% / 89% |
| 3d | 0.044 | 0.053 | 81% / 81% |
| **3c** (false numbers) | **0.754** | **0.750** | **0%** |
| **3** (true numbers) | **0.771** | **0.770** | **0%** |

ATE_true = **+0.745** and **+0.738**, both p < 1e-4.

Under abstract labels Qwen cooperates almost totally until the block contains
numeric score fields; then it defects ~77% and **not one episode in 2,000**
avoids defecting. Three properties argue against a strategic reading:

1. off-task is clean, so it is not a readout artefact
2. false numbers do what true numbers do (0.754 vs 0.771) - not information
3. it is **opponent-invariant** to three decimals (0.771 ALLC, 0.770 TFT)

A strategic response would differ between a retaliator and a pushover.

### Mistral under semantic labels sits at the floor

99.8% of episodes never defect; every ATE is ~0.0002. Clean, but it cannot
inform any contrast. Report the floor rather than the estimates.

### Sign-flip verdicts across all 9 groups

REJECTED 4 · UNDERPOWERED 3 · SUPPORTED 2 (`mistral_abs`, excluded above;
`qwen_swap`, whose ATE_true of +0.456 vs ALLC is an order of magnitude beyond
anything in the semantic data).

Only `llama_sem` speaks to the registration. The rest are exploratory, and a
hypothesis that holds in 2 of 9 conditions and reverses in 4 is
framing-dependent - which is the finding, not support.

### FINDING: the block is read, ~always

`analysis/04_donor_echo.py`, arm 3c only, `own_score` probes, degenerate-donor
rows excluded (every turn 0 — no distinct donor can exist at score 0).

Each answer is classified against three references: the true score, the number
the block **displayed**, and true ± 1.

| group | CORRECT | DONOR_ECHO | OFF_BY_ONE | OTHER |
|---|---|---|---|---|
| llama_sem allc / tft | 0.000 / 0.054 | **1.000 / 0.946** | 0.000 | 0.000 |
| qwen_sem allc / tft | 0.000 / 0.053 | **1.000 / 0.943** | 0.001 | 0.003 |
| mistral_sem allc / tft | 0.000 / 0.000 | **1.000 / 1.000** | 0.000 | 0.000 |
| llama_abs allc / tft | 0.000 / 0.060 | **1.000 / 0.940** | 0.000 | 0.000 |
| qwen_abs allc / tft | 0.000 / 0.012 | **1.000 / 0.983** | 0.004 | 0.002 |
| llama_swap allc / tft | 0.000 / 0.019 | **1.000 / 0.981** | 0.000 | 0.000 |
| qwen_swap allc / tft | 0.000 / 0.009 | **0.994 / 0.991** | 0.000 | 0.006 |
| mistral_swap allc / tft | 0.000 / 0.000 | **1.000 / 0.999** | 0.000 | 0.001 |

64,000 probed turns. Flat across turn index, so block-reading is unconditional
rather than growing as the true state drifts from the displayed one.

`OFF_BY_ONE` was added because the first two rows inspected by hand read
`got 11 / want 12` and `got 23 / want 24`. At scale it is ~0.000, so those were
coincidence and no arithmetic-slip account is needed. `OTHER` ~0.000 rules out a
parser fault.

**The description matters.** This is not "the model echoes a false score." The
block says `Your score: 45`; the model answers 45. That is *correct reading of a
lying prompt*. CPR in arm 3c therefore measures **trust in the injected block
over the raw history**, both of which are in the same context window — not
comprehension failure. Report it that way.

**What it buys.** Reading is ~100% in arm 3c *and* arm 3 — same template, same
position, only the numbers differ. So arm 3 vs 3c is a content test with
attention held constant:

| exp3 defect rate | 3c | 3 |
|---|---|---|
| llama_sem allc | 0.097 | 0.085 |
| llama_sem tft | 0.124 | 0.100 |
| qwen_sem allc | 0.045 | 0.051 |
| qwen_sem tft | **0.306** | **0.068** |

Everywhere but Qwen-vs-TFT, false numbers and true numbers produce the same
play. The information is demonstrably ingested and still does not move
behaviour — a stronger statement than "comprehension is not the bottleneck".

Qwen-vs-TFT is the exception. A plausible mechanism is the donor's
`Opponent's last move` reading `Defect` when the truth is `Cooperate`, so a
retaliation-sensitive model punishes a betrayal that never happened. **This is a
hypothesis, not a measurement**: only `donor_agent_score` is persisted, not the
donor's last move, so the echo cannot be verified for that field. Add the column
if there is ever another run.

### Superseded by exp4

`ReadoutMode.SCRATCHPAD` was listed here as *still not run*, and as the largest
open question in the project. exp4 ran it on all three models.

---

## Experiment 4 — chain-of-thought ablation

**Driver** `scripts/exp4_cot.sh` · **code** `249c486` · **N=1,000** ·
72 cells = 3 models × 2 framings × 2 readouts × 3 arms × 2 opponents

Answers the most predictable reviewer objection to exp1–exp3: every number came
from LOGIT readout, where the action is taken from the next-token distribution
with no reasoning space. The literature this work corrects lets models reason in
text first.

Arms 3c and 3d are omitted — this is an ablation of a known contrast, not a
second factorial.

### Both readouts re-run, deliberately

exp3's environment cannot be reproduced on this driver: it ran vLLM 0.27.1 /
torch 2.13+cu130 / transformers 5.15.0 against driver 580.159.04; exp4 ran
0.11.0 / 2.8.0+cu128 / 4.57.6 against 570.195.03. Comparing exp4's SCRATCHPAD
cells to exp3's LOGIT cells would confound readout with four version changes.

Re-running LOGIT costs ~1/100th of a scratchpad cell and makes the
CoT-vs-LOGIT contrast entirely within-session. It also buys a free measurement
of the stack itself, at full N.

### The stack is not inert, and the asymmetry is informative

| ATE_true | exp3 | exp4 |
|---|---|---|
| llama_abs allc | +0.0028 | +0.0004 |
| llama_abs tft | −0.1728 | −0.1848 |
| llama_sem allc | −0.0135 | −0.0145 |
| llama_sem tft | −0.0207 | −0.0228 |
| qwen_abs allc | +0.7451 | +0.7409 |
| qwen_abs tft | +0.7375 | +0.7373 |
| qwen_sem allc | +0.0035 | +0.0090 |
| qwen_sem tft | +0.0160 | +0.0203 |

ATE_true replicates to ~0.002. **Perturbation does not** — llama_sem allc moves
−0.1806 → −0.2218, a 4pp shift on identical inputs.

Arm 3 vs 3b compares two prompts of the same shape; arm 1 vs 3b compares a
prompt with a block against one without. The block-vs-no-block contrast is the
stack-fragile one. A study measuring only ATE_true would have concluded the
stack was inert and been wrong.

### Excluded

| group | off-task | note |
|---|---|---|
| `exp4_mistral_abs_logit` | 1.000 | replicates exp3's exclusion on a second stack |
| `exp4_qwen_abs_scratchpad` | 0.201 | smoke predicted 0.163 at N=4; N=1,000 removed the noise defence |

Both are reported in full below and excluded from causal claims. Excluded means
*not used to support a conclusion*, not deleted.

`exp4_mistral_sem_logit` is **degenerate rather than excluded**: 120,000
decisions, mean P(Cooperate) 0.9176, mean P(Defect) 0.0002, max P(Defect) 0.149
— never within a factor of three of tipping. Mistral under semantic labels with
constrained readout is an unconditional cooperator. Zero variance, so it informs
no contrast. Report the description, not the estimates.

### Sign-flip verdicts

REJECTED 6 · UNDERPOWERED 3 · **SUPPORTED 0** · EXCLUDED 3.

Zero supported verdicts across every valid group, at N=1,000, with reasoning
space, on three architectures and two readouts. exp3 produced two SUPPORTED
verdicts, one of them from a 100%-off-task group.

### FINDING: defection rises sharply under chain-of-thought

| semantic, P(D given arm 3b) | LOGIT | SCRATCHPAD |
|---|---|---|
| llama | 0.102 | 0.579 |
| qwen | 0.042 | 0.597 |
| mistral | 0.000 | 0.335 |

Three architectures, three different floors, same direction. **This comparison
is confounded in exp4** — see below — and is only licensed by exp5.

### FINDING: the placebo arm earns its keep

`exp4_qwen_sem_scratchpad`, vs ALLC, episode-level bootstrap:

| contrast | estimate | 95% CI | p |
|---|---|---|---|
| perturbation (1 → 3b) | +0.1934 | [+0.1741, +0.2126] | <1e-4 |
| ATE_true (3b → 3) | −0.2135 | [−0.2279, −0.1989] | <1e-4 |
| **ATE_naive (1 → 3)** | **−0.0202** | [−0.0397, −0.0000] | 0.046 |

Two ~20-point effects of opposite sign cancelling to near zero. The standard
no-block-vs-block comparison reports "state does not matter". Without arm 3b the
decomposition is invisible.

### DEFECT: the horizon confound

The SCRATCHPAD instruction reads:

> "Before choosing, reason step by step about the current state, the opponent's
> behaviour so far, and **how many rounds remain**."

Naming the finite horizon invites backward induction, which is the textbook
argument for defecting from round one. So the readout comparison above has two
explanations and exp4 cannot separate them:

- "reasoning shatters the cooperative prior" vs "naming the horizon does"
- "reasoning kills the container effect" vs "'reason about the current state'
  does, by replacing passive priming with active attention"

**The confound is applied identically to every arm**, so every within-readout
contrast in exp4 stands — ATE_true, both perturbation figures, the Qwen
cancellation. Only the LOGIT-vs-SCRATCHPAD claim is unsafe. exp5 exists to
resolve it and is the reason the default scratchpad prompt is pinned by
`tests/test_guided_still_names_the_horizon`: exp4's databases were produced by
this exact wording and it must not be silently "fixed".

---

## Experiment 5 — minimal-CoT salience control

**Driver** `scripts/exp5_minimal_cot.sh` · **code** `249c486` · **N=1,000** ·
18 cells = 3 models × semantic × scratchpad × 3 arms × 2 opponents

Identical to exp4's semantic scratchpad groups in every respect except the
instruction, which becomes:

> "Before choosing, think step by step."

No state, no opponent, no horizon, no action label, no output format.
Selected with `--scratchpad-prompt minimal`; the default remains `guided` so
exp4 reproduces from this commit without extra flags.

LOGIT is not re-run — it never touches the scratchpad instruction. Abstract
framing is not run — llama's abstract perturbation was +0.011/+0.043, nothing to
collapse, and `qwen_abs_scratchpad` is already excluded.

Ran on the **same pod as exp4**. A fresh pod would carry the ~4pp stack offset
measured above, on top of the effect being measured.

120,000 of 120,000 turns produced non-trivial scratchpads in every group.

### RESULT: the confound did not cause the level shift

| semantic, P(D given arm 3b) | LOGIT | CoT guided | CoT minimal |
|---|---|---|---|
| llama allc | 0.102 | 0.579 | 0.639 |
| llama tft | 0.130 | 0.666 | 0.661 |
| qwen allc | 0.042 | 0.597 | 0.597 |
| qwen tft | 0.045 | 0.699 | 0.735 |
| mistral allc | 0.000 | 0.335 | 0.406 |
| mistral tft | 0.000 | 0.295 | 0.377 |

Removing the horizon clause leaves defection unchanged or **higher**, in all six
cells. Averaged over arms: llama 0.603 → 0.694, qwen 0.529 → 0.689,
mistral 0.369 → 0.407.

The exp4 readout claim survives its own control.

**The horizon effect runs backwards.** Naming the rounds made llama and qwen
defect *less* than a bare instruction (arm 1: 0.731 → 0.557 llama,
0.650 → 0.404 qwen). That is the opposite of the backward-induction prediction,
and evidence against a textbook-retrieval account. A plausible reading is that
the guided prompt also points at *the opponent's behaviour so far*, surfacing
their cooperation. Not tested.

### RESULT: the container collapse was partly the instruction

llama, perturbation (arm 1 → 3b), episode-level bootstrap:

| condition | allc | tft |
|---|---|---|
| LOGIT | −0.2218 [−0.2387, −0.2052] | −0.2069 [−0.2253, −0.1880] |
| CoT minimal | −0.0920 [−0.1054, −0.0788] | −0.0747 [−0.0869, −0.0620] |
| CoT guided | +0.0215 [+0.0039, +0.0390] | +0.0161 [−0.0012, +0.0334] |

No CI overlap between any pair. Reasoning attenuates the container effect by
~60% but does **not** abolish it; directing attention at the state removes the
remainder. Two separable mechanisms, both significant.

"CoT kills the container effect" — as exp4's guided cells implied — is too
strong and is withdrawn.

### RETRACTED: Qwen's state-comprehension effect

exp4's `qwen_sem_scratchpad` ATE_true of −0.2135 / −0.1015 was read during the
run as a conditional revival of the state hypothesis. Under a neutral
instruction it is **+0.0437 / +0.0171**. It was produced by directing attention
at the block, not by the block's content. Withdrawn.

### FINDING: under neutral reasoning the container effect generalises

| perturbation, allc | LOGIT | CoT minimal |
|---|---|---|
| llama | −0.222 | −0.092 *** |
| qwen | **+0.033** | **−0.053** *** |
| mistral | 0.000 | −0.021 *** |

Qwen's sign reverses. Five of six point estimates negative, four significant.
Under constrained readout the container effect was Llama-only; under neutral
reasoning it appears in all three, same direction. New in exp5; not implied by
exp4's guided cells.

### Sign-flip verdicts

| model | ATE_true allc | ATE_true tft | verdict |
|---|---|---|---|
| llama | +0.0738 [+.060, +.088] | +0.0237 [+.011, +.037] | REJECTED |
| mistral | +0.0294 [+.020, +.039] | +0.0360 [+.027, +.045] | REJECTED |
| qwen | +0.0437 [+.023, +.065] | +0.0171 [−.001, +.035] | UNDERPOWERED |

Every cell positive. Every ALLC cell matches the prediction; every TFT cell
violates it. Under neutral reasoning, true state makes all three models defect
**more** than a matched placebo, and does so **regardless of opponent** — they
use the state and do not condition that use on who they are playing.

Qwen's TFT cell is p = 0.058 and must be reported as inconclusive, not rejected.

### Scope limit — salience, not availability

The rules section already states the game lasts exactly 20 rounds, so the
horizon was never hidden; the exp4 instruction only made it **salient**. exp5
answers *"does drawing attention to the horizon cause defection?"* It does not
answer *"does knowing the horizon cause defection?"*

The stronger test is `HorizonMode.STOCHASTIC`, where no last round exists and
backward induction cannot apply. The machinery is in the codebase; it was not
run. Describe exp5 as a salience control in any writeup.

---

## Experiment 6 — field-level falsification

**Question.** Arm 3c replaces the whole `[STATE]` block with another episode's,
so the 24pp qwen-vs-TFT swing in exp2/exp3 could not be attributed to a field.
exp6 splits 3c into its two falsifiable components.

| arm | manipulation |
|---|---|
| `3s` | **only** `Your score`, shifted by ±15 |
| `3m` | **only** `Opponent's last move`, flipped |

Everything else — template, position, token count, padding — is identical to
arm 3. `tests/test_field_falsification.py` compares rendered blocks line by line
and asserts exactly one line differs.

**Design.** 3 models × semantic × {1, 3b, 3, 3s, 3m, 3c} × {tft, allc} × LOGIT ×
N=1000, plus {1, 3b, 3, 3m} × SCRATCHPAD (minimal prompt, 128 tokens). 60 cells,
1.28M turns. H100, vLLM 0.11.0, ~1h55m.

### Why not a fourth opponent

Three opponent designs were rejected first: a threshold-defector (exhaustive
search gives "cooperate once then defect forever" as optimal at every X) and a
lead-guard (75,582 sequences tie for the optimum at X=45, and a policy reading
only the round index attains it). Both die to one theorem — for a
**deterministic** opponent from a fixed start the open-loop optimum equals the
closed-loop optimum, so state-tracking is never payoff-necessary. A fourth
scripted opponent fails identically.

Falsifying a field sidesteps the critique those designs existed to answer. The
standing objection is that cumulative score is a *sunk* variable against TFT and
ALLC. That is true of the score and false of the last move, which is the entire
input to optimal play against TFT.

### Parity

`block_tokens` unchanged at 34 / 39 / 45. `_derive_block_tokens()` still iterates
only `treatment_text`, `nondiagnostic_text`, `syntactic_text`, so the new arms
fit inside a target the old arms set and **exp1–exp5 reproduce byte-identically
from HEAD**. Pinned by `tests/test_exp1_to_exp5_unchanged.py`.

### Manipulation integrity

`turns.displayed_opponent_last` records what the block asserted. Verified against
the previous turn's `opponent_action` by self-join, so the check cannot share a
bug with the writer.

| arm | opponent | falsified | rows | rate |
|---|---|---|---|---|
| `3m` | both, all models | 19,000 | 19,000 | **1.0000** |
| `3s` | both, all models | 0 | 19,000 | **0.0000** |
| `3c` | allc, all models | 0 | 19,000 | **0.0000** |
| `3c` | tft, llama | 5,678 | 19,000 | 0.2988 |
| `3c` | tft, mistral | 2,644 | 19,000 | 0.1392 |
| `3c` | tft, qwen | 7,153 | 19,000 | 0.3765 |

### FINDING: the last-move field dominates; the score does not

Episode-level bootstrap, 10,000 resamples, seed 20260811, **turn 0 excluded from
both arms** — at turn 0 there is no last move, so arm 3m is byte-identical to
arm 3 and those rows carry no manipulation.

| group | opp | `3−3s` (score) | `3−3m` (move) | ratio |
|---|---|---|---|---|
| qwen | tft | +0.0138 [+.0059, +.0217] | **−0.4049** [−.4160, −.3938] | 29× |
| qwen | allc | +0.0048 [−.0017, +.0113] ns | **−0.2839** [−.2922, −.2758] | 59× |
| llama | allc | −0.0227 [−.0327, −.0127] | **−0.0931** [−.1043, −.0820] | 4.1× |
| llama | tft | −0.0317 [−.0447, −.0186] | **−0.0891** [−.1027, −.0754] | 2.8× |
| mistral | tft | +0.0001 [−.0002, +.0003] ns | **−0.0141** [−.0163, −.0120] | — |
| mistral | allc | −0.0004 [−.0007, −.0001] | **−0.0115** [−.0132, −.0099] | 29× |

The move effect is larger in **6 of 6 cells**, by 2.8× to 59×.

**The score effect is not a clean null.** Four of six score contrasts exclude
zero. They are small (max |0.032|, in llama) and in qwen the sign *reverses* —
the false score slightly **reduced** defection. The licensed claim is "the
last-move field dominates," never "the score does nothing."

### FINDING: the false score is read perfectly and acted on barely

Arm 3s displays a score 15 points wrong. The own-score probe answer was compared
against both the displayed value and the true cumulative payoff:

| group | probes | matched **displayed** lie | matched true |
|---|---|---|---|
| llama_sem_logit | 10,000 | **10,000 (100.0%)** | 0 |
| qwen_sem_logit | 10,000 | **10,000 (100.0%)** | 0 |
| mistral_sem_logit | 10,000 | **10,000 (100.0%)** | 0 |

Unanimous across 30,000 probes. Read alongside `3−3s` above: the model ingests a
false state, reproduces it with perfect fidelity, and changes its behaviour by
0.5–3.2 percentage points. This is the direct measurement that C3 and C1
previously had to infer, and it is the strongest available answer to "perhaps the
model never read the block."

### FINDING: arm 3c vs ALLC is a zero-dose positive control

Against ALLC every donor episode also shows `Cooperate`, so arm 3c **cannot**
falsify the last move — 0 of 19,000 rows in all three models. Its effect there:

| model | 3c falsification rate vs allc | `3−3c` vs allc |
|---|---|---|
| llama | 0.0000 | −0.0140 |
| mistral | 0.0000 | +0.0000 ns |
| qwen | 0.0000 | +0.0128 |

Zero dose, ~zero effect, in a cell that was already in the corpus. Against TFT,
where the field *can* be corrupted, qwen's `3−3c` is −0.2016 — the exp2/exp3
effect replicating in a fresh run.

This retires the reviewer's sunk-variable critique. The critique was correct
about the score and irrelevant, because the score was never the operative field.

### LIMITATION: the effect does not survive chain-of-thought in Llama

| model | opp | `3−3m` LOGIT | `3−3m` SCRATCHPAD |
|---|---|---|---|
| qwen | allc | −0.2839 | −0.1447 [−.1612, −.1281] |
| qwen | tft | −0.4049 | −0.0520 [−.0652, −.0383] |
| mistral | allc | −0.0115 | −0.0328 [−.0416, −.0237] |
| mistral | tft | −0.0141 | −0.0344 [−.0431, −.0257] |
| **llama** | **allc** | **−0.0931** | **+0.0126** [+.0007, +.0244] |
| **llama** | **tft** | **−0.0891** | **−0.0155** [−.0267, −.0040] |

Under reasoning qwen's effect survives but shrinks 50–87%, mistral's **grows**
(it was floor-limited under LOGIT at 0.0001 defection and reaches 0.41–0.44 under
CoT), and **llama's collapses to approximately zero and reverses sign vs ALLC**.

The literature uses CoT. Any claim that the last-move effect is readout-invariant
is false and must not be made. State it as: present under LOGIT in all three
models, present under CoT in two of three.

### OPEN: the per-falsified-row overshoot

Rescaling each arm's effect by its own falsification rate should make 3c and 3m
agree if the move is the only operative field:

| model | `3m` per lied row | `3c` per lied row (tft) |
|---|---|---|
| qwen | −0.4049 | **−0.5356** |
| llama | −0.0891 | **−0.1127** |
| mistral | −0.0141 | −0.0019 |

3c overshoots by 26% (llama) and 32% (qwen). Qwen's score effect runs the *wrong*
direction, so the score cannot account for it. Remaining candidates: the two
fields interacting when both are wrong, or 3c's falsified rows landing
disproportionately on high-leverage turns. Unresolved; report as such.

### Defect resolved

Known defect 4 — "donor's last move is not persisted" — is closed by
`turns.displayed_opponent_last`, added in exp6 for every arm in
`Arm.falsifies_field`. The exp3 mechanism is no longer a hypothesis.

### Reproduce

```bash
gunzip -kf exp6_*.sqlite.gz
python analysis/13_exp6_fields.py | tee EXP6_FIELDS.txt
```

---

## Experiment 7 — the two confounds on exp6's result

**Question.** exp6 showed that falsifying only `Opponent's last move` moves
defection far more than falsifying only the score. Three accounts predict that
identically and exp6 cannot separate them:

| account | prediction |
|---|---|
| **state use** | the model conditions on the field |
| **conflict** | `[HISTORY]` renders every round one section below, so arms 3c/3s/3m were never false-state manipulations — they were *contradiction* manipulations, and the model may simply be picking a side |
| **lexical** | arm 3m injects the literal token `Defect`, and D2 shows labels moving Llama's baseline from 0.28 to 0.74 |

**Design.** 3 models × {1, 3b, 3, 3s, 3m} × {tft, allc} × N=1000 × LOGIT in four
conditions: `abs` (X/Y labels), `swap` (inverted mapping), `nohist` (history
removed), `absnohist`. 9 groups, 90 cells, 1.8M turns, ~1h48m on an H100.
Frozen in `PREREGISTRATION_EXP7.md` before data.

`include_history` is a keyword argument on `assemble`, **not** a `ScaffoldConfig`
field, for the same reason the scratchpad variant is not one — it would change
`config_fingerprint` on every historical row. Verified against an independent
reimplementation of the pre-flag assembler: **56 prompts, 0 mismatches**, so
exp1–exp6 render byte-identically. Parity targets unchanged at 34/39/45.

### FINDING: the effect survives with the block as the only source of state

With `[HISTORY]` removed, arm 1 has no state source at all. Deprivation is total
and repair is complete:

| group | arm-1 own score | arm-1 last move | arm-1 rounds | arm-3 CPR |
|---|---|---|---|---|
| llama nohist | 0.000 | 0.007 | 0.000 | 1.000 |
| mistral nohist | 0.000 | 0.000 | 0.000 | 1.000 |
| qwen nohist | 0.000 | 0.000 | 0.000 | 1.000 |

And `3−3m` still fires, same sign as semantic, surviving Holm across a 72-member
family:

| group | opp | semantic (exp6) | **nohist (exp7)** |
|---|---|---|---|
| llama | allc | −0.088 | **−0.0564** [−.0635, −.0495] |
| llama | tft | −0.085 | **−0.0337** [−.0397, −.0274] |
| mistral | allc | −0.011 | **−0.0106** [−.0138, −.0073] |
| mistral | tft | −0.013 | **−0.0054** [−.0082, −.0025] |

**The conflict account is refuted.** The model uses the block as state when the
block is the only state there is. llama retains 40–64% of its semantic effect,
mistral essentially all of it. qwen is uninformative — `P(D|3) = 0.0007` with
history removed, a floor.

### DEFECT: the placebo leaks under no-history

In **every** no-history condition arm 3b responds to its `Round parity` line and
arm 1 does not (detrended, differencing each odd turn against its neighbours):

| group | 3b detrended | arm 1 |
|---|---|---|
| llama nohist | −0.0488 / −0.0526 | flat |
| llama absnohist | +0.0579 / +0.0568 | flat |
| mistral nohist | −0.0077 / −0.0063 | flat |
| qwen absnohist | −0.0879 / −0.0926 | flat |

With the history gone, parity is the only turn-index signal left and the model
uses it. Magnitudes reach 0.093 — **larger than llama's headline effect.**

Scope precisely: this contaminates `ATE_true` and `perturbation`, which involve
arm 3b. It does **not** touch `content_move`, `content_score` or
`content_donor`, which compare arm 3 against arms carrying the real state block.
The headline is clean; the pre-registered contrast needs the caveat.

### The abstract condition is a different regime, not a compressed one

| model | P(D\|3) semantic | P(D\|3) abstract |
|---|---|---|
| llama | 0.086 / 0.104 | 0.714 / 0.593 |
| qwen | 0.051 / 0.067 | 0.770 / 0.769 |

`ATE_true` in `qwen_abs` is **+0.75** against +0.017 in `qwen_sem`. That is not
the same estimand measured under different labels; C4 already established that
qwen under X/Y is driven by the presence of numeric fields and is
opponent-invariant. The lexical account therefore remains **open** — exp7 did
not settle it. Report on the odds scale, per `PREREGISTRATION_EXP7.md`.

exp7 replicates exp3's `qwen_abs` arm 3 at 0.770 / 0.769 against 0.771 / 0.770 —
independent run, different code revision, four-decimal agreement.

### FINDING: the pre-registered criteria are met in one condition, and replicate

After the swap rescore below, `exp3_qwen_swap` and `exp7_qwen_swap` both satisfy
the registered sign-flip with a **passing** manipulation check:

| run | ATE_true allc | ATE_true tft | verdict |
|---|---|---|---|
| exp3_qwen_swap | **+0.4559** | −0.0659 | criteria met |
| exp7_qwen_swap | **+0.4636** | −0.0792 | criteria met |

**But the mechanism is inverted from the hypothesis, and that replicates too.**
Opponent spread — how far each arm's defection moves between TFT and ALLC:

| arm | exp3 | exp7 |
|---|---|---|
| 1 (no block) | 0.073 | 0.078 |
| **3b (contentless block)** | **0.693** | **0.710** |
| 3 (true state block) | 0.171 | 0.167 |

A block containing **no real state** makes Qwen roughly ten times more
opponent-sensitive than no block at all, and the **true** state block damps that
back to 0.17. The criteria are satisfied because arm 3b swings, not because arm
3 does. The registered hypothesis was that true state *enables* conditioning;
what replicates is that an empty block produces it and true state suppresses it.

llama shows none of this: `ATE_true` is +0.067/+0.033 (exp3) and +0.055/+0.025
(exp7) — same sign both opponents, no flip. Qwen-specific, twice-replicated.

**Reporting rule.** Any sentence about this must say *"the criteria are met, in
the label-swap condition, driven by the placebo arm."* Writing "the hypothesis
was supported" without that clause is an overclaim.

### DEFECT: the swap probe scorer, and a second defect in the rescore itself

The swap condition inverts the action words. The probe scorer compared answers
against **unswapped** truth, so every non-turn-0 `opponent_last` answer was
marked wrong and CPR collapsed to exactly the turn-0-only rate, 0.200. The
contingency tables are clean inversions (66.9%–79.1% exactly inverted), so
rescoring is justified. Behavioural data was never affected — `_resolve_action_tokens`
(`cdx/backends_vllm.py:154`) inverts the surface forms correctly, so `P(defect)`
is always the true action.

Rescored, **all five swap groups pass arm 3 at 1.000**:

| group | arm 3 as run | rescored |
|---|---|---|
| exp3_llama_swap | 0.200 | **1.000 PASS** |
| exp3_mistral_swap | 0.200 | **1.000 PASS** |
| exp3_qwen_swap | 0.200 | **1.000 PASS** |
| exp7_llama_swap | 0.200 | **1.000 PASS** |
| exp7_qwen_swap | 0.200 | **1.000 PASS** |

`exp3_mistral_swap` first rescored to 0.800 and appeared to still fail. That was
a **second defect, in `analysis/10_rescore_swap.py` itself**: it compared
`str(got).strip() == str(want).strip()` instead of importing
`cdx.probe.normalise`, so Mistral's verbose turn-0 answer
`"None (since no round has been played yet)"` was scored wrong. The production
scorer always handled it — the stored `cpr_score` for that turn is 1. The script
now imports the production normaliser, and the test is that its "CPR as run"
column reproduces the stored `cpr_score` exactly: **0 mismatches across all five
groups.** Mistral is recovered by fixing an analysis script, not by anything
about the model.

### Reproduce

```bash
gunzip -kf exp7_*.sqlite.gz exp3_*swap*.sqlite.gz
python analysis/13_exp6_fields.py --glob 'exp7_*.sqlite' --out EXP7_FIELDS.json
python analysis/14_reviewer_responses.py --glob 'exp7_*.sqlite' --out EXP7_REVIEWER.json
python analysis/10_rescore_swap.py --glob 'exp3_*swap*.sqlite' --out SWAP_RESCORE.md
python analysis/10_rescore_swap.py --glob 'exp7_*swap*.sqlite' --out EXP7_SWAP_RESCORE.md
```

---

## Experiment 8 — does the instrument's answer survive a different prompt?

**Pre-registered** (`PREREGISTRATION_EXP8.md`, frozen before data).
Driver `scripts/exp8_templates.sh`, `MODE=screen`. Analysis
`analysis/15_exp8_stability.py` (probability scale) and
`analysis/16_exp8_logodds.py` (**the registered scale — see deviation 6**).
Ran 2026-08-15 on one H100 80GB,
vLLM 0.11.0 / torch 2.8.0+cu128, commits `f086476` → `de5b4af`. ≈$9.45.

### Why

Two referees made the same objection: *a method that has only ever been run on
one prompt template is not an instrument.* exp1–exp7 — ~300,000 episodes — used
one `[STATE]` template, one field order, one insertion position. **D2** already
showed that a purely lexical change reverses the sign of the container effect in
Qwen. So **C5** had been measured at exactly one point in a three-dimensional
space in which this project had already documented a sign reversal.

### Design

Half fraction `2^(3−1)` on all three models, plus the foldover on qwen.

| factor | − | + |
|---|---|---|
| **T** template | `original` | `reworded` (no shared content word except `Your`, `Rounds`) |
| **O** order | canonical | permuted (score first→last, last move third→first) |
| **P** position | `insertion_index = 1` | `= 2` (after `[HISTORY]`) |

Half A is `I = −TOP` and contains the anchor, which is exp6's condition exactly
— **re-run in session**, not inherited, so no contrast in the design is
between-stack (**B5**: 4pp of movement across images). Half B is the foldover,
run on qwen only, giving qwen the **full 2³**.

16 groups × 10 cells. Arms `1 3b 3 3s 3m`, opponents `tft allc`, N = 1000,
semantic labels, `[HISTORY]` present, LOGIT. 200,000 turns and 76,000 recorded
falsifications per group.

Primary estimand `A = P(D|3m) − P(D|3s)`: arms 3s and 3m are byte-identical but
for one line, at the same width and position, so every between-condition
nuisance is common to both and cancels. Arm 3 drops out algebraically.

### The anchor reproduced exp6

qwen `ATE_true` +0.028 / +0.022 against exp6's +0.0339 / +0.0174; each point
estimate inside the other's interval. The stack had not drifted, so everything
below is a statement about prompts.

### Result

**C13.** Registered verdict **INCONCLUSIVE (TFT) / PARTIAL (ALLC)**. On the
registered log-odds scale the asymmetry is not portable, but the failure is
narrower and more specific than the probability scale suggested: qwen **2 of 7**
conditions outside `[0.50, 2.00]` on TFT — `origpermp1` (0.399) and
`rewordpermp1` (0.325), **both permuted field order at insertion index 1** —
while llama's `origpermp2` sits at **S_lo = 5.085**, outside in the opposite
direction, at permuted order and index **2**. Mistral changes qualitatively:
`A_lo` = +0.166 **[−0.200, +0.549]**, CI including zero, at `rewordp2`, against
+5.032 at `rewordpermp1`. The **order × position interaction** (+0.1070) is the
largest term in qwen's full 2³, and it reappears on the log-odds scale — every
`O+` condition at position 1 leaves the band, every `O+` at position 2 stays in.

### Deviations from `PREREGISTRATION_EXP8.md` — all five, in full

1. **§7 gate 6 is unsatisfiable as registered.** It requires
   `COUNT(DISTINCT prompt_tokens)` per (opponent, turn) over block arms to be 1.
   Once arms behave differently their histories differ in text, and wherever the
   tokeniser gives `Cooperate` and `Defect` different lengths, in token count.
   Run against the committed exp6 databases the registered form fails **36 of 40
   groups on mistral** — it rejects the data **C5** is built on — while passing
   0 of 40 on qwen, whose action labels happen to tokenise to equal length.
   **Restricted to turn 0**, where `[HISTORY]` is empty and any difference is a
   real parity violation. Still catches an injected +7-token break.
2. **A cross-template density check was added after freezing**, then
   **downgraded from abort to warning** when it blocked a registered factor. It
   is not one of §7's eleven gates and has no standing to kill a registered
   condition. Both states are in the git history.
3. **The T factor carries a measured density confound.** Under Llama-3.1 the
   `reworded` treatment block is 81–84% real content against `original`'s
   91–94% — a **10.3%** gap, because the reworded syntactic body is the longest
   of its three and drags the per-template parity target from 34 to 37 tokens.
   Reported, not tuned away: resizing placebo bodies to hit density targets
   across three BPE vocabularies *after* seeing the failure would be instrument
   p-hacking. **O and P are unaffected** — `original_permuted` matches
   `original` to 0.0% on every block type, and the headline instability comes
   from those factors.
4. **§4 names `analysis/02_episode_level.py` as the estimator. It is not one.**
   analysis/02 computes `ATE_true`, `perturbation` and `ATE_naive`; it does not
   compute `A`, has no turn-0 exclusion, and opens one `--db` so it cannot
   difference across groups. `analysis/15_exp8_stability.py` is the correction.
   §4 also states `A = E_move − E_score`; that identity is false —
   `E_move − E_score = P(3s) − P(3m) = −A`. The code was always correct.
5. **F1 could never fire.** §5.7 gates the magnitude rule on at least one
   non-anchor **CLEAN** condition, and §5.4 marks a cell COMPRESSED when
   `min P(D)` over {3, 3s, 3m} ≤ 0.15. Arms 3 and 3s run 0.003–0.085 across all
   32 cells, so `min P(D)` is always ≤ 0.15 and **zero CLEAN cells exist at any
   N**. The primary falsification rule was inoperable from the moment it was
   written. F2 needed a sign reversal and all 31 non-floored cells are positive.
   Hence PARTIAL — while the measurement shows the asymmetry collapsing to a
   fifth of its anchor in every alternative configuration.

6. **The registered reporting scale was not computed until after the run.**
   §4 states that `A_lo = logit P(D|3m) − logit P(D|3s)` is *"reported for every
   cell, and the **only** form quoted for COMPRESSED cells (§5.4)."* All 28
   non-floored cells are COMPRESSED, so log-odds is the only quotable form —
   and `analysis/15` computes only the probability scale.
   `analysis/16_exp8_logodds.py` supplies it, imports `analysis/15` so there is
   one loader and one turn-0 rule, and self-checks that the probability-scale
   numbers reproduce bit-for-bit.
   **It materially narrows the claim.** Seven of seven qwen/TFT conditions
   outside the band on probability becomes **two of seven** on log-odds; three
   are not statistically distinguishable from the anchor at all. Ten cells
   disagree between the scales, **all qwen and all in one direction** (outside
   on probability, inside on log-odds), because `P(D|3s)` falls to 0.009 against
   the anchor's 0.062 — a probability difference compresses against the floor
   where an odds ratio does not. That is the compression §5.4 anticipated and
   the reason §4 registered log-odds in the first place.
   The registered *verdict* is unchanged (F2 is provably scale-invariant; F1
   cannot fire on either scale). Zero treatment: no cell has an observed rate of
   zero — mistral's `0.000` is display rounding of 3/19000 — but 2 of 32 cells
   produce zero-rate bootstrap draws; Haldane–Anscombe h=0.5 applied uniformly
   to turn counts, with two sensitivity analyses that change no conclusion.

Items 5 and 6 are the transferable ones. Item 5: **a magnitude rule gated on a cell class that
rare-event base rates make unreachable cannot fire.** Anyone pre-registering
over low-probability outcomes should check that their vote-eligibility class is
attainable before freezing.

---

## Reviewer-response analyses — `analysis/14`

Not an experiment. Eight analyses run on the **committed** exp2–exp6 databases,
written to answer the objections three simulated reviewers raised independently,
before submission rather than after. No new data.

```bash
gunzip -kf exp6_*.sqlite.gz
python analysis/14_reviewer_responses.py | tee REVIEWER_RESPONSES.txt
```

### FINDING: the pre-registered rejection survives its best defence

The objection: along cooperative trajectories TFT and ALLC emit an identical
sequence, so until the agent defects **no arm contains a bit that distinguishes
them** and the sign-flip is untestable by construction.

Restricting to turns after the agent's first defection rescues the hypothesis in
**no group**. Report the stratified test — it forecloses the objection.

Caveat printed on every run: "has defected at least once" is an outcome, so
conditioning on it breaks randomisation. The arm-3/arm-3b selection gap is the
size of the bias and must be quoted beside the estimate. In
`exp6_qwen_sem_logit` that gap is **+0.371**, which is why the −0.4533 figure in
that cell is selection and not effect.

### FINDING: models track the field they use and fail the field they do not

Arm-1 CPR decomposed into its three sub-probes, turn 0 excluded. See `CLAIMS.md`
C9 for the full table. Summary: the opponent's last move is recalled at
**0.91–1.00 with no block at all**, in every model and both readouts, while the
cumulative score sits at 0.00–0.25.

The block therefore repairs the field the model can neither compute nor use, and
adds nothing to the field it already tracks. That is the mechanism behind the
project's central null, and it replaces the earlier reading of arm-1 CPR = 0.20,
which was a conjunction floored by an arithmetic sub-probe.

### FINDING: the 3c overshoot is explained, and the ratio behind it is invalid

Three candidates were tested. In qwen the two fields **interact** when both are
wrong (move+score 0.4636 vs move-only 0.3442, interaction +0.1211
[+0.0904, +0.1519], p = 0.0001). In llama there is no interaction (p = 0.48) and
turn composition covers it.

Separately, the per-falsified-row rescaling is **not** a per-row causal effect:
falsification in arm 3c is post-treatment selected, with the recipient's own
prior-defection rate differing by +0.088 (llama) and +0.241 (qwen) between
falsified and unfalsified rows. Report the marginal 3c effect and the
falsification rate separately; drop the ratio.

### DEFECT: the placebo is not perfectly inert

Arm 3b's block contains `Round parity: even/odd`. A raw even-minus-odd contrast
is not a parity test — over turns 0..19 the odd turns average one index later, so
any monotone trajectory manufactures a gap of roughly one turn's slope. Arm 1 and
arm 3b show the *same* raw gap (−0.031 each in `exp6_llama_sem_logit`) and mean
opposite things, because arm 1's slope is 0.0178/turn and 3b's is 0.0055/turn.

Differencing each odd turn against its two neighbours cancels a locally linear
trend of any slope. Detrended, the parity effect is −0.026 (llama logit), −0.069
(mistral CoT) and +0.016 (qwen CoT), CIs excluding zero, while arm 1 is flat
where it can be checked.

**Consequence:** `ATE_true` compares the treatment against a control that is
itself partly active, so it is **conservative**. Report the detrended coefficient
in Methods and do not describe arm 3b as "non-diagnostic" without the
qualification. This does not overturn the null — a leaky placebo biases toward
finding nothing, which is the direction already reported.

### DEFECT: one contrast rests on near-ties

`exp6_qwen_sem_scratchpad` has **31.2%** of decisions in its largest-effect cells
below `action_mass_total` 0.25 — neither off-task nor solid. Quote the fragile
share beside `3−3m` = −0.1447, or restrict to solid decisions first. Every other
group is under 3.3%.

### Multiplicity and heterogeneity

- Across the 48-member exp6 contrast family: 41 raw, **34 survive Holm**, 40
  survive BH. Across all experiments (178 contrasts) the proportions are the
  same, 124 and 143. The correction family must be **pre-specified**.
- Model heterogeneity (`CLAIMS.md` G8, previously untested): joint p = 0.0001 in
  all four strata, spread up to 0.084. **Three case studies, not a pooled
  effect.**

---

## Known defects, open

Carried forward rather than fixed, with the reason.

1. **`episodes.prompt_hash` is the probe-suite hash, not a prompt digest.**
   `scripts/gpu_run.py` sets `prompt_hash=PROBE_SUITE_HASH[:32]`, a constant
   across every arm, framing, readout and model. Present in exp2–exp5.
   `cdx/runner.py` computes a real digest; the driver never wired it up. Affects
   no measurement — nothing reads it. Not fixed before exp5 because editing the
   driver between exp4 and its control would have added a second difference to a
   comparison designed to have one. Provenance is carried instead by
   `turn_details.prompt_full`, which stores the complete decoded prompt for the
   first 3 episodes of every cell.
2. **`config_fingerprint` does not include the scratchpad variant.** It hashes
   `ExperimentConfig`, which does not carry it. exp4 and exp5 are distinguished
   by `run_id`, by `run_meta.config_json`, and by `prompt_full`. Adding the field
   would change the fingerprint of every historical run.
3. **Probe prompts inherit the instruction suffix.** `_probe_turn` builds
   `list(prompt) + probe_suffix`, so CPR is measured on a prompt that still
   carries the reasoning instruction. CPR is therefore **not comparable between
   exp4 and exp5**. Within a run all arms share the instruction, so every CPR
   contrast is unaffected.
4. ~~**Donor's last move is not persisted.** Only `donor_agent_score`. The
   Qwen-vs-TFT mechanism in exp3 is consequently a hypothesis.~~
   **RESOLVED in exp6** by `turns.displayed_opponent_last`, written for every arm
   in `Arm.falsifies_field`. exp2–exp5 databases still lack the column, so the
   mechanism is *measured* in exp6 and remains an inference in exp3. Kept visible
   because the exp3 numbers are still quoted.
5. **`enable_prefix_caching=True`** in exp4 and exp5. Episodes are independently
   seeded with no carried game state, so the design is i.i.d.; under continuous
   batching numerics can depend on batch composition. State the engine
   configuration rather than writing "i.i.d." unqualified.
6. **SE inflation below 1.0×** appears in valid groups
   (`exp5_mistral_sem_minimal` 0.91–1.08, `exp4_qwen_abs_logit` from 0.86), not
   only in excluded ones. It indicates stable per-episode rates, **not** a broken
   readout. Use off-task as the gate; do not add this one.

---

## Reporting rules

1. **exp1 is the only pre-registered test.** Its rejection stands. Nothing run
   afterwards re-tests the sign-flip hypothesis or is presented as if it had
   been registered.
2. exp2 and exp3 are labelled exploratory throughout.
3. Instrument-gate failures are reported, including the distinct-trajectory
   failures. Those reflect **low entropy, not lost sample size** — episodes are
   independently seeded, so a repeated trajectory is a repeated draw. The gate
   exists to catch deterministic collapse, which would look identical in that
   column but would show zero variance in the logit masses.
4. All intervals come from `analysis/02_episode_level.py`. Turn-level intervals
   have an episode/turn SE ratio of **0.46×–4.07× across exp2–exp5 (0.61×–4.07× excluding the excluded `exp3_mistral_abs` group; the previously quoted 0.62×–3.75× is exp4's range alone). Turn-level intervals are wrong in BOTH directions — too narrow in most cells, too wide in 50 of 208**; never quote them.
5. Every defect above is reported. The exp1 → exp2 placebo comparison is the
   strongest methodological result in the project and depends on both runs
   being on the record.
6. **exp4's cross-readout claim is licensed only by exp5.** Any statement that
   reasoning raises defection must cite the minimal-CoT control alongside it.
   The horizon confound is reported in exp4's own section, before the result it
   qualifies — not relegated to a later note.
7. A claim withdrawn by a later run is marked **RETRACTED** in place and the
   original left standing. exp5 retracts Qwen's state-comprehension effect and
   the "CoT kills the container effect" reading; both remain visible in exp4.
