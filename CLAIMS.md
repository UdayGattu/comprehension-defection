# Claim register

Every sentence intended for the paper, the exact evidence behind it, and its
status. Nothing enters the manuscript that is not in this file with a status of
CONFIRMATORY or SUPPORTED.

`EXPERIMENTS.md` records what was run. This records what may be said about it.

Numbers are episode-level bootstrap estimates from `analysis/02_episode_level.py`
unless marked otherwise. Turn-level intervals are never quoted.

| status | meaning |
|---|---|
| **CONFIRMATORY** | Tested against `PREREGISTRATION.md`. Only exp1 qualifies. |
| **SUPPORTED** | Exploratory, holds across ≥2 independent runs or ≥2 models. |
| **SUPPORTED, NARROW** | Holds in one model or one condition. State the scope in the sentence. |
| **REJECTED** | The data contradict it. |
| **RETRACTED** | Claimed earlier in this project, withdrawn by later evidence. Kept visible. |
| **NOT TESTED** | Plausible, unmeasured. Belongs in Limitations or Future Work, never Results. |
| **OVERCLAIM** | A sentence someone has already written or said that the data do not license. Listed so it does not reappear. |

---

## Vocabulary — fix this before drafting

The word **"comprehension"** must not do unearned work. What is measured:

| term | operational definition | measured by |
|---|---|---|
| **availability** | the state is present in the context window | by construction |
| **readability** | the model can reproduce a state field when asked | CPR, arm 3 = 1.000 |
| **block-reading** | the model reports the *injected block's* value in preference to the raw history | DONOR_ECHO, 0.94–1.00 |
| **behavioural use** | play changes when the block's content changes | ATE_true (arm 3 vs 3b) |
| **strategic conditioning** | that change differs by opponent type | the pre-registered sign-flip |

The study measures each rung separately. That separation is the contribution.
**"The model comprehends the state but does not act on it"** conflates rungs 2–5
and is an OVERCLAIM. Use the ladder.

---

## A. The pre-registered hypothesis

### A1. Comprehension repair does not produce opponent-conditional play
**Status: CONFIRMATORY (exp1), then SUPPORTED across every subsequent run.**

Prediction: defection **down** vs TFT, **up** vs ALLC.

| run | group | ATE_true allc | ATE_true tft | verdict |
|---|---|---|---|---|
| exp1 | llama | +0.042 *** | +0.052 *** | rejected — both wrong-signed |
| exp2 | llama | −0.012 (p=.004) | −0.005 (p=.32) | underpowered |
| exp2 | qwen | −0.012 (p=.0003) | −0.003 (p=.45) | underpowered |
| exp3 | llama_sem | −0.0135 [−.0213, −.0061] | −0.0207 [−.0299, −.0115] | REJECTED |
| exp4 | 9 valid groups | — | — | 6 REJECTED, 3 UNDERPOWERED, **0 SUPPORTED** |
| exp5 | llama | +0.0738 [+.060, +.088] | +0.0237 [+.011, +.037] | REJECTED |
| exp5 | mistral | +0.0294 [+.020, +.039] | +0.0360 [+.027, +.045] | REJECTED |
| exp5 | qwen | +0.0437 [+.023, +.065] | +0.0171 [−.001, +.035] | UNDERPOWERED |

Only exp1 is confirmatory. Everything after is exploratory and must be labelled
so. The registered manipulation gate (CPR ≥ 0.85) **failed in exp1** at 0.24–0.31
and first passed in exp2 at 1.000.

### A2. Qwen's exp5 TFT cell is inconclusive, not rejected
**Status: SUPPORTED — a reporting constraint, not a finding.**

+0.0171, CI [−0.0010, +0.0348], p = 0.058. Report as inconclusive. Writing
"rejected in all three models" is an OVERCLAIM.

### A3. Under neutral reasoning the state effect is positive and opponent-invariant
**Status: SUPPORTED (exp5, three models).**

All six cells positive; every ALLC cell matches the prediction, every TFT cell
violates it. Defensible sentence: *models use the state and do not condition that
use on whether the opponent retaliates.*

---

## B. Instrumentation

### B1. Token parity is necessary and not sufficient; density must match
**Status: SUPPORTED (exp1 → exp2, matched data both sides).**

Same model (Llama), same contrast, placebo densified from 44% to 85% content
against a 94%-content treatment:

| opponent | exp1 (44% placebo) | exp2 (85% placebo) | what changed |
|---|---|---|---|
| ALLC | +0.042, p < 1e-4 | **−0.0116, p = 0.0040** | **sign reversed, still significant** |
| TFT | +0.052, p < 1e-4 | **−0.0051, p = 0.3195** | collapsed to non-significance |

Source: `ep_exp2_llama.json`. Qwen replicates the pattern: −0.0121 (p = 0.0003)
vs ALLC, −0.0028 (p = 0.4481) vs TFT.

**CORRECTION.** This entry previously read "exp2 −0.012 (p = 0.24)". **That
p-value appears in no run of this project.** The effect size was right and the
p-value was invented, which understated the finding — vs ALLC the estimate did
not fade, it *reversed and remained significant*. Left visible per reporting
rule 7.

**OVERCLAIM: "the effect went away once the placebo was matched."** It went away
vs TFT and reversed vs ALLC. Name the opponent.

**Call it a confounded estimate, not a false positive.** A false positive is a
true null rejected by chance; this was a real effect of an unintended
manipulation. OVERCLAIM: "exp1 was a false positive."

### B2. Character padding cannot control token count under BPE
**Status: SUPPORTED.** `" 12"` and `"100"` differ in token count. Parity is
enforced at token-ID level, target auto-derived per tokenizer: Llama 34, Qwen 39,
Mistral 45. Mistral has no single-token newline; filler falls through to `' '`
(id 29473).

### B3. Turn-level standard errors understate uncertainty
**Status: SUPPORTED.** Inflation 0.62×–3.75× across exp2–exp5.

**Sub-1.0 inflation does not indicate a broken readout.** It appears in valid
groups (`exp5_mistral_sem_minimal` 0.91–1.08, `exp4_qwen_abs_logit` from 0.86).
OVERCLAIM, mine, retracted mid-project: "inflation below 1.0 is a second
signature of a dead cell."

### B4. A significant p-value from an uninspected cell is worth nothing
**Status: SUPPORTED.** `exp3_mistral_abs` printed `SIGN-FLIP: SUPPORTED`,
ATE_true(tft) = −0.2266, p<1e-4, at off-task 1.000 — computed entirely from
prose that contained no action tokens.

### B5. Measured causal effects shift with the inference stack
**Status: SUPPORTED (exp3 → exp4, full N).**

ATE_true replicates to ~0.002 across vLLM 0.27.1→0.11.0 / torch 2.13→2.8 /
transformers 5.15→4.57. Perturbation does not: llama_sem allc −0.1806 → −0.2218.

The block-vs-no-block contrast is stack-fragile; the block-vs-block contrast is
not. A study measuring only ATE_true would have concluded the stack was inert.

---

## C. Reading and truth

### C1. Models report the injected block in preference to the raw history
**Status: SUPPORTED (exp3, 3 models × 3 framings, 64,000 probed turns).**

DONOR_ECHO 0.940–1.000. `OFF_BY_ONE` ~0.000 rules out an arithmetic account;
`OTHER` ~0.000 rules out a parser fault. Flat across turn index.

Both sources are in the same context window. The model takes the block.

OVERCLAIM: "the model echoes a false score." The block says `Your score: 45` and
the model answers 45 — that is correct reading of a lying prompt. CPR in arm 3c
measures **trust in the block over the history**, not comprehension failure.

### C2. With reading held constant, the truth of the state barely matters
**Status: SUPPORTED (exp3), with one documented exception.**

Reading is ~100% in arm 3c *and* arm 3 — same template, same position, only the
numbers differ.

| exp3 defect rate | 3c (false) | 3 (true) |
|---|---|---|
| llama_sem allc | 0.097 | 0.085 |
| llama_sem tft | 0.124 | 0.100 |
| qwen_sem allc | 0.045 | 0.051 |
| qwen_sem tft | **0.306** | **0.068** |

This is stronger than "comprehension is not the bottleneck": the information is
demonstrably ingested and still does not move behaviour.

### C3. Qwen-vs-TFT retaliates against a betrayal that did not occur
**Status: was NOT TESTED (exp3) → SUPPORTED (exp6, 3 models, 114,000 falsified rows).**

The exp3 version of this claim rested on an ALLC/TFT asymmetry, because only
`donor_agent_score` was persisted. exp6 added `turns.displayed_opponent_last` and
falsified the field deliberately:

| arm | what it corrupts | qwen `3 − arm` vs tft |
|---|---|---|
| `3s` | score only | +0.0138 [+.0059, +.0217] |
| `3m` | **last move only** | **−0.4049** [−.4160, −.3938] |

Flipping one word moves defection 29× more than shifting the score by 15 points,
and *further* than replacing the entire block with a donor's (`3−3c` = −0.2016).
The mechanism is now measured, not inferred.

### C4. Numeric fields, not their content, drive Qwen under abstract labels
**Status: SUPPORTED, NARROW (exp3_qwen_abs, one model, one framing).**

ATE_true +0.745 / +0.738, p<1e-4. Three properties argue against a strategic
reading: off-task is clean; false numbers ≈ true numbers (0.754 vs 0.771);
opponent-invariant to three decimals (0.771 ALLC, 0.770 TFT). A strategic
response would differ between a retaliator and a pushover.

### C5. The opponent's last move dominates; the cumulative score does not
**Status: SUPPORTED (exp6, 3 models × 2 opponents × N=1000, LOGIT).**

Turn 0 excluded from both arms; episode-level bootstrap, 10,000 resamples.

| group | opp | `3−3s` score | `3−3m` move | ratio |
|---|---|---|---|---|
| qwen | tft | +0.0138 [+.0059, +.0217] | −0.4049 [−.4160, −.3938] | 29× |
| qwen | allc | +0.0048 [−.0017, +.0113] ns | −0.2839 [−.2922, −.2758] | 59× |
| llama | allc | −0.0227 [−.0327, −.0127] | −0.0931 [−.1043, −.0820] | 4.1× |
| llama | tft | −0.0317 [−.0447, −.0186] | −0.0891 [−.1027, −.0754] | 2.8× |
| mistral | tft | +0.0001 [−.0002, +.0003] ns | −0.0141 [−.0163, −.0120] | — |
| mistral | allc | −0.0004 [−.0007, −.0001] | −0.0115 [−.0132, −.0099] | 29× |

Larger in 6 of 6 cells, by 2.8×–59×.

**OVERCLAIM: "score falsification has no effect."** Four of six score contrasts
exclude zero. They are small and in qwen the sign reverses, but they are not
null. Write "dominates", never "does nothing". This sentence was drafted in the
wrong form once already, from an MDE heuristic rather than a bootstrap CI.

### C6. The model reads the false state perfectly and acts on it barely
**Status: SUPPORTED (exp6, 30,000 probes, 3 models, unanimous).**

Arm 3s displays a score wrong by 15 and changes nothing else. Own-score probe
answers compared against both the displayed value and the true cumulative payoff:

| group | probes | matched **displayed** | matched true |
|---|---|---|---|
| llama_sem_logit | 10,000 | **100.0%** | 0 |
| qwen_sem_logit | 10,000 | **100.0%** | 0 |
| mistral_sem_logit | 10,000 | **100.0%** | 0 |

Perfect reproduction of a falsehood, against a behavioural change of 0.5–3.2pp.
This is the direct form of what C1 and C2 had to infer, and the answer to
"perhaps the model never read the block".

**Corollary — CPR in a falsifying arm is a belief measure, not a validity gate.**
CPR scores against the *true* state, so a model that reads and trusts a lying
block is marked wrong. Arm 3s scores CPR **0.000** in every group. Gate CPR on
arm 3 alone (1.000 in all six groups); applying it to 3b, 3c, 3s or 3m discards
the cells carrying the result.

### C7. The last-move effect does not survive chain-of-thought in every model
**Status: SUPPORTED (exp6, minimal-CoT, 128 tokens) — a limitation, state it.**

| model | opp | `3−3m` LOGIT | `3−3m` CoT |
|---|---|---|---|
| qwen | allc | −0.2839 | −0.1447 [−.1612, −.1281] |
| qwen | tft | −0.4049 | −0.0520 [−.0652, −.0383] |
| mistral | allc | −0.0115 | −0.0328 [−.0416, −.0237] |
| mistral | tft | −0.0141 | −0.0344 [−.0431, −.0257] |
| llama | allc | −0.0931 | **+0.0126** [+.0007, +.0244] |
| llama | tft | −0.0891 | **−0.0155** [−.0267, −.0040] |

Qwen shrinks 50–87%, mistral **grows** (freed from a LOGIT floor of 0.0001),
llama collapses to ~zero and reverses sign vs ALLC.

**OVERCLAIM: "falsifying the last move causes defection in LLMs."** Present under
LOGIT in three of three models, under CoT in two of three. The literature uses
CoT. Name the readout and name llama.

### C8. Arm 3c is a weak instrument, and vs ALLC a zero-dose control
**Status: SUPPORTED (exp6, 114,000 rows, self-join against ground truth).**

| arm | opponent | falsification rate |
|---|---|---|
| `3m` | both, all models | 1.0000 |
| `3c` | **allc**, all models | **0.0000** |
| `3c` | tft — llama / mistral / qwen | 0.2988 / 0.1392 / 0.3765 |

Against ALLC every donor also shows `Cooperate`, so 3c cannot corrupt the field.
Its effect there is −0.0140 / +0.0000 / +0.0128 — zero dose, ~zero effect, in a
cell already in the corpus. Against TFT, qwen's `3−3c` is −0.2016, replicating
exp2 (−0.2407) and exp3 (−0.2375).

The historical 24pp effect was never about score. It was the last move,
appearing only where a donor happened to corrupt it.

**NOT TESTED — the per-falsified-row overshoot.** Rescaled by its own dose, 3c
exceeds 3m by 26% (llama, −0.1127 vs −0.0891) and 32% (qwen, −0.5356 vs −0.4049).
Qwen's score effect runs the wrong direction, so score cannot account for it.
Candidates: the fields interacting when both are wrong, or 3c's falsified rows
landing on higher-leverage turns. Limitations section, not Results.

---

### C9. Models track the field they use and fail the field they do not
**Status: SUPPORTED (`analysis/14` part I, 6 groups, 3 models, both readouts).**

Arm 1 has no state block, so every field must be reconstructed from `[HISTORY]`.
CPR is **all three probes correct**, and the three probes are not the same task.
Decomposed, turn 0 excluded (there every field is 0/None and a correct answer is
not a tracking result):

| model | readout | own score — ARITHMETIC | **opponent's last move — RECALL** | rounds — COUNTING |
|---|---|---|---|---|
| llama | logit | 0.000 | **0.988** | 1.000 |
| llama | scratchpad | 0.000 | **0.912** | 1.000 |
| qwen | logit | 0.250 | **0.999** | 1.000 |
| qwen | scratchpad | 0.150 | **0.921** | 1.000 |
| mistral | logit | 0.250 | **1.000** | 1.000 |
| mistral | scratchpad | 0.000 | **0.965** | 0.962 |

Unanimous: the opponent's last move is recalled at **0.91–1.00 with no block at
all**, while the cumulative score sits at 0.00–0.25.

Read with C5 (the last move is the only field whose falsification moves
behaviour), this converts the project's central null into a mechanism:

> the state block repairs the field the model can neither compute nor use, and
> adds nothing to the field it already tracks — which is why a perfectly-read
> block produces no behavioural change.

**It also sharpens the open question.** The model demonstrably recalls the true
last move, and the truth is one section below the block in `[HISTORY]`. When the
block asserts otherwise, behaviour follows **the block**. That is not a
comprehension failure; it is preference for an injected summary over the model's
own accurate recall. Whether that is *use of state* or *conflict resolution* is
exactly what exp7's no-history condition tests (`PREREGISTRATION_EXP7.md`).

**RETIRED BY THIS CLAIM — "arm-1 CPR is 0.20, so models cannot track the
state".** The 0.20 is a conjunction dragged to the floor by one sub-probe that
asks a 7B model to sum twenty payoffs. It is an ARITHMETIC result. Report CPR
per field; never quote the all-three figure as a state-tracking denominator.

### C10. The last-move effect does not require a contradiction
**Status: SUPPORTED (exp7 `nohist`, llama + mistral; qwen at a floor).**

Every experiment before exp7 rendered `[HISTORY]` directly below the block, so
arms 3c/3s/3m were *contradiction* manipulations, not false-state ones. The
rival account — "the model discounts a claim the adjacent context refutes, and
discounts it more when refuting it is cheap" — predicted the entire
score-vs-move asymmetry. exp7 removes the history so the block is the only
source of state and nothing can contradict it.

Deprivation is total and repair is complete:

| group | arm-1 own score | arm-1 last move | arm-1 rounds | arm-3 CPR |
|---|---|---|---|---|
| llama nohist | 0.000 | 0.007 | 0.000 | 1.000 |
| mistral nohist | 0.000 | 0.000 | 0.000 | 1.000 |

And the effect persists:

| group | opp | semantic | **nohist** | survives |
|---|---|---|---|---|
| llama | allc | −0.088 | **−0.0564** [−.0635, −.0495] | Holm + BH |
| llama | tft | −0.085 | **−0.0337** [−.0397, −.0274] | Holm + BH |
| mistral | allc | −0.011 | **−0.0106** [−.0138, −.0073] | Holm + BH |
| mistral | tft | −0.013 | **−0.0054** [−.0082, −.0025] | Holm + BH |

llama retains 40–64% of its semantic effect, mistral essentially all of it. The
shrinkage is itself informative: part of the semantic effect *was* the
contradiction, and part is genuine use of the block.

**qwen contributes nothing here.** `P(D|3) = 0.0007` with history removed —
a floor, not a null. Report it as uninformative.

**This is the claim that closes the loop.** Without any state source the model
reports nothing; the block restores it completely; restoring it still produces
no opponent-conditional play, and falsifying one field of it still moves
behaviour. "Comprehension is not the bottleneck" is earned rather than inferred.

**NOT SETTLED — the lexical account.** exp7's abstract condition does not test
it, because X/Y puts the models in a different regime rather than a compressed
one: `P(D|3)` goes 0.086 → 0.714 (llama) and 0.051 → 0.770 (qwen), and
`ATE_true` in `qwen_abs` is +0.75 against +0.017 in `qwen_sem`. C4 already
established that qwen under X/Y is driven by numeric-field presence and is
opponent-invariant. Limitations.

### C11. The registered criteria are met in one condition, by the wrong arm
**Status: SUPPORTED as a description, REJECTED as support for the hypothesis.
Replicated across two independent runs.**

After the swap rescore (see C12), `exp3_qwen_swap` and `exp7_qwen_swap` both
satisfy the pre-registered sign-flip with a passing manipulation check:

| run | ATE_true allc | ATE_true tft |
|---|---|---|
| exp3_qwen_swap | **+0.4559** | −0.0659 |
| exp7_qwen_swap | **+0.4636** | −0.0792 |

Arm-level rates agree to two decimals across runs. **But the mechanism inverts
the hypothesis, and that replicates too.** Opponent spread per arm:

| arm | exp3 | exp7 |
|---|---|---|
| 1 (no block) | 0.073 | 0.078 |
| **3b (contentless block)** | **0.693** | **0.710** |
| 3 (true state block) | 0.171 | 0.167 |

A block with no real state in it makes Qwen ~10× more opponent-sensitive than no
block at all; the true state block damps it back to 0.17. The criteria are met
because arm 3b swings, not because arm 3 does. The registered hypothesis was
that true state *enables* opponent-conditional play.

llama shows no flip in either run (+0.067/+0.033 and +0.055/+0.025). Qwen and
swap only.

**OVERCLAIM: "the pre-registered hypothesis was supported."** Required form:
*"the criteria are met in Qwen under label swap, replicated across two runs, and
driven by the placebo arm's opponent sensitivity rather than the treatment's."*

**OVERCLAIM: "the hypothesis was rejected in every group."** No longer true.
Rejected in every semantic and abstract group across seven experiments; met in
this one condition.

### C12. The swap probe scorer graded in the wrong label space
**Status: SUPPORTED — a scorer defect, found after the fact, five groups recovered.**

The swap condition inverts the action words; the probe scorer compared answers
against **unswapped** truth. Every non-turn-0 `opponent_last` answer was marked
wrong and CPR collapsed to exactly the turn-0-only rate, 0.200. Contingency
tables are clean inversions (66.9%–79.1%), so rescoring is justified.

Behaviour was never affected: `cdx/backends_vllm.py:154` inverts the action
surface forms correctly, so `P(defect)` is always the true action. Only the
manipulation check was misgraded.

| group | arm 3 as run | rescored |
|---|---|---|
| exp3_llama_swap · exp3_mistral_swap · exp3_qwen_swap | 0.200 | **1.000 PASS** |
| exp7_llama_swap · exp7_qwen_swap | 0.200 | **1.000 PASS** |

**A second defect sat inside the fix.** `analysis/10_rescore_swap.py` compared
`str(got).strip() == str(want).strip()` rather than importing
`cdx.probe.normalise`, so Mistral's verbose turn-0 answer *"None (since no round
has been played yet)"* scored wrong and `exp3_mistral_swap` appeared to cap at
0.800. The production scorer always handled it — the stored `cpr_score` for that
turn is 1. Corrected, and the test is that the script's "CPR as run" column
reproduces the stored `cpr_score`: **0 mismatches across all five groups.**
Mistral was recovered by fixing an analysis script, not by anything about the
model.

---

## D. Presentation

### D1. Inserting a state block changes behaviour more than its content does
**Status: SUPPORTED, model-dependent — always name the model.**

exp3 perturbation, semantic: llama −0.181/−0.192; qwen +0.039/+0.038;
mistral 0.000 (floor).

OVERCLAIM: "the container effect exists in LLMs." It is large and negative in
Llama, small and **positive** in Qwen, absent in Mistral.

### D2. The container effect is lexical
**Status: SUPPORTED — pre-specified falsification test, survived.**

| model | semantic | abstract |
|---|---|---|
| llama allc | −0.181 [−.192, −.169] | +0.007 [−.003, +.016] ns |
| llama tft | −0.192 [−.205, −.179] | +0.029 [+.019, +.040] |
| qwen allc | +0.039 [+.033, +.044] | −0.055 [−.064, −.046] |
| qwen tft | +0.038 [+.032, +.044] | −0.065 [−.076, −.053] |

In Llama it disappears; **in Qwen it reverses sign.** Baselines move too: Llama
defects 0.28–0.31 under Cooperate/Defect and 0.71–0.74 under X/Y.

OVERCLAIM: "under abstract labels the container effect vanishes." True of Llama,
false of Qwen.

### D3. A matched placebo is required or two large effects cancel
**Status: SUPPORTED (exp4_qwen_sem_scratchpad, allc).**

| contrast | estimate | 95% CI | p |
|---|---|---|---|
| perturbation (1 → 3b) | +0.1934 | [+.1741, +.2126] | <1e-4 |
| ATE_true (3b → 3) | −0.2135 | [−.2279, −.1989] | <1e-4 |
| **ATE_naive (1 → 3)** | **−0.0202** | [−.0397, −.0000] | 0.046 |

The standard no-block-vs-block comparison reports "state does not matter" while
two ~20-point effects of opposite sign are present.

---

## E. Readout and reasoning

### E1. Defection rises sharply from constrained readout to chain-of-thought
**Status: SUPPORTED — but only jointly by exp4 and exp5. Never cite exp4 alone.**

| semantic, P(D given arm 3b) | LOGIT | CoT guided | CoT minimal |
|---|---|---|---|
| llama | 0.102 | 0.579 | 0.639 |
| qwen | 0.042 | 0.597 | 0.597 |
| mistral | 0.000 | 0.335 | 0.406 |

Three architectures, three different floors, same direction.

### E2. The rise is not caused by naming the finite horizon
**Status: SUPPORTED (exp5).**

Removing "how many rounds remain" leaves defection unchanged or higher in all six
cells. Averages: llama 0.603 → 0.694, qwen 0.529 → 0.689, mistral 0.369 → 0.407.

**The horizon effect runs backwards.** Naming the rounds made llama and qwen
defect *less* (arm 1: 0.731 → 0.557 llama; 0.650 → 0.404 qwen) — the opposite of
the backward-induction prediction, and evidence against a textbook-retrieval
account.

### E3. exp5 controls salience, not availability
**Status: SUPPORTED — a scope limit that must appear beside every E1/E2 claim.**

The rules section already states the game lasts exactly 20 rounds. The horizon
was never hidden; the instruction only made it salient.
`HorizonMode.STOCHASTIC` — where no last round exists and backward induction
cannot apply — is implemented and **was not run**.

OVERCLAIM: "reasoning inherently drives defection", "backward induction is
falsified." Neither is licensed. One instruction, one 128-token budget, one task.

### E4. Reasoning attenuates the container effect by ~60% without abolishing it
**Status: SUPPORTED, NARROW (llama only — the other two had no effect to attenuate).**

| condition | allc | tft |
|---|---|---|
| LOGIT | −0.2218 [−.2387, −.2052] | −0.2069 [−.2253, −.1880] |
| CoT minimal | −0.0920 [−.1054, −.0788] | −0.0747 [−.0869, −.0620] |
| CoT guided | +0.0215 [+.0039, +.0390] | +0.0161 [−.0012, +.0334] |

No CI overlap between any pair. Two separable mechanisms: reasoning removes ~60%,
directing attention at the state removes the remainder.

### E5. Under neutral reasoning the container effect appears in all three models
**Status: SUPPORTED (exp5). New — not implied by exp4's guided cells.**

Perturbation vs allc: llama −0.222 → −0.092 ***; qwen **+0.033 → −0.053** ***;
mistral 0.000 → −0.021 ***. Qwen's sign reverses. Five of six point estimates
negative, four significant.

---

## F. Retracted

Kept visible. Each was asserted during this project and withdrawn by later data.

| claim | asserted | withdrawn by |
|---|---|---|
| A block containing the answer makes turn-0 reporting worse (1.000 → 0.190) | exp2 reading | `.ljust()` padding artefact; natural rendering gives 1.000 in every arm |
| exp1's +0.042 was a false positive | exp1 write-up | It was a confounded estimate — see B1 |
| Chain-of-thought kills the container effect | exp4 guided cells | exp5: −0.092, p<1e-4 under a neutral instruction (E4) |
| Qwen shows a conditional state-comprehension effect (−0.2135 / −0.1015) | exp4 during the run | exp5: +0.0437 / +0.0171 under a neutral instruction |
| Inflation below 1.0× marks a dead cell | mid-project | Appears in valid groups (B3) |

---

## G. Not tested — Limitations

1. **Availability vs salience.** Stochastic horizon implemented, unrun (E3).
2. **Scale and family.** Three 7–8B open-weight instruct models. No frontier
   model, no scale ladder.
3. **Scripted opponents only.** No LLM-vs-LLM. `game.py` raises on
   `OpponentPolicy.LLM`; the two-sided perspective flip is unbuilt and untested.
4. **One decoding configuration.** Single temperature, `LOGPROBS_TOP_K = 20`,
   128-token reasoning budget.
5. **Coverage gaps from exclusions.** `mistral_abs` in both readouts,
   `qwen_abs_scratchpad`. Reported, still gaps.
6. **Stack drift measured, not diagnosed.** ~4pp (B5). Which change caused it is
   unknown.
7. **Donor's last move not persisted** (C3).
8. **Does the treatment effect differ by model?** ATE_true vs ALLC is +0.074 /
   +0.044 / +0.029 across llama / qwen / mistral. **Untested.** A bootstrap
   difference-of-differences decides whether the paper reports a pooled effect or
   three case studies. Run before drafting Results.

---

## I. Retired by `analysis/14` — do not write these

Six sentences that were available before the reviewer-response analyses and are
no longer licensed. Every one was found internally, before submission. Kept
visible, per reporting rule 7.

| # | retired sentence | what killed it |
|---|---|---|
| 1 | "arm-1 CPR is 0.20, so models cannot track the state" | part I. It is a conjunction floored by an arithmetic sub-probe; the *used* field is recalled at 0.91–1.00 (C9). |
| 2 | qwen logit revealed-stratum ATE = −0.4533 | part A. Selection gap **+0.371** — arm 3 reaches the revealed state 43.6% of the time, arm 3b 6.5%. It compares two different populations. |
| 3 | the "per lied row" rescaling, and the 26–32% overshoot from it (C8) | part D(iii). Falsification in arm 3c is post-treatment selected: prior-defection differs by +0.088 (llama) and **+0.241** (qwen) between falsified and unfalsified rows. Report the marginal 3c effect and the falsification rate separately. |
| 4 | any pooled cross-model effect | part B. Joint heterogeneity p = 0.0001 in all four strata, spread up to 0.084. **Three case studies, not one effect.** Every "models do X" names the model. |
| 5 | "arm 3b is non-diagnostic / contentless" | part H. The detrended parity contrast is −0.026 (llama), −0.069 (mistral CoT), +0.016 (qwen CoT), CIs excluding zero, while arm 1 — which has no parity line — is flat. The placebo carries one bit and the model acts on it. ATE_true is therefore **conservative**. **exp7 makes this worse and sharper:** with `[HISTORY]` removed, parity is the only turn-index signal left and the leak reaches −0.049 to −0.093 — *larger than llama's headline effect* — in every no-history group, with arm 1 flat throughout. Scope it precisely: this contaminates `ATE_true` and `perturbation`, which involve arm 3b. It does **not** touch `content_move` / `content_score` / `content_donor`, which compare arm 3 against arms carrying the real state block. |
| 6 | qwen scratchpad `3−3m` = −0.1447, unqualified | part G. **31.2%** of decisions in that contrast's cells sit below action-mass 0.25, so the estimate is partly a renormalisation of a small number. Quote the fragile share beside it, or restrict to solid decisions. |

**Two claims were strengthened rather than retired.**

- **A1 survives its best defence.** Restricting to turns where the opponent's
  type has actually been revealed rescues the pre-registered hypothesis in **no**
  group (part A). Report the stratified test; it forecloses the objection.
- **C5's score sentence stands.** All four score contrasts that exclude zero also
  survive Holm within the six-member score family (part F). Note the family:
  34 of 48 survive Holm and 40 of 48 survive BH across the full exp6 family, and
  the correction family must be pre-specified rather than chosen after the fact.

**The C8 overshoot is explained and can leave Limitations.** In qwen the two
fields interact when both are wrong: move+score = 0.4636 against move-only =
0.3442, interaction **+0.1211 [+0.0904, +0.1519], p = 0.0001**. In llama there is
no interaction (p = 0.48) and turn composition covers it. Different mechanisms in
different models — consistent with the heterogeneity verdict above.

---

## H. Recommended spine

The dissociation ladder, because the study has a measurement at every rung and
nothing else in the literature does:

| rung | measured | result |
|---|---|---|
| available | by construction | — |
| **reconstructable without the block** | **arm-1 CPR per field** | **the used field yes (0.91–1.00), the score no (0.00–0.25) — C9** |
| readable | CPR arm 3 = 1.000, 12,800 probes | yes |
| block preferred over history | DONOR_ECHO 0.94–1.00; 30,000/30,000 echo a falsified score | yes |
| truth of content matters | arm 3 vs 3s vs 3m | only for one field (C5) |
| presence/form matters | perturbation | a lot, model-dependent (D1, D2) |
| conditioned on opponent | sign-flip, pooled **and** revealed-stratum | no, in any valid group (A1) |

The second rung is new and load-bearing. Without it the ladder starts at
"readable", and a reviewer can ask what was ever broken. With it the claim is
specific: the model already tracks the field it acts on, the block repairs a
field it cannot compute and does not use, and repairing it changes nothing.

Chain-of-thought (E) is a **second contribution**, not the spine. It answers
"does this reach the regime the literature uses" and carries its own control. If
it becomes the headline, the paper turns into a claim about CoT with a
single-instruction control behind it — a much smaller target than the ladder.
