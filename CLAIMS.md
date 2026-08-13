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

exp1 ATE_true +0.042 (p<1e-4) with a 44%-content placebo against a 94%-content
treatment; exp2 −0.012 (p=0.24) after densifying to 85%.

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

## H. Recommended spine

The dissociation ladder, because the study has a measurement at every rung and
nothing else in the literature does:

| rung | measured | result |
|---|---|---|
| available | by construction | — |
| readable | CPR arm 3 = 1.000, 12,800 probes | yes |
| block preferred over history | DONOR_ECHO 0.94–1.00 | yes |
| truth of content matters | arm 3 vs 3c | barely (C2) |
| presence/form matters | perturbation | a lot, model-dependent (D1, D2) |
| conditioned on opponent | sign-flip | no, in any valid group (A1) |

Chain-of-thought (E) is a **second contribution**, not the spine. It answers
"does this reach the regime the literature uses" and carries its own control. If
it becomes the headline, the paper turns into a claim about CoT with a
single-instruction control behind it — a much smaller target than the ladder.