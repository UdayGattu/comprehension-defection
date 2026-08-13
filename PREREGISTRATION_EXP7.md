# Pre-registration — Experiment 7

**Status: frozen before data. Not yet run.**
Driver `scripts/exp7_confounds.sh` · probe suite unchanged
(`12c9a10d970099c1e56fc00b77d17f8ab00e14ff6a084290f27066583106a689`).

exp1 is the only pre-registered experiment in this project. exp7 is registered
too, deliberately: it exists to answer reviewers who believe the exp6 headline
is confounded, and an exploratory answer to that charge is worth very little.
Predictions below are committed before any exp7 inference runs, and every
outcome — including the one that retracts the headline — is written down here
with what it would license.

---

## 1. The claim under test

**CLAIMS.md C5**, from exp6: *the opponent's last-move field dominates; the
cumulative score does not.* Falsifying only `Opponent's last move` moves
defection by up to **−0.4049** (qwen vs TFT), **2.8×–59×** more than falsifying
only the score, in **6 of 6** cells. 30,000/30,000 probes reproduced the
falsified score perfectly while behaviour moved 0.5–3.2 pp (**C6**).

Two confounds are unaddressed, and both are unaddressable from any database in
this repository, because neither factor was ever varied.

### Objection 1 — lexical priming

exp6 ran under `Framing.SEMANTIC` only, and arm 3m injects the literal token
**"Defect"**. This project's own exp3 (**D2**) measured labels dominating
everything else it tested: Llama's baseline defection 0.28–0.31 under
Cooperate/Defect versus 0.71–0.74 under X/Y, and Qwen's container effect
**reversing sign** between framings. So "the model conditions on the last-move
field" is currently indistinguishable from "the string 'Defect' raises
P(Defect)".

### Objection 2 — the block is redundant with `[HISTORY]`

`PromptAssembler.assemble` has always rendered `[HISTORY]` with every round in
it, one section below the block. Arms 3c/3s/3m are therefore not false-state
manipulations but **contradiction** manipulations: the truth sits in the same
context window, trivially checkable for the last move (read the final line) and
arithmetically expensive for the score (sum twenty payoffs).

That admits a rival account of the whole exp6 pattern — models react to a
**detected** contradiction between an injected summary and the raw log, and
detection is cheap for the move and expensive for the score. It predicts the
score/move asymmetry, the perfect probe reproduction of the score lie
(reproducing is not believing), and the ALLC/TFT difference (**C8**).

---

## 2. Hypotheses

| tag | account | mechanism |
|---|---|---|
| **H-state** | the exp6 reading | behaviour conditions on the *information* in the last-move field, which is the entire input to optimal play against TFT |
| **H-lex** | objection 1 | the token "Defect" in context raises P(Defect); the field is incidental |
| **H-conflict** | objection 2 | behaviour responds to a *detected* contradiction between block and log; the field is incidental, and detection cost explains the asymmetry |

H-lex and H-conflict are not mutually exclusive and both may hold in part. The
design is built so that each has a cell where it makes a prediction the others
do not.

---

## 3. Design

3 models × 5 arms × 2 opponents × N=1000 × LOGIT, in four conditions.
`{1, 3b, 3, 3s, 3m}`; arm 3c is dropped (exp6 measured its dose at **0.0000**
vs ALLC and 0.14–0.38 vs TFT — a weak instrument whose two fields the
deliberate arms measure at a known dose).

| cond | labels | `[HISTORY]` | models | cells | answers |
|---|---|---|---|---|---|
| `abs` | X / Y | present | llama, qwen | 20 | objection 1 |
| `swap` | Cooperate/Defect, **inverted mapping** | present | llama, qwen | 20 | objection 1, ceiling-proof |
| `nohist` | Cooperate / Defect | **absent** | llama, qwen, mistral | 30 | objection 2 |
| `absnohist` | X / Y | **absent** | llama, qwen | 20 | interaction |

**N = 1000, matching exp6 exactly.** The primary estimands are between-run
differences against `exp6_*_sem_logit`, so N, arms, opponents, readout, budget
and code path must match or the comparison acquires a second difference. exp6's
episode-level CI half-widths were ±0.011 (qwen) to ±0.014 (llama), so a
difference-in-differences resolves at ≈0.016.

**Controls are exp6, not re-run.** `exp6_${model}_sem_logit` is the
semantic-with-history cell for every contrast below. Re-running it would cost
40 minutes and buy nothing; the stack is the same H100 / vLLM 0.11.0 image.
*If the image differs, `exp6_*_sem_logit` must be re-run in-session* — B5
measured perturbation contrasts moving 4 pp across stacks, and the
block-vs-no-block contrast is the stack-fragile one.

**Readout.** LOGIT only for the registered tests. Every content effect in this
project is a LOGIT effect, `exp4_qwen_abs_scratchpad` was excluded at off-task
0.201, and C7 already records that the CoT regime attenuates the effect. A
no-history CoT phase exists (`MODE=pad`, llama + qwen, minimal prompt, 128
tokens) as an **exploratory** extension, labelled so.

---

## 4. Estimands

Episode-level bootstrap, 10,000 resamples, seed 20260811,
`analysis/02_episode_level.py`. Turn-level intervals are never quoted (B3).
Turn 0 excluded from both arms of every move contrast — at turn 0 there is no
last move, arm 3m is byte-identical to arm 3, and those rows carry no
manipulation (`donor_degenerate = 1`).

```
E_move(cond)  = P(defect | 3, cond) − P(defect | 3m, cond)
E_score(cond) = P(defect | 3, cond) − P(defect | 3s, cond)
R(cond)       = |E_move(cond)| / |E_score(cond)|
DiD(cond)     = E_move(cond) − E_move(sem, history)        [exp6 reference]
```

`E_move` is reported on **two scales** and both are pre-committed:

1. **risk difference** — comparable to every number in exp6 and CLAIMS.md;
2. **log-odds ratio** — because the abstract cells sit near a ceiling. exp3
   measured qwen_abs arm 3 at 0.771 / 0.770 with **zero** of 2,000 episodes
   never defecting, and llama's abstract baselines at 0.71–0.74. `E_move` is
   negative (3m defects *more* than 3), so the largest risk difference
   observable there is ≈0.23 against a semantic effect of −0.4049. A compressed
   estimate would mimic H-lex winning.

**Ceiling rule, fixed in advance.** Any cell with P(defect | arm 3) ≥ 0.85 or
≤ 0.15 is **ceiling-limited**: its risk difference is reported but never used
to support or reject a hypothesis, and only the odds ratio is interpreted. If
both `abs` cells are ceiling-limited and their odds-ratio CIs include 1 with
half-width > 0.5 in log-odds, objection 1 is recorded **UNRESOLVED** by the
abstract cell and the `swap` cell carries it.

---

## 5. Predictions

Signed, before data.

### 5.1 `abs` — objection 1

| | H-state | H-lex | H-conflict |
|---|---|---|---|
| `E_move(abs)` | negative, OR CI excludes 1 | ≈ 0, CI includes 0 in **both** models | negative (X/Y is contradicted by the log just as words are) |
| `DiD(abs)` | small relative to `E_move(sem)` | ≈ −`E_move(sem)`, CI excludes 0 | small |
| `E_score(abs)` | ≈ 0, as exp6 | ≈ 0 | ≈ 0 |

**The abstract cell does not separate H-state from H-conflict.** Both predict
survival. It separates H-lex from the other two. That asymmetry is the reason
objection 2 is the one to run first.

### 5.2 `swap` — objection 1, on a sign

With the mapping inverted the word "Cooperate" *means* defect. Arm 3m flips the
**action**, so where the opponent truly cooperated the block asserts a betrayal
and renders it with the cooperative word. Recorded `defect_rate` is the action,
not the token.

| | H-state | H-lex |
|---|---|---|
| `E_move(swap)` | **same sign as exp6** (negative): the asserted betrayal still provokes retaliation | **opposite sign** (positive): the cooperative token suppresses the defect action |

A sign test survives any monotone compression of the scale, which is what a
ceiling is. This is the cheapest insurance in the design against an
uninterpretable phase 1, at ~7 minutes per model.

### 5.3 `nohist` — objection 2

| | H-state | H-conflict | H-lex |
|---|---|---|---|
| `E_move(nohist)` | **preserved or larger** — the block is the only source and is uncontested | **collapses toward 0** — no contradiction remains to detect | unchanged from semantic (the token is still there) |
| `E_score(nohist)` | ≈ 0 (score is sunk against TFT and ALLC whether believed or not) | ≈ 0 (no conflict to detect) | ≈ 0 |
| `R(nohist)` vs `R(sem)` | preserved, ≫ 1 | undefined / falls, both terms → 0 | preserved |
| `CPR(3, nohist)` | ~1.000: the block is read even with nothing to corroborate it | ~1.000 | ~1.000 |
| `CPR(1, nohist)` | floor: no state is present | floor | floor |

**`CPR(3, nohist)` is what makes a behavioural null interpretable.** With the
log gone, the block is the only way to answer a state probe. If CPR(3) stays at
1.000 while `E_move` collapses, the collapse cannot be attributed to "the model
stopped reading the block" or "the model distrusts an uncorroborated summary" —
it read it, reported it, and did not act on it. That measurement does not exist
anywhere in exp1–exp6, because the history could always answer the probe.

### 5.4 `absnohist` — interaction

Only cell that can show non-additivity. If `E_move` survives X/Y with the log
present but dies once the log is removed, H-lex and H-conflict are entangled and
neither single-factor phase would reveal it. No directional prediction is
registered; this cell is exploratory and labelled so.

### 5.5 `pad` — exploratory

C7 records that the last-move effect does not survive CoT in llama
(−0.0891 → −0.0155) and shrinks 50–87% in qwen. Under H-conflict, reasoning is
*when* the cross-check happens, so removing the log should **restore** the
effect under CoT. Under H-state the CoT collapse is about reasoning overriding
the field and removal should not restore it. Directional, but exploratory:
one instruction, one 128-token budget, two models.

---

## 6. What each result licenses

| # | result | licensed claim | action on CLAIMS.md |
|---|---|---|---|
| 1 | `E_move(abs)` survives (OR CI excludes 1, both models) **and** `E_move(nohist)` preserved or larger | *Behaviour conditions on the last-move field's information, robust to the labels used and to the presence of a contradicting log.* The strongest form of C5 available at this budget. | C5 upgraded to SUPPORTED with a pre-registered falsification test, matching D2's standing |
| 2 | `E_move(abs)` CI includes 0 in both models, semantic effect replicates, `DiD(abs)` CI excludes 0 | *The exp6 effect is lexical: the token "Defect" raises P(Defect); the field is incidental.* Converges with D2, which found the container effect lexical from the same manipulation. | **C5 RETRACTED in place** (reporting rule 7); rewritten as a token-level claim; C3 and C8's mechanism sentences rewritten with it |
| 3 | `E_move(nohist)` collapses (≥50% toward 0 in ≥2 of 3 models) while CPR(3) ≈ 1.000 | *Models respond to a detected conflict between an injected summary and the raw log, not to the summary's content, and detection cost explains which field wins.* Sharper and more general than the game-theoretic version — a claim about context engineering. | C5 restated as a conflict-resolution result; C3/C6/C8 rewritten; the paper's spine gains a rung: **"block preferred over history" holds for reporting but not for acting** |
| 4 | `E_move(nohist)` preserved **and** `E_move(abs)` null | The information is not what matters and the contradiction is not what matters: the token is. Same action as #2, with the added statement that the effect does not need a log to contradict. | C5 RETRACTED; report both cells |
| 5 | `E_move(nohist)` **grows** and `R(nohist) ≫ R(sem)` | *An uncontested injected summary is acted on; a contested one is discounted, and the discount is field-specific.* The most useful outcome for practice, and it makes the exp6 numbers a lower bound. | C5 kept with a scope clause; new claim added for the uncontested regime |
| 6 | both `abs` cells ceiling-limited with wide odds CIs | nothing about objection 1 from `abs` | record UNRESOLVED; `swap` decides; if `swap` is also ambiguous, objection 1 stands as a Limitation |

### Explicitly: what would falsify the headline

C5 ("the last-move field dominates; the score does not") is **falsified** if
either holds:

* **(a)** `E_move(abs)` CI includes 0 in **both** llama and qwen while the
  cells are not ceiling-limited and `E_score` is unchanged — the field's
  information was preserved and the effect went away with the word; **or**
* **(b)** `E_move(nohist)` shrinks by more than 50% toward 0 in **≥2 of 3**
  models while `CPR(3, nohist)` ≥ 0.95 — the model reads the block, has no
  competing source, and still does not act on it, so the exp6 effect was a
  response to contradiction rather than to state.

In either case C5 is marked RETRACTED in place, the original left standing, per
reporting rule 7 — as exp5 retracted Qwen's state-comprehension effect.

---

## 7. Gates and abort criteria

Pre-run exclusions and in-run gates, fixed here so no cell can be dropped after
seeing its estimate.

1. **`mistral_abs` is excluded before the run.** Off-task 1.000 in
   `exp3_mistral_abs` and 1.000 again in `exp4_mistral_abs_logit` — two
   independent inference stacks. Not run in any abstract condition.
2. **Off-task gate: > 0.10 in arms 1, 3b or 3** of a group excludes the group
   from causal claims (B4). Reported in full, as `exp3_mistral_abs` was.
3. **The CPR gate applies to arms 1, 3b and 3 only.** CPR scores against the
   *true* state, so arms 3s and 3m are expected at ~0.000 — that is the
   manipulation working, not a failure (C6 corollary).
4. **Deprivation gate (`nohist` only): CPR(3) − CPR(1) ≥ 0.30.** Below that,
   removing the history did not create a deprivation; if arm-1 CPR stays high,
   state is leaking from another section and the condition is void. Printed by
   `scripts/gpu_run.py` and by the driver.
5. **History gate (driver, per group).** Two independent readings, both must
   agree: `[HISTORY]` present in stored `prompt_full` iff expected, and
   `COUNT(DISTINCT turns.prompt_tokens)` = 1 per cell iff history is off.
   History is the only section that grows with the turn index, so the second
   check covers 100% of rows rather than the three episodes that store
   `prompt_full`. Either disagreement aborts the group.
6. **Falsification gate (inherited from exp6).** `displayed_opponent_last` must
   be populated for arms 3s/3m or the lying arms told the truth.
7. **Pre-flight gate.** `scripts/gpu_run.py` renders two prompts on the real
   tokenizer before loading any cell and aborts if the header presence or the
   turn-invariance of prompt length disagrees with the flag.
8. **Distinct trajectories < N/4** flags low entropy, not lost sample size
   (reporting rule 3).
9. Turn-0 rows excluded from every 3m contrast.

---

## 8. Cost

Calibration from exp6 on an H100 at N=1000: LOGIT ≈ 8 min/model for 12 cells
(0.67 min/cell); SCRATCHPAD ≈ 30 min/model for 8 cells (3.75 min/cell). Group
overhead ≈ 2 min (process + vLLM init), first download per model ≈ 4 min.
Pricing at **$3/h** for an H100 80 GB.

| MODE | groups | cells | compute | overhead | wall | cost | +25% contingency |
|---|---|---|---|---|---|---|---|
| `nohist` | 3 | 30 | 20 min | 18 min | ~38 min | $1.90 | $2.40 |
| `abs` | 2 | 20 | 13 min | 12 min | ~25 min | $1.25 | $1.60 |
| `swap` | 2 | 20 | 13 min | 4 min | ~17 min | $0.85 | $1.10 |
| `both` (abs+nohist) | 5 | 50 | 33 min | 22 min | ~55 min | $2.75 | $3.45 |
| **`cross` (all four LOGIT conditions)** | **9** | **90** | **60 min** | **30 min** | **~1 h 30** | **$4.50** | **$5.60** |
| `pad` (nohist × CoT, 2 models) | 2 | 16 | 60 min | 12 min | ~1 h 12 | $3.60 | $4.50 |

**Recommended scope: `MODE=cross`, one session, ~1 h 30, ~$5.60 worst case.**
That buys both objections, the ceiling-proof `swap` control, and the
interaction cell, and leaves >$4 of a $10 budget for `MODE=pad` if the LOGIT
result is ambiguous or a reviewer insists on the CoT regime the literature
uses. Full scope `cross` + `pad` is ~2 h 45 and ~$10.10 at worst case, ~$8.10
at the point estimate — run them as two sessions, with `pad` conditional.

Storage: ~450 MB per uncompressed database, evicted per model; ~40 MB per
committed archive. Nine archives ≈ 360 MB, inside the pattern exp3–exp6 set.

---

## 9. If only one can be afforded: run `nohist`

**Objection 1 can lose and the paper survives.** If the effect turns out
lexical, C5 becomes a claim about token priming — a real finding this project is
already equipped to make, because D2 established exactly that for the container
effect from exp3's own data. The paper reframes around the ladder and the
labels-dominate result, both of which are already SUPPORTED.

**Objection 2 cannot lose that way.** It says the manipulation was never the
manipulation described. If it lands, every field-level sentence in C3, C5, C6
and C8 is mis-stated, and a single alternative account explains the entire
corpus including the asymmetry that is the headline.

Four further reasons, in order of weight:

1. **The abstract cell cannot separate H-state from H-conflict** (§5.1). Even a
   clean survival under X/Y leaves objection 2 fully open. The reverse is not
   true: `nohist` constrains H-conflict directly and, via `E_score` and
   `CPR(3)`, bounds what H-lex can be doing.
2. **It cannot be lost to an exclusion.** `nohist` runs under semantic labels
   where all three models are clean. The abstract condition has one model
   excluded before it starts and two more sitting near a ceiling.
3. **It is cheaper and covers all three models** — 3 groups against 2, one flag
   against a framing change with a known baseline shift.
4. **It buys two measurements nothing else in the project can produce**: arm 1
   as a genuine state-deprivation condition, and an arm-3 CPR that is not a copy
   task.

It is also the ACL reviewer's single required change and the AC's W1.
