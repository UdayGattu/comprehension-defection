# Pre-registration — Experiment 8

**Status: frozen before data. Not yet run.**
Driver `scripts/exp8_templates.sh` · probe suite unchanged
(`12c9a10d970099c1e56fc00b77d17f8ab00e14ff6a084290f27066583106a689`).

**Correction, 2026-08-15, made before any exp8 inference and before this file
was committed.** §3 originally read *"no shared content word"*. That is false as
written: lowercasing and splitting the two label sets gives an intersection of
exactly `{your, round}`. The claim is corrected in place below, with the
measurement that produced it. It is recorded here rather than edited silently
because a pre-registration whose text can move without a visible note is not a
pre-registration. No exp8 data existed when this correction was made.

exp1 and exp7 are pre-registered. exp8 is registered for a stronger reason than
either: it is the experiment that decides whether the paper's central
contribution is a method or an anecdote. An exploratory answer to *"does your
instrument work on more than one prompt?"* is worth nothing at all.

Predictions below are committed before any exp8 inference runs. **§5 contains a
single binding aggregation rule.** `PREREGISTRATION_EXP7.md` §6 condition (b)
said *">50% in ≥2 of 3 models"* without stating whether to aggregate over
opponents; the two readings disagree, and that ambiguity is now a declared
defect of the paper. It is not repeated. Everything about how cells combine into
a verdict — the unit, the opponent rule, the tie-break, the floor rule, the
exclusion arithmetic — is fixed here, in advance, in one place.

---

## 1. The claim under test

The paper has been reframed as a **methods** contribution:

> Prompt ablations are uninterpretable without a token- and density-matched
> placebo. Here is the instrument, and here is what it measures that an
> unmatched ablation cannot.

Two referees independently made the same top objection. **A method that has only
ever been run on one prompt template is not an instrument.** All seven
experiments — ~300,000 episodes — used:

* **one `[STATE]` template** — `Your score` / `Opponent score` /
  `Opponent's last move` / `Rounds played`;
* **one field order** — that one;
* **one insertion position** — `insertion_index = 1`, after the rules.

The repository already argues that this is unsafe, in its own words:

* `cdx/config.py`, on `insertion_index`: *"lost-in-the-middle effects produce
  >30% swings from position alone, which would masquerade as a treatment
  effect."* No driver in exp1–exp7 ever set it to anything but `1`. The only
  non-default use anywhere in the repository is `tests/test_no_history.py`.
* **CLAIMS.md D2**: a purely **lexical** change — `Cooperate/Defect` → `X/Y` —
  **reverses the sign** of the container effect in Qwen. From this project's own
  exp3 data.

So the headline (**C5**: the last-move field dominates, the cumulative score does
not) has been measured at exactly one point in a three-dimensional space in which
this project has already documented a sign reversal. exp8 measures it at more
than one point.

---

## 2. Hypotheses

| tag | account | what it predicts for exp8 |
|---|---|---|
| **H-field** | the C5 reading. Behaviour conditions on the *information* the last-move field carries, which is the entire input to optimal play against TFT. | the asymmetry survives a different vocabulary, a different field order and a different position, within a factor of two |
| **H-surface** | the asymmetry is a property of the exact strings `Opponent's last move` / `Defect`, not of the field's role. Direct descendant of D2. | the asymmetry collapses or reverses under `reworded` |
| **H-serial** | the asymmetry is a **position** effect. The last-move line is the third of four in the block and the block sits directly above `[HISTORY]`; the score line is first and furthest from the decision point. | the asymmetry collapses or reverses under `permuted` order (which moves the score to last and the last move to first), and/or under `insertion_index = 2` |

H-surface and H-serial are not mutually exclusive, and either landing is a
methods result rather than a failure: it would mean *"a matched-placebo ablation
gives a template-specific answer, and here is how much"*, which is a stronger
statement about instruments than a clean survival. What would be fatal is not
knowing.

---

## 3. Design

`{1, 3b, 3, 3s, 3m}` × `{tft, allc}` × **N = 1000** × **LOGIT** × **semantic**
framing, `[HISTORY]` present, in a fraction of a 2³ over:

| factor | − | + |
|---|---|---|
| **T** template | `original` | `reworded` |
| **O** order | canonical | permuted |
| **P** position | `insertion_index = 1` | `insertion_index = 2` |

**T.** `Your cumulative points` / `Their cumulative points` /
`Their previous choice` / `Rounds elapsed`. Same four fields, same four values,
and **no shared content word except `Your` and `Rounds`**. Measured, not
asserted: the intersection of the two lowercased label word-sets is exactly
`{your, round}`, and it is pinned by
`tests/test_template_family.py::test_the_two_label_families_share_exactly_two_words`.
Every other word differs — `score`/`points`, `Opponent`/`Their`,
`last move`/`previous choice`, `played`/`elapsed`. Each template carries its own
density-matched placebo bodies and its own parity target (§7 gate 1).

**O.** `(score, opp score, last move, rounds)` → `(last move, rounds, opp score,
score)`. Both falsifiable fields move as far as a four-field block allows: the
score first→last, the last move third→first. This is the only permutation that
can separate *"the last-move **field** dominates"* from *"the **third line**
dominates"*.

**P.** After the rules and before `[HISTORY]`, versus after `[HISTORY]` and
before the instruction.

Abstract framing and `--no-history` are **exp7's** factors and are deliberately
not crossed here. The anchor cell must be exp6's condition exactly, or the
design has two differences in it.

### 3.1 The fraction

Full crossing is 8 conditions × 10 cells × 3 models = **240 cells**, ≈ 3 h 40,
≈ **$11** on an H100 at $3/h. It is **not** dollar-limited at a $40 budget, and
this pre-registration says so rather than inventing a constraint. The fraction is
chosen for two other reasons:

1. **Multiplicity.** 8 conditions × 3 models × 2 opponents = 48 stability
   verdicts on one claim. The falsification rule in §5 is a **worst-case rule
   over conditions** — it fires if *any* condition leaves the band. A worst-case
   rule over 8 conditions is a strictly harsher test than the same rule over 4,
   for reasons that have nothing to do with the instrument. Halving the
   condition count halves the number of chances a null instrument has to look
   broken.
2. **The spare budget buys more elsewhere.** The freed ≈$15 is pre-committed to
   `MODE=pad`: a SCRATCHPAD replication, which is the regime the prior
   literature actually runs in and the regime **C7** says attenuates the effect
   by 50–87%. No number of additional LOGIT conditions substitutes for that.

**The design is a 2^(3−1) half fraction replicated on all three models, plus its
foldover on the primary model.**

**Half A**, defining relation **I = −TOP**:

| cond | T | O | P | template | index |
|---|---|---|---|---|---|
| `anchor` | − | − | − | `original` | 1 |
| `origpermp2` | − | + | + | `original_permuted` | 2 |
| `rewordp2` | + | − | + | `reworded` | 2 |
| `rewordpermp1` | + | + | − | `reworded_permuted` | 1 |

**Half B** (the foldover), defining relation **I = +TOP**:

| cond | T | O | P | template | index |
|---|---|---|---|---|---|
| `origp2` | − | − | + | `original` | 2 |
| `origpermp1` | − | + | − | `original_permuted` | 1 |
| `rewordp1` | + | − | − | `reworded` | 1 |
| `rewordpermp2` | + | + | + | `reworded_permuted` | 2 |

**Why Half A is the half we keep.** It contains `(−,−,−)`, which is exp6's
condition exactly — same arms, same opponents, same N, same readout, same
labels, same code path. That makes the anchor a **within-session control**
rather than a comparison to a database produced by a different vLLM image. **B5**
measured perturbation contrasts moving 4 pp across stacks, and the
block-vs-no-block contrast is the stack-fragile one. The anchor is therefore
**re-run, not inherited**: 3 groups, 20 minutes, and it removes the only
between-run difference in the design.

**What the half costs, stated plainly.** In a 2^(3−1) resolution-III design each
main effect is aliased with the complementary two-factor interaction:

```
T ≡ −OP        O ≡ −TP        P ≡ −TO
```

So on **llama** and **mistral**, *"the template moved the asymmetry"* and
*"order and position interact"* are the same number and cannot be told apart.
The **foldover on qwen** breaks every one of those aliases — a foldover of a
resolution-III design always does — giving the **full 2³ on qwen**, with every
main effect and every two-factor interaction clear.

**The licensing assumption, stated:** the three-factor interaction **TOP is
negligible**. This is the standard assumption for a half fraction and it is
checkable after the fact on qwen, where the full 2³ estimates it directly. If
|TOP| on qwen exceeds the stability band, the half fractions on llama and
mistral are **uninterpretable as main effects** and §5's rule falls back to its
condition-wise worst-case form, which does not depend on the factorial
decomposition at all. That fallback is why the primary falsification rule is
written condition-wise rather than factorially.

**Why qwen is primary, fixed before the run.** Three pre-existing measured
facts: the largest last-move effect in the corpus (`E_move = −0.4049` vs TFT);
the cleanest off-task record under semantic LOGIT; and — decisively — it is the
model in which **D2** found a purely lexical change *reversing* the sign of the
container effect. If any model's asymmetry is going to move under a reworded
template, the prior says it is this one. Spending the de-aliasing budget
anywhere else would spend it on the model least likely to need it.

### 3.2 Scope by MODE

| MODE | conditions | models | groups | cells |
|---|---|---|---|---|
| `anchor` | `anchor` | all 3 | 3 | 30 |
| `half` | Half A | all 3 | 12 | 120 |
| `fold` | Half B | qwen | 4 | 40 |
| **`screen`** | **Half A + Half B on qwen** | **all 3 / qwen** | **16** | **160** |
| `full` | Half A + Half B | all 3 | 24 | 240 |
| `pad` | `anchor`, `rewordpermp2` (SCRATCHPAD) | all 3 | 6 | 60 |

`full` is **conditional** and pre-committed: it runs only if qwen's 2³ shows a
two-factor interaction on `A` outside the stability band — i.e. only if the
aliases on llama and mistral are known to matter. Running it unconditionally
would be the full factorial and the fraction would have bought nothing.

---

## 4. Estimands

Episode-level bootstrap, **10,000 resamples, seed 20260814**,
`analysis/02_episode_level.py`. Turn-level intervals are never quoted (**B3**).
**Turn 0 is excluded from every arm of every move contrast** — at turn 0 there
is no last move, arm 3m is byte-identical to arm 3, and those rows carry no
manipulation (`donor_degenerate = 1`).

For a *contrast cell* = (model `m`, condition `c`, opponent `o`):

```
E_move (m,c,o) = P(D | 3,  m,c,o) − P(D | 3m, m,c,o)
E_score(m,c,o) = P(D | 3,  m,c,o) − P(D | 3s, m,c,o)

A(m,c,o)  =  E_move − E_score  =  P(D | 3m, m,c,o) − P(D | 3s, m,c,o)
S(m,c,o)  =  A(m,c,o) / A(m, anchor, o)
```

### Why `A` and not the pair `(E_move, E_score)`

`A` is **the primary estimand and the only one a falsification rule reads.**

Arms 3s and 3m are **byte-identical except for one line**, padded to the same
parity target, at the same insertion position, in the same template, in the same
prompt. Every between-condition nuisance — block width, template verbosity,
field order, position — is common to both arms and **cancels exactly** in their
difference. Arm 3 cancels out of `A` algebraically.

This is what makes a cross-template comparison legitimate *despite* the two
templates having different parity targets and therefore different prompt widths.
A cross-template comparison of a raw defection rate, or of `E_move` alone, would
not be legitimate, because arm 3's own level moves with the template. **No
falsification condition in §5 reads a raw rate or a single-arm contrast across
templates.** `E_move` and `E_score` are reported per condition for continuity
with exp6/exp7 and are descriptive only.

### Secondary, descriptive

* `R(m,c,o) = |E_move| / |E_score|` — exp6's ratio form. Reported; **never used
  in a verdict**, because it is a ratio of two quantities either of which can be
  near zero.
* Log-odds form of `A`, `A_lo = logit P(D|3m) − logit P(D|3s)` — reported for
  every cell, and the **only** form quoted for COMPRESSED cells (§5.4).
* `CPR(3)` per condition — the readability check. A template whose block is not
  read is not a template test.
* Factorial main effects on `A`, per model and per opponent:
  `ΔT = mean A over T+ − mean A over T−`, and likewise `ΔO`, `ΔP`; plus the
  two-factor interactions on qwen. On llama and mistral in `MODE=screen` these
  must be quoted as `"T + (−OP)"`, never as `"T"`.

---

## 5. THE AGGREGATION RULE, AND THE FALSIFICATION CONDITIONS

This section is binding. Where §6 or any later document appears to disagree with
it, this section governs.

### 5.1 The unit

The unit is the **contrast cell**: one (model, condition, opponent). There are at
most 8 × 3 × 2 = 48 of them; `MODE=screen` produces 32.

### 5.2 Opponents are never pooled and never averaged

> **Every rule below is evaluated independently at `opponent = tft` and at
> `opponent = allc`. Every verdict is reported as an ordered pair
> (TFT verdict, ALLC verdict).**
>
> **Where a single verdict is required — and only there — TFT is decisive and
> ALLC is reported without a vote.**

Reason, fixed in advance: under ALLC the opponent cooperates unconditionally, so
the true last move is **constant** and the last-move field is not diagnostic for
optimal play. **C8** already measured the content effect as smaller vs ALLC in
**8 of 9** cells, and exp6 measured arm 3c's falsification dose at **0.0000** vs
ALLC. Counting ALLC as an independent vote would let a mechanically attenuated
cell outvote the cell the claim is about, and pooling would average a real effect
with a near-null and halve it by construction.

**This is the exp7 defect being repaired.** `PREREGISTRATION_EXP7.md` §6(b) —
*"shrinks by more than 50% toward 0 in ≥2 of 3 models"* — admits both "≥2 models
on at least one opponent" and "≥2 models on the opponent-averaged estimate", and
those two readings can disagree on the same data. exp8 has one reading.

### 5.3 Models are counted, not averaged; and the denominator moves

* *"k of N models"* means **k distinct model identifiers** whose per-model rule
  evaluates TRUE **on the decisive opponent alone**. Model-level estimates are
  never averaged into a pooled estimate.
* A model **excluded** by a pre-run or in-run gate (§7) is removed from **both**
  numerator and denominator. The threshold is then a **strict majority of the
  surviving models**: 2 of 3 → 2 of 2 → 1 of 1.
* **If exactly two models survive and they disagree (1 of 2), the result is
  recorded `SPLIT` and the verdict is `UNRESOLVED`. It is never rounded to a
  majority in either direction.**
* **If only one model survives, no falsification or support verdict may be
  issued at all.** The run is recorded `INCONCLUSIVE` and C5 keeps the scope
  clause it already has.

### 5.4 Floors, ceilings, and what they may vote on

Computed **per contrast cell from that cell's own data**, over arms {3, 3s, 3m}:

| label | definition | may vote on F1 | may vote on F2 | counted in denominator |
|---|---|---|---|---|
| **FLOOR** | `max P(D)` over {3, 3s, 3m} ≤ 0.05 | no | **no** | **no** |
| **CEILING** | `min P(D)` over {3, 3s, 3m} ≥ 0.95 | no | **no** | **no** |
| **COMPRESSED** | not FLOOR/CEILING, but `min P(D)` ≤ 0.15 or `max P(D)` ≥ 0.85 | **no** | **yes** | yes |
| **CLEAN** | otherwise | yes | yes | yes |

* A FLOOR or CEILING cell is **reported in full** and **casts no vote in either
  direction**. It cannot falsify and it cannot support. mistral under semantic
  labels is the expected case — exp7 found 99.8% of its episodes never defect.
* A COMPRESSED cell has a mechanically compressed risk difference, so it is
  ineligible for the magnitude rule **F1** but eligible for the sign rule **F2**:
  a sign reversal survives any monotone compression of the scale, which is
  exactly what a ceiling is. This is the same reasoning exp7 used to justify its
  `swap` phase.
* **If every non-anchor condition of a model is FLOOR or CEILING on the decisive
  opponent, that model is treated as excluded** and §5.3's denominator rule
  applies.

**The anchor guard.** `S` is a ratio and its denominator is measured.

> **`S` is never computed when `|A(m, anchor, o)| < 0.05`.**

If the anchor cell for a (model, opponent) is FLOOR/CEILING, or if
`|A(anchor)| < 0.05`, then `S` is undefined for that model and opponent and the
**model is excluded for that opponent** under §5.3. A ratio to a near-zero
denominator manufactures arbitrarily large apparent instability, and this is the
specific arithmetic trap the rule exists to close. In that case the cell's `A`
values are still reported, as differences, with the note `S undefined`.

### 5.5 Conditions are aggregated worst-case, never averaged

> A model is **UNSTABLE** if **at least one** of its non-anchor, vote-eligible
> conditions violates the band. Conditions within a model are **not averaged.**

Reason: the claim under test is *"the instrument returns the same answer under
any of these renderings"*. An average over conditions would let six stable
conditions hide one that reverses — which is precisely the failure **D2** found
in Qwen, where one lexical change flipped a sign that survived everything else.

There is no tie to break: a worst-case rule over a finite set of conditions is
total.

### 5.6 The stability band

```
S ∈ [0.50, 2.00]   with the same sign as A(anchor)
```

* **0.50** is inherited deliberately from exp7's own ">50% shrink" criterion, so
  the two pre-registrations use one threshold rather than two.
* **2.00** is its reciprocal. The band is symmetric on a log scale, because
  "the effect tripled under a reworded template" is exactly as much of an
  instrument failure as "it vanished".
* **Resolution check.** exp6's episode-level CI half-widths were ±0.011 (qwen) to
  ±0.014 (llama), so a difference of two independent contrasts resolves at
  ≈ 0.016 → ≈0.023 for a difference of differences. Against
  `A(qwen, anchor, tft) ≈ 0.40`, the band's half-width is ±0.20 — roughly **9×**
  the resolution. The test is not resolution-limited; a band violation will be a
  real one.

### 5.7 Falsification conditions

Let **C-INST** be the claim exp8 exists to test:

> *The field asymmetry this instrument measures is a property of the manipulated
> content, and is reproduced within a factor of two under a different field
> vocabulary, a different field order and a different insertion position.*

**F1 — MAGNITUDE (binding).**
C-INST is **FALSIFIED** if, at `opponent = tft`, in **≥ a strict majority of the
surviving models** (§5.3), there exists at least one non-anchor **CLEAN**
condition `c` satisfying **both**:

* **(i)** `S(m,c,tft) < 0.50` **or** `S(m,c,tft) > 2.00`; **and**
* **(ii)** the 95% episode-bootstrap CI on `A(m,c,tft) − A(m,anchor,tft)`
  **excludes 0**.

Both are required. (i) alone is a point estimate; (ii) alone is a statistically
resolvable but practically irrelevant wobble.

**F2 — SIGN (binding, ceiling-proof).**
C-INST is **FALSIFIED** if, at `opponent = tft`, in ≥ a strict majority of the
surviving models, there exists at least one non-anchor **CLEAN or COMPRESSED**
condition `c` with

* `sign A(m,c,tft) ≠ sign A(m,anchor,tft)`, **and**
* the 95% CIs on `A(m,c,tft)` and on `A(m,anchor,tft)` **both exclude 0**.

F2 is not a subset of F1: it applies to COMPRESSED cells, which F1 excludes, and
it does not require a magnitude threshold. A sign reversal survives any monotone
compression, so it is the statement that can still be made at a ceiling. **This
is the outcome D2 already produced once, from this project's own data, under a
smaller manipulation than `reworded`.**

**F3 — READABILITY (binding, and it is a validity condition, not a result).**
The exp8 result is **VOID**, not falsified and not supported, if
`CPR(3, m, c, o) < 0.80` in any condition whose cells are otherwise used in a
verdict, while `CPR(3, m, anchor, o) ≥ 0.95`. A block the model cannot read is
not a rendering of the state; it is a broken prompt, and a "collapse" there
measures the renderer rather than the model. Such a condition is dropped from
every rule above, reported in full, and the driver's manipulation gate is
re-examined before anything else is said.

**SUPPORT.**
C-INST is **SUPPORTED** if, at `opponent = tft`, in ≥ a strict majority of the
surviving models, **every** non-anchor vote-eligible condition has
`S ∈ [0.50, 2.00]` with the anchor's sign, **and** `max(|ΔT|, |ΔO|, |ΔP|)` on
that model is `< 0.50 × |A(anchor)|`.

**F1/F2 and SUPPORT are mutually exclusive by construction** with 3 or 2
surviving models: both require a strict majority, and two disjoint strict
majorities cannot exist in a set of size ≤ 3. Where neither fires the verdict is
**PARTIAL**, and PARTIAL licenses only a scope clause (§6, row 4) — never a
generality claim.

### 5.8 Opponent disagreement

If F1 or F2 fires at ALLC but not at TFT, **C-INST is not falsified.** The
outcome is recorded as *opponent-moderated instability*, reported in the same
table, and a scope clause naming ALLC is added to the methods section. The
converse — firing at TFT but not ALLC — **does** falsify, because TFT is the
decisive opponent under §5.2 and ALLC does not vote.

### 5.9 What happens on falsification

C5's field-level sentence is **RETRACTED IN PLACE** under reporting rule 7 — the
original left standing, as exp5 retracted Qwen's state-comprehension effect —
and rewritten with the template, order and position it was measured under named
in the sentence. **The methods contribution survives either way**, and this is
the reason exp8 is worth running: the instrument's *value* is that it can detect
this, and a demonstration that a matched-placebo ablation is template-sensitive
by a measured amount is a stronger methods claim than an unfalsified assertion
that it is not.

---

## 6. What each result licenses

| # | result (TFT, decisive) | licensed claim | action |
|---|---|---|---|
| 1 | SUPPORT: every condition in band, all main effects < 0.5·\|A(anchor)\|, ≥2 surviving models | *The asymmetry is a property of the field's role, not of its wording, its rank in the block, or its distance from the decision point. Measured over 2 vocabularies × 2 orders × 2 positions.* | C5 upgraded; the methods section states the instrument's measured invariance band and the number of renderings it was measured over |
| 2 | **F2** fires under `reworded` (sign reversal), `origpermp1`/`origp2` in band | *The asymmetry is carried by the surface form of the field labels.* Converges with D2 from an independent manipulation. | C5 RETRACTED in place, rewritten as a surface-form claim; the methods section gains its most useful sentence — **a matched-placebo ablation is not portable across templates, and here is the measured size of that** |
| 3 | **F1** or **F2** fires under `permuted` order or `insertion_index = 2`, but `reworded` at position 1 is in band | *The asymmetry is a serial-position effect, not a field effect.* The block's third line, or its adjacency to `[HISTORY]`, is doing the work. | C5 RETRACTED in place; C3/C6/C8's mechanism sentences rewritten; position promoted from a nuisance to a reported factor everywhere in the paper |
| 4 | PARTIAL — one model unstable, others in band, or SPLIT under §5.3 | nothing general | verdict recorded verbatim (`PARTIAL` / `SPLIT` / `INCONCLUSIVE`); C5 keeps a scope clause naming the exact conditions tested; `MODE=full` is run if budget remains |
| 5 | qwen's 2³ shows \|TOP\| outside the band | the half fractions on llama and mistral are uninterpretable **as main effects** | §5's condition-wise rules still stand (they never used the factorial); `MODE=full` becomes required rather than conditional before any factor is named |
| 6 | anchor does not reproduce exp6 within its CI | nothing about templates at all | **STOP.** B5 stack drift. Re-run exp6's cell, or re-baseline every exp8 contrast against the in-session anchor and say so |
| 7 | `MODE=pad`: asymmetry stable under LOGIT but not under SCRATCHPAD | *The instrument is readout-dependent.* C7 already says the effect attenuates under CoT; this would say the attenuation is also template-dependent. | reported as a limitation with a measured size; labelled exploratory unless `MODE=pad` ran on ≥2 non-floored models |

---

## 7. Gates and abort criteria

Fixed here so no cell can be dropped after seeing its estimate.

1. **Parity-target isolation (CPU, pre-flight, and it is the load-bearing one).**
   Each template derives its **own** parity target from **its own** three block
   types. The reworded template's blocks are longer; if the derivation were ever
   a max over the registry, the original's target would rise, **every block in
   every arm of exp1–exp7 would get wider**, and the study's own exp3→exp4
   measurement says a prompt-width change of that kind moves a causal estimate
   by up to 0.04 against effects as small as 0.017. Pinned as a literal in
   `tests/test_template_family.py` (`ORIGINAL_CHAR_TARGET = 119`) and asserted
   again by the driver before any model is loaded.
2. **Fingerprint isolation (CPU, pre-flight).** The template is **not** a
   `ScaffoldConfig` field — it is a keyword argument to `ScaffoldBuilder`,
   exactly as the scratchpad variant and `include_history` are, and for exactly
   the documented reason (EXPERIMENTS.md, known defect 2): a field there would
   change `config_fingerprint` on every historical row. The default
   `ExperimentConfig` fingerprint is pinned as a literal. Provenance is carried
   by `run_id`, by `run_meta.config_json`, and by `turn_details.prompt_full`.
   `insertion_index` **is** a config field and **does** move the fingerprint —
   safely, because every historical row carries the default `1`.
3. **One-line falsification under both templates (CPU, pre-flight).** Arms 3s and
   3m must each change **exactly one rendered line** relative to arm 3, under
   every template and both framings, and the changed line must be the declared
   field. Under the permuted order the changed line must be at the **permuted**
   index — a permutation tuple that never reached the renderer would otherwise
   pass silently.
4. **Position (CPU, pre-flight, plus a real-tokeniser check in `gpu_run.py`).**
   `insertion_index = 2` must place the block after `[HISTORY]` and before the
   instruction, verified as an **exact token slice computed from the section
   lengths**, not searched for. `insertion_index = 2` with `--no-history` is
   refused before the model loads: there is no second seam and the block would
   land after the instruction suffix instead of failing.
5. **Manipulation gate (driver, per group, on stored prompts).** Four readings,
   all must agree: every declared label present; **no foreign template's label
   present**; label byte offsets inside `[STATE]` monotonically increasing in the
   declared order; `[STATE]` before/after `[HISTORY]` as `insertion_index`
   demands. A run that inherited the default template while its `run_id` said
   `reworded` is a clean replication of exp6 mislabelled as a generalisation
   test, and nothing downstream could tell.
6. **Parity gate (driver, per group, 100% of rows).**
   `COUNT(DISTINCT scaffold_tokens)` over block arms must be **1**, and
   `COUNT(DISTINCT prompt_tokens)` per (opponent, turn) over block arms must be
   **1**. This is the paper's central property audited on every row rather than
   on the three episodes that store `prompt_full`.
7. **History gate (inherited from exp7).** `[HISTORY]` present in every stored
   prompt, and prompt width **grows** with the turn index in every cell. exp8
   never runs `--no-history`; a cell where width is turn-invariant means the
   history silently vanished.
8. **Falsification gate (inherited from exp6).** `displayed_opponent_last` must
   be populated for arms 3s/3m or the lying arms told the truth.
9. **Off-task gate:** > 0.10 in arms 1, 3b or 3 of a group excludes the group
   from causal claims (**B4**) and moves the denominator under §5.3. Reported in
   full, as `exp3_mistral_abs` was.
10. **CPR gate** applies to arms 1, 3b and 3 only. CPR scores against the *true*
    state, so arms 3s and 3m are expected at ≈0.000 — that is the manipulation
    working, not a failure (**C6** corollary). See also **F3**.
11. **Distinct trajectories < N/4** flags low entropy, not lost sample size
    (reporting rule 3).
12. **Turn-0 rows excluded from every 3m contrast**, and therefore from `A`.
13. **Anchor-first ordering.** The anchor group runs first for every model. If it
    fails any gate, the model's remaining conditions are not run — there would be
    no denominator for `S`.

---

## 8. Cost

Calibration, measured: **LOGIT ≈ 8 min/model for 12 cells at N=1000 on an H100**
= **0.667 min/cell**; SCRATCHPAD ≈ 30 min/model for 8 cells = 3.75 min/cell.
Group overhead ≈ **2 min** (process + vLLM init); first download per model
≈ **4 min**. Pricing at **$3/h** for an H100 80 GB. Each exp8 group is 5 arms ×
2 opponents = **10 cells** = 6.67 min of compute.

| MODE | groups | cells | compute | overhead | wall | cost | +25% |
|---|---|---|---|---|---|---|---|
| `anchor` | 3 | 30 | 20 min | 18 min | ~38 min | $1.90 | $2.40 |
| `half` | 12 | 120 | 80 min | 36 min | ~1 h 56 | $5.80 | $7.25 |
| `fold` (qwen) | 4 | 40 | 27 min | 12 min | ~39 min | $1.95 | $2.45 |
| **`screen` = half + fold** | **16** | **160** | **107 min** | **44 min** | **~2 h 31** | **$7.55** | **$9.45** |
| `full` (delta over `screen`) | +8 | +80 | 53 min | 16 min | ~1 h 09 | $3.45 | $4.30 |
| `pad` (SCRATCHPAD, 2 conds × 3 models) | 6 | 60 | 225 min | 24 min | ~4 h 09 | $12.45 | $15.55 |

**Recommended scope: `MODE=screen`, one session, ~2 h 31, $7.55 (~$9.45 worst
case).** That buys a stability verdict on all three models plus a full
de-aliased 2³ on the primary one.

**Full programme**, run as three sessions with the last two conditional:

```
screen            $7.55        always
full  (delta)     $3.45        only if qwen's 2³ shows a 2fi outside the band
pad               $12.45       only if the LOGIT verdict is SUPPORT or PARTIAL
                 -------
                  $23.45  point estimate      $29.30 at +25% contingency
```

Inside a **$40** budget with ≈$10 of headroom. For comparison, the **full
factorial** — all 8 conditions on all 3 models — is 24 groups, 240 cells,
~3 h 40, **$11.00** ($13.75 at +25%); it is affordable, and §3.1 says why it is
still not what should be run.

Storage: ~450 MB per uncompressed database, evicted per model; ~40 MB per
committed archive. 16 archives ≈ 640 MB, inside the pattern exp3–exp7 set.

---

## 9. If only one MODE can be afforded: run `half`

**`half` is the only mode that produces a verdict on all three models**, and the
claim under test is a claim about the **instrument**, not about qwen. A foldover
on one model with no half on the others answers *"which factor moved it"* for a
question nobody has yet established has an answer.

**It also contains the anchor**, which is the only cell in the design capable of
invalidating every other cell. If exp6's condition does not reproduce in this
session, every difference measured is **stack drift wearing a template costume**
(B5: 4 pp), and `MODE=fold` would spend 39 minutes measuring it without ever
noticing.

`MODE=fold` is worth running second and only second. Its entire value is
de-aliasing, and there is nothing to de-alias until the half fraction has shown
that something moved.

**A note on what failure would mean.** exp7 could lose objection 1 and the paper
would survive by reframing. exp8 is different in kind: if the asymmetry is
template-dependent, the *methods* contribution does not weaken — it sharpens.
"Prompt ablations need a matched placebo" becomes "prompt ablations need a
matched placebo **and** their conclusions do not port across templates, by this
measured amount, and here is the instrument that shows it." The only outcome
this design cannot recover from is not running it, and continuing to report a
one-template measurement as a method.
