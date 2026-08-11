# Experiment registry

Every run that produced data, what it tested, what changed since the previous
one, and what it found. Runs are **append-only**: nothing here is edited to
match a later result, and no run is deleted because a better one exists.

`PREREGISTRATION.md` is Experiment 1's and is never modified.

| run | tag / commit | status | models | cells |
|---|---|---|---|---|
| exp1 | `exp1-frozen` · data `823313e` · code `8aad06f` | **pre-registered, closed** | Llama | 6 |
| exp2 | `f880b9a` `2574f3c` `c282959` · code `29e3aa7`, `2192dd9` | exploratory | Llama, Qwen | 28 |
| exp3 | *pending* | exploratory, confirmatory for exp2 | Llama, Qwen, Mistral | 90 |

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
   zero. 49.7% of treatment score-probe failures are attributable to that one
   format spec. Measured comprehension was therefore ~30% when the model could
   in fact read the block perfectly.
2. **Placebo density mismatch.** Arm 3b was two content lines padded with ~19
   blank ones — 44% content against a treatment that was 94% text. Token parity
   held; the stimuli were not comparable.

Consequence: **exp1's ATE_true of +0.042 (p < 1e-4) is a false positive**
produced by the placebo, not by the treatment. exp2 measures −0.012 (p = 0.24)
for the same contrast with a density-matched placebo.

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
  inferred indirectly from Qwen's 3c `own_score` dropping 0.400 → 0.200

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

### Record per model when the run completes

Both differ by tokenizer and belong in the methods section:

| model | parity target | filler token |
|---|---|---|
| Llama-3.1-8B | 34 | `'\n'` (id 198) |
| Qwen2.5-7B | | |
| Mistral-7B-v0.3 | | |

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
   understate width by 1.4×–2.8×.
5. Every defect above is reported. The exp1 → exp2 placebo comparison is the
   strongest methodological result in the project and depends on both runs
   being on the record.