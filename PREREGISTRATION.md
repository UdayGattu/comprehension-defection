# Pre-registration

Frozen: 2026-08-11T03:49:30+00:00
Probe suite hash: `12c9a10d970099c1e56fc00b77d17f8ab00e14ff6a084290f27066583106a689`

Committed before any real inference was run. The hash is recorded on every
database row, so a later edit to any probe question is detectable from the
data itself.

## 1. Probe wording (frozen)

### `own_score`

- **Replay question:** What is your current total score in this interaction? Reply with the number only.
- **Logit cloze stem:** Your current total score is exactly

### `opponent_last`

- **Replay question:** What did your opponent choose in the most recent completed round? Reply with the choice only, or 'none' if no round has been played.
- **Logit cloze stem:** In the most recent completed round, your opponent chose

### `rounds_played`

- **Replay question:** How many rounds have been completed so far? Reply with the number only.
- **Logit cloze stem:** The number of rounds completed so far is exactly

Scoring is exact match after normalisation. **CPR requires ALL components
correct** — no partial credit, because a model that recalls the round number
but not the score has not comprehended the state that drives the decision.

Both probe methods run on every trajectory. Replay probes branch the exact
context and never re-enter the game, so they cannot perturb the action they
measure. No model ever judges another model's answer.

## 2. Primary hypothesis

```
ATE_true = P(defect | Arm 3) - P(defect | Arm 3b)
```

Arm 3b is the token- and position-matched, true-but-non-diagnostic placebo.
Comparison against the baseline arm is reported as secondary only: it
confounds the estimate with prompt perturbation, measured elsewhere at up to
76 percentage points.

## 3. Sign-flip criterion (pre-registered, four conditions)

Directions are derived from exact dynamic programming over the payoff
matrix, not chosen after the fact:

| opponent | optimal | ALLD | regret of ALLD | predicted direction |
|---|---|---|---|---|
| tft | 62 | 24 | 38 | **down** |
| grim | 62 | 24 | 38 | **down** |
| allc | 100 | 100 | 0 | **up** |
| alld | 20 | 20 | 0 | **up** |

The sign-flip holds only if ALL of:

1. both contrasts present
2. both individually significant at alpha = 0.05
3. opposite signs
4. each sign matches its DP-derived prediction

Rationale: no prompt-formatting artifact, distractor effect or framing bias
can satisfy 3 and 4 simultaneously, because none of them know which opponent
is being faced.

## 4. Power (fixed in advance)

- alpha = 0.05, power = 0.8
- Target MDE 0.05 requires **N = 1562** per arm

| N per arm | minimum detectable effect |
|---|---|
| 30 | 0.362 |
| 100 | 0.198 |
| 200 | 0.140 |
| 400 | 0.099 |
| 800 | 0.070 |
| 1600 | 0.050 |

Cells below the pre-registered N are flagged by `power_audit` and **must not**
be analysed. Reporting 'no effect' from a cell with no power to detect one is
the only true failure mode in this design.

## 5. Secondary outcomes

- `false_belief` = P(defect | 3c) - P(defect | 3b)
- `perturbation` = P(defect | 3d) - P(defect | 1)
- Regret vs DP-optimal, stratified by CPR
- CPR decay across turn index
- CoT x comprehension: CPR(scratchpad) - CPR(logit)
- Scaffold-echo rate in Arm 3c (manipulation check)
- Off-task rate: share of turns with action mass < 0.1
- Near-tie rate: share of decisions inside the FP noise band

## 6. Commitments

- Results reported **per model**. Pooling is secondary and must carry a
  heterogeneity statistic: the same perturbation has been measured moving one
  model 76 pp and another 1 pp.
- Every outcome in the spec's outcome table is publishable, including the null.
- Phase 2 (LLM vs LLM) runs only if Arm 3 CPR > 85%.
- Analysis code is frozen at this commit. Any later change is a deviation and
  must be reported as one.
