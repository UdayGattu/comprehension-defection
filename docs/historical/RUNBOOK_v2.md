> # SUPERSEDED — HISTORICAL RECORD, DO NOT FOLLOW
>
> This was the operator runbook written **between exp1 and exp2**, when the
> zero-padded score field had just been diagnosed and the question was how much
> GPU time the repair needed. It is kept for provenance: it records the decision
> gate that sent the project to exp2, and the reporting discipline
> (pre-registered / robustness / exploratory) that `CLAIMS.md` now enforces.
>
> **It is not a current instruction sheet.** Its Step 2 patch was superseded
> before it was applied: character padding cannot deliver token parity at all,
> so the fix was not `0` → `>` in the format specs but natural number rendering
> plus token-ID-level parity. exp2 through exp6 all ran on that. Steps 5b–5d
> describe runs that were folded into the exp2/exp3 factorial instead.
>
> **Three edits were made when this file was moved to `docs/historical/`:**
>
> 1. **An absolute personal path** was on line 3 (`/Users/<user>/Documents/…`).
>    Removed. Nothing in this repository should tell a reader where the author's
>    home directory is, and a path that only exists on one laptop is not
>    provenance.
> 2. **`test_treatment_block_token_count_is_constant` does not exist** and never
>    did. Step 3 told the operator it "is the one that matters" and to stop if it
>    failed — a test that always passes by never running is worse than no gate.
>    Replaced with the real gates, named from `tests/` as they stand.
> 3. **`git push` of `*.sqlite.gz` with `git add -f`** is left as written,
>    because that is how the archives entered history and `.gitignore` documents
>    why they stay there.
>
> Current entry points: [`README.md`](../../README.md),
> [`scripts/reproduce.sh`](../../scripts/reproduce.sh),
> [`CLAIM_MAP.md`](../../CLAIM_MAP.md),
> [`EXPERIMENTS.md`](../../EXPERIMENTS.md).

---

# Runbook v2 — post-sweep repair and extension (historical)

Run from the repository root. `sweep.sqlite` and `sweep.json` are the frozen
exp1 artefacts.

**Non-negotiable:** `sweep.sqlite` and `sweep.json` are frozen artefacts. Never
re-run into them, never edit them, never delete them. Everything below writes to
new files with new `run_id`s and is compared against the frozen data.

---

## Status

| Result | State |
|---|---|
| Perturbation effect (Arm 1 → 3b), −26.7 / −26.1 pp | **Confirmed**, episode-level, p < 1e-4 |
| Pre-registered sign-flip | **Rejected**, both signs positive, both significant |
| Arm 3 `opponent_last`, `rounds_played` | Restored to ~100% |
| Arm 3 `own_score` | ~14% outside turn 0 — zero-padding confound |
| Arms 3c, 3d | Never run |
| Second model | Never run |

The sign-flip test is **closed**. Nothing below re-opens it. The optimal policy
against TFT depends on the opponent's last action and the rounds remaining —
both delivered at ~100% — and not on the cumulative score, so the rejection
stands independent of the padding bug.

---

## STEP 1 — Scorer audit (free, no GPU)

```bash
python analysis/03_scorer_audit.py --db sweep.sqlite | tee analysis/out_03.txt
```

Resolves two open questions and **decides how much GPU time is needed**:

- the turn-16 spike (0.749 vs ~0.15 elsewhere)
- turn-0 `opponent_last` scoring differently across arms (1.000 / 0.216 / 0.000)

### Decision gate

| `Q3` dominant mechanism | Action |
|---|---|
| `off_by_padding` | **Scorer bug.** Fix `normalise()`, re-score offline from frozen data. **No GPU at all.** |
| `leading_zero` | **Format bug.** Proceed to Step 2. |
| `last_payoff` | **Model cannot sum.** A finding. Still do Step 2, to show space-padding doesn't rescue it. |
| `other_number` | Stop. Post the output before spending anything. |

Do not skip this gate. The cheapest outcome requires zero GPU spend.

---

## STEP 2 — Patch the scaffold

`cdx/scaffold.py`, in `ScaffoldBuilder.treatment_text` (~line 163).

**Before**

```python
return (
    "[STATE]\n"
    f"Your score: {state.agent_score:0{width}d}\n"
    f"Opponent score: {state.opponent_score:0{width}d}\n"
    f"Opponent's last move: {last_str}\n"
    f"Rounds played: {state.turn_index:0{width}d}\n"
)
```

**After** — the only change is `0` → `>` in three format specs

```python
return (
    "[STATE]\n"
    f"Your score: {state.agent_score:>{width}d}\n"
    f"Opponent score: {state.opponent_score:>{width}d}\n"
    f"Opponent's last move: {last_str}\n"
    f"Rounds played: {state.turn_index:>{width}d}\n"
)
```

Apply the same change in `nondiagnostic_text` (~line 182) so the placebo
template stays structurally matched.

Fixed character width is preserved, which is why the padding existed. Only the
pad character changes.

> **This step was never applied, and the reasoning above is wrong.** Space
> padding keeps *character* width constant and not *token* count: `" 12"` and
> `"100"` tokenise differently under byte-level BPE. Character padding cannot
> deliver token parity at all. What shipped instead, from exp2 onward, was
> natural number rendering (`12`, not `012`) with parity enforced at token-ID
> level against a target auto-derived per tokenizer. `ScaffoldConfig.
> score_field_width` survives only so old configs still load, and says so.
> See `CLAIMS.md` B2.

---

## STEP 3 — Prove the patch didn't break parity (free)

```bash
python -m pytest tests/ -q
python -m pytest tests/test_score_field_parity.py -v
python scripts/tokenizer_check.py --models meta-llama/Llama-3.1-8B-Instruct
```

The gates that matter, named as they actually exist in `tests/`:

- `test_target_holds_for_every_state` — sweeps every reachable state for every
  block-injecting arm and asserts the rendered block hits the parity target
  exactly. This is the one that catches a padding change silently altering token
  count.
- `test_target_is_derived_from_the_tokenizer_not_the_config` — the target is a
  property of the tokenizer, not a constant someone typed.
- `test_score_field_is_not_zero_padded` and
  `test_no_character_padding_anywhere_in_the_block` — the direct descendants of
  the exp1 defect.
- `test_token_parity_is_exact_for_every_placebo_arm` (in `tests/test_engine.py`)
  — the same property from the engine's side.

**If any of them fails:** do not proceed. A parity regression invalidates the
causal estimate silently, which is precisely what `TokenParityError` exists to
prevent.

---

## STEP 4 — Commit before renting anything

```bash
git add analysis/ tests/test_score_field_parity.py cdx/scaffold.py
git commit -m "Post-sweep: scorer audit, parity test"
git push
```

The GPU run records `git_commit` into `run_meta`. An uncommitted working tree
means the new data cannot be tied to the code that produced it.

---

## STEP 5 — GPU session (~30 min, ~$1)

One pod, three runs, then terminate. Same setup as last time.

```bash
# 5a  sanity, ~2 min
python scripts/gpu_run.py --verify

# 5b  clean treatment arm — robustness check on a REJECTED hypothesis
python scripts/gpu_run.py --episodes 1600 --arms 3 --opponents tft allc \
    --run-id arm3_spacepad --db arm3_spacepad.sqlite --out arm3_spacepad.json \
    --budget-minutes 12

# 5c  decompose the perturbation effect — EXPLORATORY, not pre-registered
python scripts/gpu_run.py --episodes 1600 --arms 3c 3d --opponents tft allc \
    --run-id perturb_decomp --db perturb_decomp.sqlite --out perturb_decomp.json \
    --budget-minutes 20

# 5d  model heterogeneity — EXPLORATORY
python scripts/gpu_run.py --episodes 1600 --arms 1 3b --opponents tft allc \
    --model Qwen/Qwen2.5-7B-Instruct \
    --run-id qwen_perturb --db qwen_perturb.sqlite --out qwen_perturb.json \
    --budget-minutes 15
```

Then, before terminating:

```bash
gzip -k arm3_spacepad.sqlite perturb_decomp.sqlite qwen_perturb.sqlite
git add -f *.sqlite.gz arm3_spacepad.json perturb_decomp.json qwen_perturb.json
git commit -m "Exploratory follow-ups: space-pad arm 3, 3c/3d decomposition, Qwen"
git push
```

**Terminate, do not Stop.** Stop keeps billing storage.

### What each run buys

| Run | Question |
|---|---|
| 5b | Was `own_score` comprehension suppressed by zero-padding, or can't the model sum? |
| 5c | Is the −26pp effect structure, position, content, or correctness? **Highest value — reviewers will demand it.** |
| 5d | Is the perturbation effect Llama-specific? Qwen defected 0% on the laptop. |

> **What actually happened.** None of 5b–5d ran under these `run_id`s. The three
> questions were folded into exp2 (`scripts/exp2_mechanism.sh`: arms 1/3/3b/3c/3d
> × Llama and Qwen, N=1,600) and exp3's full factorial. 5b's question was
> answered without a space-pad arm: with natural rendering, arm 3 CPR is 1.000
> on all 12,800 probes. 5d's answer is that the container effect **reverses
> sign** by model — Llama −21pp, Qwen +5.8pp — which is why `CLAIMS.md` D1
> requires the model to be named in every sentence about it.

---

## STEP 6 — Analysis

```bash
python analysis/02_episode_level.py --db arm3_spacepad.sqlite --out ep_arm3_spacepad.json
python analysis/02_episode_level.py --db perturb_decomp.sqlite --out ep_perturb.json
python analysis/02_episode_level.py --db qwen_perturb.sqlite  --out ep_qwen.json
python analysis/01_diagnose_arm3.py --db arm3_spacepad.sqlite
```

Compare `own_score` outside turn 0: **0.14 (zero-pad) vs. whatever space-pad
gives.** That difference is either a formatting-fragility result or evidence the
model genuinely cannot sum. Both are reportable; they are different papers'
worth of different.

> Note the `--out ep_<stem>.json` convention above. It is load-bearing —
> `analysis/06_evidence.py` looks for exactly that filename, and its own
> remediation hint omits the flag. `scripts/reproduce.sh` documents the trap and
> always passes it.

---

## Reporting discipline

Three labels, applied in the code, the `run_id`, and the manuscript:

1. **Pre-registered and closed** — sign-flip test on arms 1/3/3b. Rejected. Not revisited.
2. **Robustness check** — Arm 3 space-pad. Reported as a limitation of the original operationalisation.
3. **Exploratory** — 3c/3d, Qwen. Labelled exploratory throughout. No p-value from these enters an abstract as if pre-registered.

The one move that would make the paper indefensible is re-running Arm 3 clean
and then quietly re-testing the sign-flip on it. Do not do that.

> This section survived intact and is now enforced by `CLAIMS.md`'s status
> vocabulary (CONFIRMATORY / SUPPORTED / SUPPORTED, NARROW / REJECTED /
> RETRACTED / NOT TESTED / OVERCLAIM) and by `EXPERIMENTS.md`'s reporting rules.
