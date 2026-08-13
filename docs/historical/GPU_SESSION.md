> # SUPERSEDED — HISTORICAL RECORD, DO NOT FOLLOW
>
> This was the operator runbook for the **first** paid GPU session, written
> before any real inference had been run. It is kept for provenance: it records
> the budget reasoning and the power table that the exp1 design was sized
> against.
>
> **It is not a current instruction sheet.** Six GPU runs have happened since;
> the experiment drivers in `scripts/` (`exp2_mechanism.sh`, `exp3_full.sh`,
> `exp4_cot.sh`, `exp5_minimal_cot.sh`, `exp6_fields.sh`) supersede every
> procedure below, and `scripts/gpu_run.py` has gained flags this file does not
> mention.
>
> **Two edits were made when this file was moved to `docs/historical/`:**
>
> 1. The pre-flight checklist instructed the operator to run
>    `python scripts/preregister.py > PREREGISTRATION.md && git commit`.
>    That would **overwrite the frozen pre-registration**, whose probe-suite
>    hash `12c9a10d…` is stamped on every row of every database in the corpus.
>    Regenerating it changes the timestamp, breaks the "committed before any
>    inference was run" property that gives the file its evidential value, and —
>    if any probe wording were ever edited — would silently erase the only
>    detector for that edit. The instruction has been removed and replaced with
>    the warning below. It was correct exactly once, in August 2026, before
>    exp1.
> 2. Nothing else. The rest is verbatim, including estimates that later data
>    superseded.
>
> Current entry points: [`README.md`](../../README.md),
> [`scripts/reproduce.sh`](../../scripts/reproduce.sh),
> [`CLAIM_MAP.md`](../../CLAIM_MAP.md).

---

# Paid GPU session — runbook (exp1, historical)

Budget: ~$10. On Runpod A100 80GB (~$1.90/hr) that is **5 hours**, not one.
Lambda A100 40GB (~$1.29/hr) is ~7.5 hours. You have more runway than you think.

## Before you start the instance

- [ ] **`PREREGISTRATION.md` is FROZEN. Do not regenerate it.** Its hash is
      recorded on every database row across exp1–exp6. `scripts/preregister.py`
      is the record of how it was produced, not a command to re-run. If you
      believe it needs to change, that is a protocol deviation and must be
      reported as one in `EXPERIMENTS.md`, not committed over the original.
- [ ] push the repo somewhere you can `git clone` from the box
- [ ] have your HF token ready (Llama-3.1 is gated)

## On the box

```bash
git clone <your repo> && cd comprehension-defection
pip install vllm transformers
huggingface-cli login
tmux new -s run                      # survives your ssh dropping

python scripts/gpu_run.py --verify   # ~5 min: instrument check + 2 episodes
```

**Read STEP 2 before anything else.** If the action tokens do not match the
framing, stop and fix — every downstream number is meaningless. This is the check
that caught a 100% off-task rate on the laptop.

## Then measure, then commit

```bash
python scripts/gpu_run.py --episodes 100 --budget-minutes 10
```

Watch the per-turn throughput line. That number replaces every estimate in the
spec. Compute what N the remaining budget buys, then run it:

```bash
python scripts/gpu_run.py --episodes <N> --budget-minutes <M> --out sweep.json
```

| N per cell | MDE @ 80% power |
|---|---|
| 200 | 0.140 |
| 500 | 0.088 |
| 1000 | 0.062 |
| 1600 | 0.050 |

Your laptop signal was **ATE_true = −0.130 (tft), +0.020 (allc)**. The TFT effect
needs ~N=200 to resolve; the ALLC effect is near zero and may need far more, or
may not be real. Prefer the largest N you can afford over more cells.

## Do not

- run without `--budget-minutes`
- leave the instance up after the run — **terminate it yourself**
- treat `--verify` output as data (n=2)
- put 4-bit laptop numbers in the paper

## Success looks like

Instrument verified, a measured throughput figure, and one properly-powered
primary contrast. That is enough to decide whether the sign flip is real.

---

> **What actually happened.** exp1 ran at N=1,600 on an A100-80GB and the
> pre-registered sign-flip was **rejected** (ATE_true +0.052 tft / +0.042 allc,
> both p < 1e-4 — same sign, wrong direction), with the manipulation gate
> failing at CPR 0.24–0.31. The estimate was later shown to be **confounded**
> by a placebo density mismatch, not a false positive. The laptop signal quoted
> above (−0.130 / +0.020) did not survive contact with real inference and is
> exactly why the "do not put 4-bit laptop numbers in the paper" rule is on this
> page. See `EXPERIMENTS.md`, Experiment 1, and `CLAIMS.md` A1 and B1.
