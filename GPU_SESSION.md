# Paid GPU session — runbook

Budget: ~$10. On Runpod A100 80GB (~$1.90/hr) that is **5 hours**, not one.
Lambda A100 40GB (~$1.29/hr) is ~7.5 hours. You have more runway than you think.

## Before you start the instance

- [ ] `python scripts/preregister.py > PREREGISTRATION.md && git commit` — worthless after data
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
