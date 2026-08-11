# comprehension-defection

P1 engine for *How Much of Reported LLM Defection Is Comprehension Failure? A Placebo-Controlled Estimate.*

Runs entirely on a laptop. No GPU, no model weights, no API keys. Build and verify here before renting anything.

## Quick start

```bash
python3 -m pytest -q tests/      # 18 instrument gates
python3 scripts/smoke_test.py    # end-to-end on the dummy backend
```

Then, the one open empirical question in the spec:

```bash
pip install transformers
python3 scripts/tokenizer_check.py
```

## What each gate protects

| Test | Protects against |
|---|---|
| `test_optimal_play_matches_hand_computed_values` | Drift in the DP that underwrites the sign-flip prediction (TFT: optimal 62, ALLD 24, regret 38) |
| `test_sign_flip_directions_are_opposite` | Loss of the core robustness property |
| `test_dp_agrees_with_live_engine` | Divergence between the DP's opponent reimplementation and the live engine |
| `test_token_parity_is_exact_for_every_placebo_arm` | A parity violation silently invalidating the causal estimate |
| `test_multi_token_filler_is_rejected` | Padding that cannot hit an exact target |
| `test_undisclosed_horizon_is_not_leaked_into_the_prompt` | Leaking the horizon and changing the equilibrium |
| `test_seeds_are_stable_across_processes` | Regression to Python's salted `hash()` |
| `test_engine_is_bit_identical_across_runs` | Non-determinism in the engine layer |
| `test_resume_reproduces_uninterrupted_run` | Crash recovery altering the experiment |
| `test_stochastic_horizon_rejects_gamma_below_threshold` | An ill-posed Phase 2 where cooperation is unsustainable |

## Design notes

**Seeds are derived, not stored.** `seed = sha256(run_id:episode_id:arm:model:readout:opponent)`. Each episode's randomness is a pure function of its coordinates, so execution order is irrelevant and resume cannot change any trajectory. This is what makes cheap preemptible instances safe.

**Token parity is enforced at token-ID level.** Placebo blocks are padded by appending raw token IDs and fed via `prompt_token_ids`. The tokenizer is never re-run, so no BPE merge can change a count that was just asserted. Truncation is never used — it would remove structural tokens.

**Every `[STATE]` field is fixed-width.** Without this the treatment block's length varies by turn, and a stale-state donor can render longer than the treatment it must match, making parity unachievable.

**The engine is the only source of truth.** No language model scores, adjudicates, or terminates a game.

**Reproducibility is stratified.** The engine is bit-identical. Logit-readout decisions are deterministic outside a measured noise band — `logit_gap` is logged on every decision so the fragile share can be reported rather than assumed away. Scratchpad decisions get statistical equivalence only. vLLM is not bit-deterministic and a bit-identity gate would block P1 forever.

## Not implemented on purpose

`VLLMBackend` raises `NotImplementedError`. It must be written against the measured behaviour of the installed vLLM version during the one-hour paid calibration session, not against assumptions. Use `DummyBackend` for all laptop work — it exercises the full pipeline including the sign-flip analysis, and proves the plumbing, never the hypothesis.

## Before spending money

1. `pytest` green
2. `tokenizer_check.py` run and its result recorded in the methods section
3. Probe text committed and hashed (`ExperimentConfig.probe_text_hash`) — required before the P1b probe pass
4. SIGKILL resume verified on real hardware, not just in the test

Then rent one hour on an 80GB card, implement `VLLMBackend`, measure real throughput, and replace the spec's 25–40 A100-hour estimate with a measured figure.
