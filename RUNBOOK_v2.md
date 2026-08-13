# MOVED — see `docs/historical/RUNBOOK_v2.md`

This file has been retired to [`docs/historical/RUNBOOK_v2.md`](docs/historical/RUNBOOK_v2.md)
with a header marking it superseded.

It was the operator runbook written **between exp1 and exp2**, when the
zero-padded score field had just been diagnosed. Its central patch was
superseded before it was applied — character padding cannot deliver token parity
at all, so what shipped from exp2 onward was natural number rendering plus
parity enforced at token-ID level.

Two defects were stripped when it was moved:

1. **An absolute personal path** on line 3 (`/Users/<user>/Documents/…`).
   Removed. Nothing in this repository should record where the author's home
   directory is.
2. **A citation of `test_treatment_block_token_count_is_constant`**, which does
   not exist and never did — while telling the operator it "is the one that
   matters" and to stop if it failed. The historical copy names the real gates:
   `test_target_holds_for_every_state`,
   `test_target_is_derived_from_the_tokenizer_not_the_config`,
   `test_score_field_is_not_zero_padded`,
   `test_no_character_padding_anywhere_in_the_block` (all in
   `tests/test_score_field_parity.py`) and
   `test_token_parity_is_exact_for_every_placebo_arm` (in `tests/test_engine.py`).

Current entry points:

- [`README.md`](README.md) — what this is, how to install, one reproduce command
- [`scripts/reproduce.sh`](scripts/reproduce.sh) — the reproduce command
- [`CLAIM_MAP.md`](CLAIM_MAP.md) — claim -> database -> command -> expected value
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — every run that produced data, append-only
- [`CLAIMS.md`](CLAIMS.md) — what may be said about it

<!--
  This is a tombstone, not content. If you would rather the path simply not
  exist, delete it:  git rm RUNBOOK_v2.md
  The full historical text is preserved at docs/historical/RUNBOOK_v2.md.
-->
