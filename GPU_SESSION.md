# MOVED — see `docs/historical/GPU_SESSION.md`

This file has been retired to [`docs/historical/GPU_SESSION.md`](docs/historical/GPU_SESSION.md)
with a header marking it superseded.

It was the operator runbook for the **first** paid GPU session, written before
any real inference had been run. Six GPU runs have happened since. Following it
today would at best waste money and at worst destroy evidence:

> Its pre-flight checklist instructed the operator to run
> `python scripts/preregister.py > PREREGISTRATION.md && git commit`. That would
> **overwrite the frozen pre-registration**, whose probe-suite hash
> `12c9a10d…` is stamped on every row of every database in the corpus. That
> instruction has been removed from the historical copy and replaced with a
> warning.

**`PREREGISTRATION.md` must never be regenerated.** `scripts/preregister.py` is
the record of how it was produced, not a command to re-run.

Current entry points:

- [`README.md`](README.md) — what this is, how to install, one reproduce command
- [`scripts/reproduce.sh`](scripts/reproduce.sh) — the reproduce command
- [`CLAIM_MAP.md`](CLAIM_MAP.md) — claim -> database -> command -> expected value
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — every run that produced data
- `scripts/exp2_mechanism.sh`, `exp3_full.sh`, `exp4_cot.sh`,
  `exp5_minimal_cot.sh`, `exp6_fields.sh` — the GPU drivers that supersede this
  file

<!--
  This is a tombstone, not content. If you would rather the path simply not
  exist, delete it:  git rm GPU_SESSION.md
  The full historical text is preserved at docs/historical/GPU_SESSION.md.
-->
