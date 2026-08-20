# comprehension-defection

Code, data and analysis for *Presence and Content Cancel: A Matched-Placebo
Instrument for Ablating Agent State Components*.

**The claim.** Agent frameworks license a memory or state component by ablation:
run the agent with the component, run it without, report the difference. That
contrast is a **sum of two channels** — the *presence and form* of the component,
and the *content* it carries — and the two need not agree in sign. So a null
contrast does not identify an inert component. A token- and density-matched
placebo arm bounds each channel separately.

In the headline cell — Qwen2.5-7B-Instruct, chain-of-thought readout, semantic
action labels, unconditional cooperator — the two channels are **+0.1934** and
**−0.2136** in episode-level defection rate. They sum to **−0.0202**, an interval
reaching to within 5×10⁻⁵ of zero, which is exactly what an ordinary
presence/absence ablation reports and would be read as an inert component.

Three results do not depend on the models, the task or the opponents: the
decomposition is a telescoping identity; observing its sum fixes neither the sign
nor the magnitude of either part; and one contrast — the single-field pair
3s against 3m — cancels the configuration channel without assumption. Everything
else in the repository is a measurement, named per model and per configuration.

**What the instrument measured.** The state is *available*, *readable*
(comprehension probe rate 1.000 in the treatment arm) and *preferentially read*
over the raw history — and defection still does not move toward the
opponent-conditional optimum. Reading a field is not the same as using it:
30,000 of 30,000 own-score probes reproduce a **falsified** score while behaviour
moves by at most 3.2 percentage points. Within the block's content, the
**opponent's last move** dominates the cumulative score in six of six cells, and
survives deleting the transcript entirely. A pre-registered test of whether the
repaired estimate transports across serialisation template, field order and
insertion position **rejects additivity in six of six model × opponent groups**;
the registered verdict is correspondingly narrow.

The full ladder of measurements and their status is in
[`CLAIMS.md`](CLAIMS.md); what was actually run is in
[`EXPERIMENTS.md`](EXPERIMENTS.md). Three experiments carry pre-registrations
frozen before their data — exp1 in [`PREREGISTRATION.md`](PREREGISTRATION.md),
exp7 in [`PREREGISTRATION_EXP7.md`](PREREGISTRATION_EXP7.md) and exp8 in
[`PREREGISTRATION_EXP8.md`](PREREGISTRATION_EXP8.md); exp2 through exp6 are
exploratory. All eight have run. exp8's six declared deviations are recorded in
`EXPERIMENTS.md`. **On the relationship between the register and the manuscript,
precisely:** every *measurement* the manuscript reports has a `CLAIMS.md` entry
with a status of CONFIRMATORY or SUPPORTED, written before or during the run it
describes. The manuscript's *identification result* — the decomposition,
Corollary 1 and Proposition 1 — was derived in the paper and entered the register
afterwards, on 20 Aug 2026, as `CLAIMS.md` §A0, which says so at the top of the
section. The register did not govern the decomposition, and this file used to
imply it did. `scripts/reproduce.sh` regenerates every
quoted interval from the committed archives on CPU alone, in seven steps ending
with the interval-provenance check; see [`DATA.md`](DATA.md) for the
artefact-by-artefact provenance and the bootstrap streams it distinguishes.

---

## Models

Three open-weight instruct models, bf16, run under vLLM on rented GPUs. Exact
Hugging Face identifiers, as invoked by the experiment drivers in `scripts/`:

| short name | Hugging Face id | parity target | filler token |
|---|---|---|---|
| llama | `meta-llama/Llama-3.1-8B-Instruct` | 34 | `'\n'` (id 198) |
| qwen | `Qwen/Qwen2.5-7B-Instruct` | 39 | `'\n'` (id 198) |
| mistral | `mistralai/Mistral-7B-Instruct-v0.3` | 45 | `' '` (id 29473) |

The parity target is the token count every injected `[STATE]` block is padded
to. It is **derived from the tokenizer at construction**, not configured
(`ScaffoldConfig.treatment_block_tokens = 0` means AUTO), because the correct
value is a property of the tokenizer and not of the study. The measured values
above are recorded in `EXPERIMENTS.md` (exp3, *Parity constants*) and re-derived
by `scripts/tokenizer_check.py`. Mistral has no single-token newline, which is
why `filler_candidates` is a list.

These targets are exp2-onward. **exp1 ran at a 32-token block**, before the
parity work; that difference is the subject of `CLAIMS.md` B1 and is why exp1's
estimate is reported as confounded rather than as a result.

`meta-llama/Llama-3.1-8B-Instruct` is **gated**; you need an accepted licence
and an `HF_TOKEN` to download it. See [`MODEL_LICENSES.md`](MODEL_LICENSES.md).

## Data

Every run's database is committed to this repository as a gzipped SQLite file
next to the code, one per cell group, named after its `run_id`:

```
sweep.sqlite.gz             1   exp1, pre-registered; confounded, see CLAIMS.md B1
exp2_*.sqlite.gz            4   mechanism + second model, density fix
exp3_*.sqlite.gz            9   full factorial
exp4_*.sqlite.gz           12   chain-of-thought ablation
exp5_*.sqlite.gz            3   minimal-CoT salience control
exp6_*.sqlite.gz            6   field-level falsification
exp7_*.sqlite.gz            9   lexical priming and history removal, pre-registered
exp8_*.sqlite.gz           16   cross-configuration stability, pre-registered
smoke_*.sqlite.gz           9   N=4 instrument checks, never a measurement
cotsmoke_*.sqlite.gz       12   N=4 chain-of-thought instrument checks
                           --
                           81   archives
```

`.gitignore` records the archives as **already in git history**; they total
**1.59 GB compressed** and expand to about 19.8 GB (see `MANIFEST.md`); they are
deliberately not ignored, because ignoring them would do nothing to files
already tracked and rewriting history would invalidate every
`run_meta.git_commit` recorded across the corpus. Uncompressed databases
(`*.sqlite`) are ignored and are regenerated by `gunzip -k`.

Schema, column semantics and the column × experiment availability matrix are in
[`DATA.md`](DATA.md). The schema itself, with the reasoning for each design
decision, is `cdx/db.py`.

The literal file inventory, per-file SHA-256 digests and per-experiment turn
and episode counts are in [`MANIFEST.md`](MANIFEST.md), counted directly from
the archives rather than derived from the drivers in `scripts/`: **81 archives,
1.59 GB compressed, 12,700,960 turns, 635,048 episodes.** exp3's "180,000
episodes, 3.6M decisions" reconciles exactly; exp6's turn count was quoted as
1.28M and is 1.2M, now corrected in `EXPERIMENTS.md`.

## Install

Analysis and the test suite are CPU-only and need Python 3.10+:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

numpy is not optional. `analysis/13` exits at import without it, and
`analysis/08` and `analysis/11` silently fall back to a **different RNG stream**
at the same seed — point estimates reproduce, interval endpoints do not. See
`DATA.md`, *Bootstrap provenance*.

Re-running inference needs a GPU and a different, pinned stack — see
[`requirements-gpu.txt`](requirements-gpu.txt). The measured effects are **not
invariant to that stack** (`CLAIMS.md` B5: the block-vs-no-block contrast moved
4 percentage points on identical inputs across a vLLM/torch/transformers
change), so the pins there are load-bearing, not cosmetic.

## Reproduce

One command, from the repository root, after `pip install -r requirements.txt`:

```bash
bash scripts/reproduce.sh
```

It decompresses each `*.sqlite.gz`, re-estimates every contrast at the episode
level with a 10,000-resample bootstrap, rebuilds the evidence and
cross-experiment tables from SQL, runs the exp6 field decomposition and the exp8
cross-configuration study, and finishes with the interval-provenance check. It
writes `ep_<stem>.json`, `EVIDENCE.md`, `EVIDENCE_cells.csv`,
`CROSS_EXPERIMENT.md`, `EXP6_FIELDS.json`, `exp8_stability.json` and
`exp8_logodds.json`. Nothing it runs writes to a database; every read is
`mode=ro&immutable=1`.

Decompressing the corpus needs **~21.4 GB free**. The script checks before it
starts and, if there is not enough, prints three targeted commands that run in
seconds under 400 MB.

```bash
RUN_OPTIONAL=1 bash scripts/reproduce.sh
```

additionally runs the diagnostics and side analyses (`analysis/01`, `03`, `04`,
`08`, `09`, `10`, `11`, `12`, `14`), which write `DECOMPOSITION.md` /
`DECOMPOSITION.json`, `STRATIFIED_DONOR.md` / `STRATIFIED_DONOR.json`,
`DOSE_RESPONSE.md`, `SWAP_RESCORE.md`, `EXP6_PREREQUISITES.md` and
`REVIEWER_RESPONSES.json`. **None of them feeds a headline number** — that is
what "optional" means here, and they are run rather than listed so it cannot be
confused with "forgotten".

The two JSON sidecars are the files the manuscript's interval-provenance
appendix cites by path; the markdown beside each is rendered from its sidecar,
so the two cannot disagree. `analysis/08` takes a `--turn-filter`: `excl_t0` is
the paper's declared rule and the default; `donor_matched` is coordinate-level
**post-treatment selection** and is a sensitivity basis only. Every sidecar
records which was used in `_turn_filter`; check that field before quoting.

`analysis/17_interval_provenance.py` reads the LaTeX source. `paper/*` is
gitignored, so it **skips on a clone** rather than failing; point `--tex` at a
checkout of the manuscript to run it.

**One known staleness.** `CROSS_EXPERIMENT.md` in this repository was generated
before `analysis/07` was corrected to reconstruct values on the episode grid
before rounding, so a handful of its four-decimal figures are a digit low on
exact ties (`−0.1049` where the value is `−0.1050`). The generator is fixed; the
file regenerates correctly on the next `reproduce.sh` run. `EVIDENCE.md` is
unaffected — `analysis/06` does no four-decimal rendering.

Which claim is checked by which command, against which file, with which
expected value, is in [`CLAIM_MAP.md`](CLAIM_MAP.md). **`CLAIM_MAP.md` currently
covers exp1–exp6 only; exp7 and exp8 have entries in `CLAIMS.md` and
`EXPERIMENTS.md` but no command rows yet.** Their artefacts are
`EXP7_FIELDS.json`, `exp8_stability.json` and `exp8_logodds.json`, regenerated
by `analysis/14`, `15` and `16`.


## How to test

Three layers, cheapest first. Each answers a different question.

**1. Is the instrument intact?** (seconds, no data)

```bash
python -m pytest tests/ -q                      # 550 cases, 188 functions, 13 files
python -m pytest tests/ -q --collect-only | tail -1   # recount; do not trust the number above
```

These are gates, not unit tests. Each protects a property the causal estimate
depends on, so **a failure is a finding, not a chore**. The ones that matter most:

| if this fails | what is no longer true |
|---|---|
| `test_token_parity_is_exact_for_every_placebo_arm` | the placebo is not width-matched; every contrast using it is confounded |
| `test_field_falsification.py` | arms 3s/3m change more than one line, so Proposition 1's cancellation no longer holds |
| `test_engine_is_bit_identical_across_runs` | the engine is non-deterministic; nothing below is reproducible |
| `test_seeds_are_stable_across_processes` | seeds are not a pure function of coordinates; resume can alter a trajectory |
| `test_exp1_to_exp5_unchanged.py` | a later experiment's arms moved the parity target and broke byte-identical reproduction of the earlier ones |

**2. Does a single claim hold?** (minutes, one database)

Every claim id in `CLAIMS.md` has a row in `CLAIM_MAP.md` giving the database,
the exact command and the value to expect. Check one without touching the corpus:

```bash
gunzip -k exp4_qwen_sem_scratchpad.sqlite.gz
python analysis/02_episode_level.py --db exp4_qwen_sem_scratchpad.sqlite \
       --out ep_exp4_qwen_sem_scratchpad.json --bootstrap 10000
jq '.contrasts["3b_minus_1|allc"], .contrasts["ate_true|allc"]' ep_exp4_qwen_sem_scratchpad.json
```

Expect `+0.1934` and `−0.2136`. **A disagreement at the fourth decimal is a
finding, not a rounding difference** — see `DATA.md`, *Bootstrap provenance*, for
the two RNG streams and the two turn-0 conventions that make artefacts disagree
legitimately, and for which one any given number must come from.

Two checks need no corpus at all and run in seconds:

```bash
python analysis/18_additivity_q.py                    # Cochran's Q from exp8_logodds.json
python analysis/18_additivity_q.py --variance halfwidth-max   # the asymmetry sensitivity
```

**3. Does everything hold?** (hours, ~21.4 GB)

```bash
bash scripts/reproduce.sh                   # the critical path
RUN_OPTIONAL=1 bash scripts/reproduce.sh    # plus every diagnostic
```

`STEP 6` skips unless you point `--tex` at a checkout of the manuscript, because
`paper/*` is gitignored. That skip is expected.

## Verify the instrument

```bash
python -m pytest tests/ -q      # 550 passed
python scripts/smoke_test.py    # end-to-end on the stub backend, no GPU
```

188 test functions across 13 files; **550 test cases** once parametrisation is
expanded. That count moves as the suite grows — recount rather than trusting it:
`python -m pytest tests/ -q --collect-only | tail -1`. They are instrument gates,
not unit tests: each one protects a property the causal estimate depends on. A
sample:

| test | protects against |
|---|---|
| `test_optimal_play_matches_hand_computed_values` | drift in the DP that underwrites the sign-flip prediction (TFT: optimal 62, ALLD 24, regret 38) |
| `test_sign_flip_directions_are_opposite` | loss of the core robustness property |
| `test_dp_agrees_with_live_engine` | divergence between the DP's opponent reimplementation and the live engine |
| `test_token_parity_is_exact_for_every_placebo_arm` | a parity violation silently invalidating the causal estimate |
| `test_no_single_token_filler_available_is_fatal` | padding that cannot hit an exact target |
| `test_undisclosed_horizon_is_not_leaked_into_the_prompt` | leaking the horizon and changing the equilibrium |
| `test_seeds_are_stable_across_processes` | regression to Python's salted `hash()` |
| `test_engine_is_bit_identical_across_runs` | non-determinism in the engine layer |
| `test_resume_reproduces_uninterrupted_run` | crash recovery altering the experiment |
| `test_exp1_to_exp5_unchanged.py` (8 tests) | exp6's new arms changing the parity target and breaking byte-identical reproduction of exp1–exp5 |
| `test_guided_still_names_the_horizon` | silently "fixing" the exp4 scratchpad prompt that exp4's databases were produced by |
| `tests/test_field_falsification.py` | arms 3s/3m altering more than the one line they are supposed to |
| `tests/test_no_history.py`, `tests/test_abstract_falsification.py` | exp7's factors changing anything exp1–exp6 rendered |

`scripts/smoke_test.py` exercises the engine, parity, persistence, resume and
the analysis path against `DummyBackend`. Its numbers are meaningless by
construction and it says so.

## Layout

```
cdx/                the engine. No language model scores, adjudicates or
                    terminates a game.
  config.py         every experimental parameter, typed, hashed onto every row
  seeding.py        hash-derived seeds
  game.py           IPD, scripted opponents, horizon modes
  optimal.py        exact DP; source of the pre-registered direction predictions
  scaffold.py       [STATE] block templates, token-ID parity, field falsification
  probe.py          the frozen probe suite and its hash
  donor.py          arm 3c donor selection
  runner.py         episode loop, persistence, resume
  db.py             SQLite schema, heavily commented
  backends.py       LLMBackend protocol, DummyBackend, CharTokenizer
  backends_vllm.py  the production GPU backend
  backends_mlx.py   Apple-silicon backend, development only
analysis/           seventeen scripts, numbered 01-04 and 06-18 (there is no 05;
                    the numbering is historical). All read-only over the
                    databases, all open them mode=ro&immutable=1. 15 and 16
                    produce the exp8 cross-configuration study; 17 traces every
                    interval the manuscript prints back to the artefact, path
                    and field that generate it; 18 runs the additivity test.
                    17 and 18 are the only two that read no database.
scripts/            seven experiment drivers, exp2_mechanism.sh through
                    exp8_templates.sh; the GPU runner (gpu_run.py);
                    reproduce.sh; and the CPU instrument checks
                    (smoke_test.py, tokenizer_check.py, calibrate_block.py,
                    pilot.py, preregister.py, density_check.py)
tests/              the instrument gates
docs/historical/    superseded operator runbooks, kept for provenance
```

### The arms

| arm | block | what it isolates |
|---|---|---|
| 1 | none | the no-component baseline |
| 3 | the true state | the agent's actual component |
| 3b | width- and density-matched, carrying no score for either player, no opponent move and no round count | presence and form, with content removed |
| 3c | another episode's state, rendered whole | veracity against schema |
| 3d | structure without language | markup against language |
| 3s | the true block with **own score** falsified | one field, at full dose |
| 3m | the true block with **opponent's last move** falsified | one field, at full dose |

3s against 3m is the pair that cancels the configuration channel without
assumption. Everything else bounds a channel rather than isolating it.

> Note: `--no-history` is **not** captured by `config_fingerprint`
> (see `DATA.md` §9).

Note on backends: `cdx/backends_vllm.VLLMBackend` is the implemented production
backend and is what `scripts/gpu_run.py` imports. A same-named **stub** still
sits in `cdx/backends.py` and raises `NotImplementedError`; it predates the GPU
work, nothing imports it, and it is left in place only because `cdx/` is frozen
against the databases' `git_commit` provenance. Do not use it and do not read
its docstring as current.


## Use this on your own agent

The result is not about this game. It is about any ablation of any context
component: **drop the component and you measure a sum, not an effect.** If your
evaluation reports "we removed the memory/scratchpad/retrieved-context and
accuracy moved by X", X is `Π + ATE_true` and you cannot sign either part.

The instrument is three arms and five requirements. The requirements are not
optional; each is stated in the paper because this project's own pipeline
violated it and got a confounded estimate.

**Build the arms.**

- [ ] **Arm 1** — your agent with the component absent. Your existing baseline.
- [ ] **Arm 3** — your agent with the real component. Your existing treatment.
- [ ] **Arm 3b** — a placebo block, same width and density as arm 3, carrying
      **none of the decision-relevant content**. This is the arm you are missing,
      and it is the whole instrument. `cdx/scaffold.py` builds ours.
- [ ] Optional but cheap: **3s / 3m**, the real block with exactly one field
      falsified. This pair is the only contrast that cancels the configuration
      channel without assumption (Proposition 1), so it is the strongest thing
      here if you can afford two more arms.

**Then satisfy R1–R5.** Section 3 of the paper states each; `tests/` gates each.

- [ ] **R1 — parity on token IDs, not characters.** Tokenise both blocks and pad
      by appending raw token **IDs**, feeding `prompt_token_ids`. Never re-run
      the tokeniser over the padded result: a BPE merge can silently change a
      count you just asserted. `" 12"` and `"100"` differ in token count under
      byte-level BPE. Derive the target from *your* tokeniser — ours are 34, 39
      and 45 for three 7–8B models, and they are properties of the vocabulary,
      not of the study. → `scripts/tokenizer_check.py`
- [ ] **R2 — one single-token filler, chosen per tokeniser.** You need a filler
      that is exactly one token in your vocabulary or you cannot hit an exact
      target. Newline works for Llama and Qwen; Mistral has no single-token
      newline, which is why the config takes a *list* of candidates and searches.
      → `test_no_single_token_filler_available_is_fatal`
- [ ] **R3 — enforce a density floor, not just token parity.** A placebo at exact
      parity can still be mostly whitespace, and then your contrast varies
      decision-relevance *and* dense-text-versus-whitespace at once. Ours was 47%
      content against a 100%-content treatment and produced a large, significant,
      confounded estimate. Assert a content floor per block type and a
      cross-template tolerance, **at construction**. → `scripts/density_check.py`
- [ ] **R4 — verify falsification by self-join against recorded ground truth**,
      not against your own bookkeeping. Store what you displayed *and* what was
      true, then join. Ours asserts the falsehood on 19,000 of 19,000 rows and
      leaves the other field untouched on 0 of 19,000. Checking a builder against
      its own intent proves nothing.
- [ ] **R5 — cluster at the level of the episode, not the turn.** Turns inside an
      episode are dependent. Turn-level intervals misstated width by 0.46× to
      4.07× across our exp2–exp5. Bootstrap over episodes.

**Then read the result correctly.**

- [ ] Report **Π and ATE_true separately**, never only their sum. A near-zero sum
      is the signature the paper is about, not evidence of an inert component.
- [ ] **Name the configuration.** Additivity is rejected in six of six of our
      groups (`analysis/18_additivity_q.py`), so the decomposition does **not**
      transport across serialisation template, field order or insertion position.
      Measure it per setting; do not assume it. One matched arm makes that cheap.
- [ ] **Do not read comprehension as use.** Our agents reproduce a falsified
      score on 30,000 of 30,000 probes while behaviour moves at most 3.2 points.
      A probe that shows the model *read* the field says nothing about whether it
      *used* it.

**What this instrument does not give you.** It bounds two channels; it does not
recover the component's causal effect. That would additionally require the
content channel to be configuration-independent, and §5.5 reports it is not.

## Design notes

**Seeds are derived, not stored.**
`seed = sha256("run_id:episode_id:arm:model_id:readout:opponent")`, first 16 hex
digits, masked to 63 bits (SQLite's `INTEGER` is signed). Each episode's
randomness is a pure function of its coordinates, so execution order is
irrelevant and resume cannot change any trajectory. This is what makes cheap
preemptible instances safe.

**Token parity is enforced at token-ID level.** Placebo blocks are padded by
appending raw token IDs and fed as `prompt_token_ids`. The tokenizer is never
re-run over the result, so no BPE merge can change a count that was just
asserted. Character padding cannot deliver parity: `" 12"` and `"100"` differ in
token count under byte-level BPE (`CLAIMS.md` B2).

**Token parity is necessary and not sufficient.** exp1's placebo was 15 content
tokens padded with 17 filler into a 32-token block — **47% content** — against a
treatment block of 32 content tokens and **no filler at all, 100%**. It produced
a large, highly significant, and confounded estimate. Density must match too
(`CLAIMS.md` B1). From exp2 the parity target is 34 tokens and the treatment
carries 2 filler tokens, 94%.

**The engine is the only source of truth.** No language model scores, adjudicates
or terminates a game.

**Reproducibility is stratified.** The engine is bit-identical (gated by
`test_engine_is_bit_identical_across_runs`). Logit-readout decisions are
deterministic outside a measured noise band — `logit_gap` is logged on every
decision so the fragile share can be reported rather than assumed away.
Scratchpad decisions get statistical equivalence only. vLLM is not
bit-deterministic and a bit-identity gate over it would block the project
forever.

**`PREREGISTRATION.md` is frozen.** Its probe-suite hash
`12c9a10d970099c1e56fc00b77d17f8ab00e14ff6a084290f27066583106a689` is stamped on
every database row, so a later edit to any probe question is detectable from the
data alone. It must never be regenerated; `scripts/preregister.py` exists for the
record of how it was produced, not to be re-run.

**Defects found in this project's own pipeline are kept visible**, not quietly
fixed: a zero-padded numeric field the model read as data, a placebo that was
47% content against a treatment carrying no filler at all, and a probe scorer
grading in the wrong label space. Each was corrected on frozen data. The
manuscript's appendix lists what this project withdrew alongside what survived.

## Licence

- **Code** (`cdx/`, `analysis/`, `scripts/`, `tests/`): MIT. See [`LICENSE`](LICENSE).
- **Data** (`*.sqlite.gz`) and **documentation**: CC BY 4.0, with the model-output
  conditions in [`MODEL_LICENSES.md`](MODEL_LICENSES.md).
- **Model weights**: not distributed here. Llama-3.1-8B-Instruct is gated under
  the Llama 3.1 Community License; Qwen2.5-7B-Instruct and
  Mistral-7B-Instruct-v0.3 are Apache-2.0. The released databases contain
  Llama-3.1 generations, so **Built with Llama** applies. Details and obligations:
  [`MODEL_LICENSES.md`](MODEL_LICENSES.md).

## Citation

See [`CITATION.cff`](CITATION.cff), or:

> Gattu, U. (2026). *Presence and Content Cancel: A Matched-Placebo Instrument
> for Ablating Agent State Components*.
> https://github.com/UdayGattu/comprehension-defection

## Contact

Uday Gattu — udaygattu007@gmail.com
