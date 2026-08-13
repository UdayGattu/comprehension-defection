# Data dictionary

Every table and column in the released databases, what it means, and which
experiments actually have it. The schema and its rationale live in
[`cdx/db.py`](cdx/db.py); this file extracts them and adds the availability
matrix, because **the schema drifted across experiments on purpose** and a
column being absent is not the same as a value being null.

- Format: SQLite 3, WAL journaling, one file per cell group, gzipped in the
  repository as `<run_id>.sqlite.gz`.
- Decompress with `gunzip -k <file>.sqlite.gz`, or run
  [`scripts/reproduce.sh`](scripts/reproduce.sh), which does it for you.
- Every analysis script opens the file read-only and immutable
  (`file:...?mode=ro&immutable=1`). The archives are the artefact of record and
  are never rewritten.
- WAL requires **local storage**. Opening a database on a network filesystem
  (Drive, FUSE, NFS) fails loudly by design — `Store._configure` raises rather
  than silently degrading crash-safety.

---

## 1. Unit of analysis

| level | what it is | where |
|---|---|---|
| cell | one (arm, opponent) combination inside one database | grouping key |
| episode | one 20-round game, independently seeded | `episodes` |
| turn | one round inside an episode | `turns` |

**The episode is the unit for every quoted interval.** Turns inside an episode
are not independent — an episode that defects on turn 4 is far more likely to
defect on turn 5 — and turn-level intervals understate width by 0.62x-3.75x
across exp2-exp5. `analysis/02_episode_level.py` is the only source of
intervals in this project.

The reverse error is also documented and also wrong: the distinct-trajectory
gate flags ~230 distinct trajectories in some placebo cells, and that is **not**
the effective N. Episodes are independently seeded, so a repeated trajectory is
a repeated *draw* from a low-entropy distribution, not a repeated observation.

---

## 2. Primary keys and coordinates

Both `episodes` and `turns` are keyed on the episode's full coordinate set:

```
(run_id, episode_id, arm, model_id, readout_mode, opponent_policy[, turn])
```

That tuple is also the **sole input to seed derivation**:

```
seed = int(sha256("run_id:episode_id:arm:model_id:readout:opponent")[:16], 16) & (2**63 - 1)
```

63 bits, not 64, because SQLite's `INTEGER` is signed and an unsigned 64-bit
seed overflows on insert. Nothing is carried between episodes, so execution
order is irrelevant and resume cannot alter a trajectory. Sub-streams within an
episode (donor selection, stochastic termination) get their own derived seed via
`derive_subseed(key, purpose)`, so a change in one component's consumption
pattern cannot shift every downstream draw.

Writes are committed **per episode**, in one transaction: ~55k commits instead
of ~1.1M, and a crash costs at most one episode, which the derived seed then
reproduces byte-identically.

---

## 3. Value domains

| field | values | notes |
|---|---|---|
| `arm` | `1`, `2`, `3`, `3b`, `3c`, `3d`, `3s`, `3m` | see the arm table below. `2` (`INLINE_PROBE`) is defined in `cdx/config.py` and was never run |
| `opponent_policy` | `tft`, `allc`, `alld`, `grim`, `qtable`, `llm` | only `tft` and `allc` appear in the corpus; `llm` raises in `cdx/game.py` |
| `readout_mode` | `logit`, `scratchpad` | an experimental factor, not an implementation detail |
| `framing` | `semantic` (Cooperate/Defect), `abstract` (X/Y) | plus a label-**swap** variant, recorded via `run_meta.config_json` / `argv`, not as a distinct `framing` value |
| `horizon_mode` | `known`, `undisclosed`, `stochastic` | **`known` everywhere in the corpus.** `stochastic` is implemented and was never run (`CLAIMS.md` E3/G1) |
| `agent_action`, `opponent_action`, `optimal_action` | `C`, `D` | `EXPERIMENTS.md` notes explicitly that in exp1 and exp2 these are the enum value `C`/`D`, not the long form |
| `horizon` | `20` | payoffs T=5, R=3, P=1, S=0 |

### Arms

| arm | name in `cdx/config.py` | block injected | asserts something false |
|---|---|---|---|
| `1` | `BASELINE` | none | — |
| `2` | `INLINE_PROBE` | — | never run |
| `3` | `TREATMENT` | true state | no |
| `3b` | `PLACEBO_NONDIAGNOSTIC` | true but non-diagnostic, token- and density-matched | no |
| `3c` | `PLACEBO_STALE` | another episode's whole block (donor) | **yes** |
| `3d` | `PLACEBO_SYNTACTIC` | structure without decision-relevant content | no |
| `3s` | `PLACEBO_SCORE` | true block, `Your score` offset by ±15 | **yes** |
| `3m` | `PLACEBO_MOVE` | true block, `Opponent's last move` flipped | **yes** |

`Arm.injects_block` and `Arm.falsifies_field` are the properties the writer uses.
Only the three falsifying arms get `displayed_opponent_last` written; elsewhere
the column is NULL, and **a NULL means "this arm told the truth", not "not
recorded"** — provided the database has the column at all. See the matrix in
section 8.

---

## 4. Table `episodes`

One row per completed episode.

| column | type | meaning |
|---|---|---|
| `run_id` | TEXT | the run tag; also the database stem |
| `episode_id` | INTEGER | index within the cell |
| `arm` | TEXT | see above |
| `model_id` | TEXT | Hugging Face id, e.g. `meta-llama/Llama-3.1-8B-Instruct` |
| `model_revision` | TEXT | HF revision; `main` unless pinned |
| `readout_mode` | TEXT | `logit` \| `scratchpad` |
| `opponent_policy` | TEXT | scripted opponent |
| `framing` | TEXT | `semantic` \| `abstract` |
| `horizon_mode` | TEXT | `known` throughout the corpus |
| `horizon` | INTEGER | 20 |
| `temperature` | REAL | decoding temperature (default 0.7) |
| `seed` | INTEGER | the derived seed, stored for convenience; it is a *function* of the coordinates, not an input |
| `config_fingerprint` | TEXT | stable hash of the whole `ExperimentConfig` |
| `prompt_hash` | TEXT | **see known defect 1 below — this is not a prompt digest** |
| `n_turns` | INTEGER | turns actually played |
| `agent_score` | INTEGER | cumulative agent payoff |
| `opponent_score` | INTEGER | cumulative opponent payoff |
| `defection_count` | INTEGER | numerator of the episode-level outcome |
| `episode_regret` | INTEGER | payoff shortfall against the DP optimum |
| `completed_at` | TEXT | ISO timestamp |

The episode-level outcome used everywhere is
`y_i = defection_count / n_turns`.

`seed`, `model_revision`, `temperature`, `config_fingerprint` and `prompt_hash`
are on **every row** deliberately: the paper's differentiator is that prior work
used closed models at provider-default temperatures, and omitting these fields
would forfeit it.

---

## 5. Table `turns`

One row per round. The hot table — kept narrow so it scans fast during analysis;
wide payloads live in `turn_details`.

| column | type | meaning |
|---|---|---|
| `turn` | INTEGER | 0-indexed round |
| `agent_action` | TEXT | `C` / `D` |
| `opponent_action` | TEXT | `C` / `D` |
| `agent_payoff` | INTEGER | this round's payoff to the agent |
| `optimal_action` | TEXT | the DP-optimal action in this state |
| `turn_regret` | INTEGER | shortfall against it |
| `logit_mass_c` | REAL | summed probability over all Cooperate surface forms |
| `logit_mass_d` | REAL | summed probability over all Defect surface forms |
| `action_mass_total` | REAL | `c + d` **before renormalisation**. This is the off-task detector: below 0.1 the model was not choosing an action at all |
| `logit_gap` | REAL | \|c - d\|. Logged on every decision so the reproducibility-fragile share can be *reported* rather than assumed away |
| `scaffold_tokens` | INTEGER | rendered block length in tokens |
| `scaffold_pad` | INTEGER | filler tokens appended to hit the parity target |
| `cpr_score` | INTEGER | 1 only if **all three** probe components are correct — no partial credit, per the pre-registration |
| `cpr_method` | TEXT | which probe method produced it (replay / logit cloze) |
| `scaffold_echo` | INTEGER | **never populated.** See below |
| `cpr_own_score` | INTEGER | per-component probe result |
| `cpr_opponent_last` | INTEGER | per-component probe result |
| `cpr_rounds_played` | INTEGER | per-component probe result |
| `turn_regret_calc` | INTEGER | regret recomputed at write time |
| `action_tokens_found` | INTEGER | how many action surface forms were present in the top-K. Makes logprob truncation visible instead of silent |
| `prompt_tokens` | INTEGER | assembled prompt length |
| `donor_agent_score` | INTEGER | **arm 3c / 3s.** The score the block actually **displayed** |
| `donor_degenerate` | INTEGER | 1 where no distinct donor existed |
| `displayed_opponent_last` | TEXT | **the last move the block asserted**, whatever the truth was |

### `donor_agent_score` and `donor_degenerate` (added in exp3)

Without `donor_agent_score` there is no offline way to answer the scaffold-echo
question: if a probe answer equals the donor's number rather than the true one,
the model demonstrably **read the block**. That is the difference between "did
not read it" and "read it and could not use it", and exp2 could only infer it
indirectly. `analysis/04_donor_echo.py` measures it directly and finds
DONOR_ECHO 0.94-1.00.

`donor_degenerate` marks turns where no distinct donor existed — every episode
at turn 0 has score 0 and no last move, so arm 3c is identical to arm 3 there
and contributes nothing. It is recorded **per row** rather than aggregated,
because the aggregate rate hides *which* turns. Every analysis that touches arm
3c excludes these rows, and this is not optional: including them manufactures
echo evidence out of rows that were never falsified.

### `displayed_opponent_last` (added in exp6)

This column closes the gap that made exp3's strongest mechanism an inference
rather than a measurement.

exp3 stored only `donor_agent_score`, so when arm 3c produced a 24pp swing in
qwen-vs-TFT the responsible field could not be identified. `analysis/12` later
showed the score falsification was far too small to explain it (sd = 3.3,
|d| >= 15 on 0.1% of rows against a slope of ~0.01/point), leaving the
opponent's last move as the only candidate — a claim resting on an ALLC/TFT
asymmetry rather than on a stored value.

With the column, the question is a `SELECT`: did the block say `Defect` when the
truth was `Cooperate`, and did the model then defect? `analysis/13` verifies it
by self-join against the previous turn's `opponent_action`, so the integrity
check cannot share a bug with the writer.

**This is why exp3's mechanism is an inference and exp6's is a measurement.**
Both are quoted in `CLAIMS.md` C3, with that distinction stated.

### `scaffold_echo` — a column that was never written

It exists in the schema from the start and **nothing ever populated it**
(`EXPERIMENTS.md`, exp2 Known gaps). Block-reading was inferred indirectly in
exp2, then measured properly in exp3 via `donor_agent_score`. Treat the column
as absent; do not read `NULL` here as evidence of anything.

---

## 6. Table `turn_details`

Wide and rare payloads, split out so `turns` stays fast to scan.

| column | type | meaning |
|---|---|---|
| `top_tokens` | TEXT (JSON) | top-K `[token, probability]` pairs; shows what the model wanted when it went off-task |
| `scratchpad` | TEXT | the model's reasoning text. SCRATCHPAD readout only |
| `probe_answers` | TEXT (JSON) | **raw model text per probe**. Without this a CPR of 0 cannot be diagnosed after the fact — and it is what `analysis/03`, `analysis/10` and exp6's echo test all read |
| `prompt_preview` | TEXT | first + last 300 characters of the assembled prompt |
| `prompt_full` | TEXT | **the complete decoded prompt**, stored only for the first `--full-prompt-episodes` (default 3) episodes of each cell |

`prompt_full` exists because `prompt_preview` truncates the middle — which is
exactly where the `[STATE]` block sits once the history grows, so on later turns
the preview cannot show whether the block rendered at all. Storing every prompt
would add gigabytes; a bounded sample costs almost nothing and answers the
question. exp1's zero-padding defect (`Your score: 012`) and its placebo density
mismatch were both invisible in the templates and visible here.

A row is written only if at least one of these five fields is non-null.

---

## 7. Table `run_meta`

One row per run. Environment provenance — without it a result cannot be
reproduced on different hardware and "we release seeded trajectories" is not
checkable.

| column | meaning |
|---|---|
| `run_id` | primary key; matches `episodes.run_id` and the file stem |
| `started_at`, `finished_at` | ISO timestamps |
| `model_id`, `model_revision`, `dtype` | what ran |
| `gpu_name`, `gpu_count`, `driver` | what it ran on |
| `vllm_version`, `torch_version`, `transformers_version`, `python_version` | **the inference stack.** `CLAIMS.md` B5 measures effects moving with it, so this is data, not metadata |
| `git_commit` | the code that produced the rows. Every driver aborts on a dirty working tree for this reason |
| `probe_hash` | the frozen probe-suite hash; a later edit to any probe question is detectable from the data alone |
| `config_json` | the full serialised `ExperimentConfig` |
| `argv` | the exact command line |

Read the stack from here, never from `requirements-gpu.txt`, for any run you did
not launch yourself.

---

## 8. Column x experiment availability

The schema is `CREATE TABLE IF NOT EXISTS`, so each database has the columns its
driver's commit defined. Every analysis script builds its queries from
`PRAGMA table_info` for this reason: a missing column becomes a row in the
coverage table rather than a crash.

Legend: `y` present and populated · `-` column absent · `n` column present but
never written · `p` partial, see the note.

| column | exp1 `sweep` | exp2 | exp3 | exp4 | exp5 | exp6 |
|---|---|---|---|---|---|---|
| `episodes.*` core (arm, model, seed, scores, regret) | y | y | y | y | y | y |
| `episodes.prompt_hash` | y | p | p | p | p | p |
| `turns.*` core (actions, payoffs, logit masses, parity) | y | y | y | y | y | y |
| `turns.cpr_score`, `cpr_own_score`, `cpr_opponent_last`, `cpr_rounds_played` | y | y | y | y | y | y |
| `turns.scaffold_echo` | n | n | n | n | n | n |
| `turns.donor_agent_score` | - | - | y | y | y | y |
| `turns.donor_degenerate` | - | - | y | y | y | y |
| **`turns.displayed_opponent_last`** | **-** | **-** | **-** | **-** | **-** | **y** |
| `turn_details.top_tokens` | y | y | y | y | y | y |
| `turn_details.probe_answers` | y | y | y | y | y | y |
| `turn_details.prompt_preview` | y | y | y | y | y | y |
| `turn_details.prompt_full` | - | - | y | y | y | y |
| `turn_details.scratchpad` | - | - | - | p | y | p |
| `run_meta.*` | y | y | y | y | y | y |

Notes on the `p` cells:

- **`episodes.prompt_hash`** is a *constant*, not a digest.
  `scripts/gpu_run.py` sets `prompt_hash = PROBE_SUITE_HASH[:32]`, identical
  across every arm, framing, readout and model. `cdx/runner.py` computes a real
  digest and the driver never wired it up. It affects no measurement because
  nothing reads it; provenance is carried by `turn_details.prompt_full` instead.
  It was not fixed before exp5 because editing the driver between exp4 and its
  control would have added a second difference to a comparison designed to have
  one.
- **`turn_details.scratchpad`** is populated only where the readout is
  SCRATCHPAD: exp4's `*_scratchpad` groups, all of exp5 (120,000 of 120,000
  turns produced non-trivial scratchpads), and exp6's `*_scratchpad` groups.
  It is absent for every `*_logit` group by construction, not by omission.

The authoritative, machine-generated version of this matrix is the **coverage
table in `EVIDENCE.md`**, produced by `analysis/06_evidence.py` from
`PRAGMA table_info` over the databases actually present. Regenerate it and
prefer it to this table if they ever disagree.

> TODO (author): three cells above are asserted from documentation rather than
> from `PRAGMA table_info`, because this is a code-only checkout and the
> databases were not read while writing this file.
>
> 1. **exp1's column set.** `analysis/01_diagnose_arm3.py` defaults to
>    `--db sweep.sqlite` and reads `cpr_score`, the three `cpr_*` components,
>    `prompt_preview`, `probe_answers` and `run_meta`, so those are present.
>    "exp1 predates most columns" (`analysis/06_evidence.py`) is not more
>    specific than that. Run `PRAGMA table_info(turns)` on `sweep.sqlite` and
>    replace the exp1 column with the literal answer.
> 2. **`prompt_hash` in exp6.** `EXPERIMENTS.md` known defect 1 says the
>    constant-`prompt_hash` bug is "Present in exp2-exp5", but
>    `scripts/gpu_run.py` at HEAD still sets `prompt_hash=PROBE_SUITE_HASH[:32]`
>    and exp6 ran from that driver — so exp6 almost certainly has it too. The
>    matrix above says `p` for exp6 on that basis. Confirm with
>    `SELECT DISTINCT prompt_hash FROM episodes;` on an `exp6_*.sqlite` and
>    correct `EXPERIMENTS.md` in place if it is understated.
> 3. **`turns.scaffold_echo` in exp1.** Recorded as present-but-never-written
>    from exp2's Known gaps; whether the column existed as early as exp1 is not
>    documented anywhere. Same `PRAGMA` check resolves it.

---

## 9. Traps

Things that produce a wrong number without producing an error.

1. **Turn-level intervals.** Never quote them. 0.62x-3.75x too narrow across
   exp2-exp5. Every interval in `CLAIMS.md` comes from
   `analysis/02_episode_level.py`.
2. **Turn 0 in arms 3m and 3c.** At turn 0 there is no last move, so arm 3m is
   byte-identical to arm 3 and arm 3c has `donor_degenerate = 1`. Those rows
   carry no manipulation and pull every `3-3m` and `3-3c` estimate toward zero.
   `analysis/13` reports both `incl_t0` and `excl_t0`; only `excl_t0` is
   quotable.
3. **CPR in a falsifying arm.** CPR scores against the **true** state, so a
   model that reads and trusts a lying block is marked wrong — arm 3s scores
   CPR 0.000 in every exp6 group. Gate CPR on **arm 3 alone**. Applying the 0.85
   gate to 3b, 3c, 3s or 3m discards exactly the cells carrying the result.
4. **Off-task cells.** `action_mass_total < 0.1` means the model emitted prose,
   not a decision. `exp3_mistral_abs` printed `SIGN-FLIP: SUPPORTED` at
   off-task 1.000, from an estimate computed entirely from text containing no
   action tokens. Read off-task before reading any effect.
5. **SE inflation below 1.0x is not a defect signature.** It appears in valid
   groups (`exp5_mistral_sem_minimal` 0.91-1.08, `exp4_qwen_abs_logit` from
   0.86) and indicates stable per-episode rates. Use off-task as the gate; do
   not add this one. This was an author OVERCLAIM, retracted mid-project.
6. **The distinct-trajectory gate does not reduce N.** Low entropy, not
   dependence.
7. **`enable_prefix_caching=True` in exp4 and exp5.** Episodes are independently
   seeded with no carried game state, so the design is i.i.d.; but under
   continuous batching numerics can depend on batch composition. State the
   engine configuration rather than writing "i.i.d." unqualified.
8. **Probe prompts inherit the instruction suffix.** `_probe_turn` builds
   `list(prompt) + probe_suffix`, so CPR is measured on a prompt that still
   carries the reasoning instruction. CPR is therefore **not comparable between
   exp4 and exp5**. Within a run all arms share the instruction, so every CPR
   *contrast* is unaffected.
9. **`config_fingerprint` does not distinguish the scratchpad variant.** It
   hashes `ExperimentConfig`, which does not carry it. exp4 and exp5 are
   distinguished by `run_id`, `run_meta.config_json` and `prompt_full`. Adding
   the field would change the fingerprint of every historical run.
10. **`config_fingerprint` does not distinguish `--no-history` either**, and for
    the same structural reason. `include_history` is an argument to
    `PromptAssembler.assemble`, passed through from `scripts/gpu_run.py
    --no-history`; it is not a field of `ScaffoldConfig` or `ExperimentConfig`.
    So two runs that differ in whether the prompt contained `[HISTORY]` at all —
    the largest prompt-level difference in the project — carry the **same**
    fingerprint. Distinguish them by `run_id`, `run_meta.argv` and
    `turn_details.prompt_full`. This is new with the exp7 code and no released
    database uses it: every database in the corpus was written with
    `include_history=True`, the default, which is why exp1–exp6 still reproduce
    byte-for-byte (pinned by `tests/test_no_history.py`).

---

## 10. Provenance chain

For any number in the paper:

```
number
  -> CLAIM_MAP.md          which claim, which command, which expected value
  -> ep_<stem>.json        the interval, episode-level, 10,000 resamples
  -> <stem>.sqlite         the rows
  -> run_meta              the stack, the GPU, the driver, the git commit, argv
  -> that git commit       the exact code
  -> PREREGISTRATION.md    probe hash 12c9a10d…, stamped on every row
```

Each link is checkable without trusting the one above it.
