# Claim -> command map

One row per claim id in [`CLAIMS.md`](CLAIMS.md): which database carries the
evidence, the exact command that produces it, and the value that command should
print. This is what converts "I read the code" into "I ran it".

Run [`scripts/reproduce.sh`](scripts/reproduce.sh) first. It performs steps 0-4
(gunzip, `analysis/02` per database, `06`, `07`, `13`) in the right order and
writes every artefact the commands below read. The individual commands are
listed so a reviewer can check one claim without re-running the corpus.

**Conventions used below**

- `ep_<stem>.json` is the output of `analysis/02_episode_level.py --db
  <stem>.sqlite --out ep_<stem>.json`. Its contrast keys are
  `ate_true|<opp>` (arm 3 - arm 3b), `3b_minus_1|<opp>` (perturbation) and
  `3_minus_1|<opp>` (ATE_naive); each holds `diff`, `se`, `ci_bootstrap`, `p`,
  `significant`. Top level also holds `sign_flip_verdict` and per-cell
  `se_inflation` / `all_cooperate_fraction`.
- `EXP6_FIELDS.json` is the output of `analysis/13_exp6_fields.py`. Its contrast
  keys are `content_move|<opp>` (arm 3 - 3m), `content_score|<opp>` (3 - 3s) and
  `content_donor|<opp>` (3 - 3c), each with an `excl_t0` block; plus
  `falsification` and `echo_3s`.
- All intervals are episode-level, 10,000 bootstrap resamples, seed 20260811.
  Turn-level intervals appear nowhere in `CLAIMS.md` and must not be quoted
  (B3).
- `jq` is used in the read-out commands for brevity only; the files are plain
  JSON and any reader will do.

---

## A. The pre-registered hypothesis

### A1 — Comprehension repair does not produce opponent-conditional play

| | |
|---|---|
| Databases | `sweep.sqlite` (exp1), `exp2_llama.sqlite`, `exp2_qwen.sqlite`, `exp3_llama_sem.sqlite`, `exp4_*_{logit,scratchpad}.sqlite` (9 valid groups), `exp5_{llama,qwen,mistral}_sem_minimal.sqlite` |
| Command | `python analysis/02_episode_level.py --db exp3_llama_sem.sqlite --out ep_exp3_llama_sem.json` |
| Read | `jq '.sign_flip_verdict, .contrasts["ate_true|allc"], .contrasts["ate_true|tft"]' ep_exp3_llama_sem.json` |

Expected, `exp3_llama_sem` (the only independent replication of the registered
contrast at full N):

| key | expected |
|---|---|
| `sign_flip_verdict` | `"REJECTED"` |
| `ate_true\|allc` `.diff` | `-0.0135`, CI `[-0.0213, -0.0061]` |
| `ate_true\|tft` `.diff` | `-0.0207`, CI `[-0.0299, -0.0115]` |

Both significant, both negative; the prediction was **down** vs TFT and **up**
vs ALLC. Same command on the other databases reproduces the rest of the A1
table: exp1 `+0.042 / +0.052`; exp2 llama `-0.012 / -0.005`; exp2 qwen
`-0.012 / -0.003`; exp5 llama `+0.0738 / +0.0237`, mistral `+0.0294 / +0.0360`,
qwen `+0.0437 / +0.0171`.

The manipulation gate (CPR >= 0.85 in arm 3) is a separate read:
`grep -A3 'cpr' EVIDENCE.md` — exp1 fails it at 0.24-0.31, exp2 onward passes it
at 1.000.

### A2 — Qwen's exp5 TFT cell is inconclusive, not rejected

| | |
|---|---|
| Database | `exp5_qwen_sem_minimal.sqlite` |
| Command | `python analysis/02_episode_level.py --db exp5_qwen_sem_minimal.sqlite --out ep_exp5_qwen_sem_minimal.json` |
| Read | `jq '.contrasts["ate_true|tft"], .sign_flip_verdict' ep_exp5_qwen_sem_minimal.json` |
| Expected | `diff = +0.0171`, `ci_bootstrap = [-0.0010, +0.0348]`, `p = 0.058`, `significant = false`, `sign_flip_verdict = "UNDERPOWERED"` |

The claim is a reporting constraint: writing "rejected in all three models" is
the OVERCLAIM this row exists to block.

### A3 — Under neutral reasoning the state effect is positive and opponent-invariant

| | |
|---|---|
| Databases | `exp5_llama_sem_minimal.sqlite`, `exp5_qwen_sem_minimal.sqlite`, `exp5_mistral_sem_minimal.sqlite` |
| Command | `for m in llama qwen mistral; do python analysis/02_episode_level.py --db exp5_${m}_sem_minimal.sqlite --out ep_exp5_${m}_sem_minimal.json; done` |
| Read | `jq -r '.contrasts["ate_true|allc"].diff, .contrasts["ate_true|tft"].diff' ep_exp5_*_minimal.json` |
| Expected | six values, **all positive**: llama `+0.0738 / +0.0237`, mistral `+0.0294 / +0.0360`, qwen `+0.0437 / +0.0171` |

---

## B. Instrumentation

### B1 — Token parity is necessary and not sufficient; density must match

| | |
|---|---|
| Databases | `sweep.sqlite` (44%-content placebo), `exp2_llama.sqlite` (85%) |
| Command | `python analysis/02_episode_level.py --db sweep.sqlite --out ep_sweep.json` then the same for `exp2_llama.sqlite` |
| Read | `jq '.contrasts["ate_true|allc"]' ep_sweep.json ep_exp2_llama.json` |
| Expected | exp1 `diff = +0.042`, `p < 1e-4`; exp2 `diff = -0.012` |

Density itself is not in the databases — it is a property of the templates.
Check it against `turn_details.prompt_full`, which stores the decoded prompt as
sent:

```sql
SELECT arm, prompt_full FROM turn_details WHERE turn = 5 LIMIT 4;
```

> TODO (author): **the two source documents disagree on exp2's p-value for this
> contrast.** `CLAIMS.md` B1 and `EXPERIMENTS.md` (exp1, "Consequence") both say
> `-0.012 (p = 0.24)`; `EXPERIMENTS.md`'s own exp2 findings table gives the
> llama-vs-ALLC ATE_true p-value as `0.004`. One of the two is wrong and the
> distinction matters — 0.24 supports "did not survive density matching", 0.004
> does not. Resolve from `ep_exp2_llama.json` and correct whichever document is
> wrong, in place, marked as a correction.

### B2 — Character padding cannot control token count under BPE

| | |
|---|---|
| Databases | none — this is a property of the tokenizers |
| Command | `python scripts/tokenizer_check.py` |
| Expected | parity target 34 (Llama-3.1-8B-Instruct), 39 (Qwen2.5-7B-Instruct), 45 (Mistral-7B-Instruct-v0.3); filler `'\n'` id 198 for Llama and Qwen, `' '` id 29473 for Mistral (Mistral has no single-token newline) |

Needs `transformers` and, for Llama, an accepted gated licence plus `HF_TOKEN`.
The same targets are re-derived at every run's startup and gated in-repo by
`tests/test_score_field_parity.py` and `tests/test_exp1_to_exp5_unchanged.py`:

```bash
python -m pytest tests/test_score_field_parity.py tests/test_exp1_to_exp5_unchanged.py -q
```

### B3 — Turn-level standard errors understate uncertainty

| | |
|---|---|
| Databases | every exp2-exp5 database |
| Command | step 1 of `scripts/reproduce.sh` |
| Read | `jq -r '.cells \| to_entries[] \| "\(.key) \(.value.se_inflation)"' ep_exp*.json` |
| Expected | inflation spanning **0.62x to 3.75x** across exp2-exp5. Sub-1.0 values appear in **valid** groups — `exp5_mistral_sem_minimal` 0.91-1.08, `exp4_qwen_abs_logit` from 0.86 — and are not a dead-cell signature |

### B4 — A significant p-value from an uninspected cell is worth nothing

| | |
|---|---|
| Database | `exp3_mistral_abs.sqlite` |
| Command | `python analysis/02_episode_level.py --db exp3_mistral_abs.sqlite --out ep_exp3_mistral_abs.json` |
| Read | `jq '.sign_flip_verdict, .contrasts["ate_true|tft"]' ep_exp3_mistral_abs.json` then the off-task column for that database in `EVIDENCE_cells.csv` |
| Expected | `sign_flip_verdict = "SUPPORTED"`, `ate_true\|tft` `diff = -0.2266`, `p < 1e-4` — **and** off-task rate `1.000` in the arm-3/ALLC and arm-3/TFT cells, i.e. the estimate is computed entirely from prose containing no action tokens |

This row is supposed to look like a result and is not one. `EVIDENCE.md` marks
the cells VOID mechanically from the off-task gate (0.10).

### B5 — Measured causal effects shift with the inference stack

| | |
|---|---|
| Databases | `exp3_llama_sem.sqlite` (vLLM 0.27.1 / torch 2.13+cu130 / transformers 5.15.0) vs `exp4_llama_sem_logit.sqlite` (0.11.0 / 2.8.0+cu128 / 4.57.6) |
| Command | `python analysis/07_cross_experiment.py --csv EVIDENCE_cells.csv --out CROSS_EXPERIMENT.md` (stack-drift table), plus `jq '.contrasts' ep_exp3_llama_sem.json ep_exp4_llama_sem_logit.json` |
| Expected | `ate_true` replicates to `~0.002` (llama_sem allc `-0.0135` -> `-0.0145`; tft `-0.0207` -> `-0.0228`). `3b_minus_1\|allc` does **not**: `-0.1806` -> `-0.2218`, a 4pp shift on identical inputs |
| Cross-check | `SELECT run_id, vllm_version, torch_version, transformers_version, driver, gpu_name FROM run_meta;` on both databases — the stacks are recorded per run, not assumed |

This claim is why `requirements.txt` and `requirements-gpu.txt` are pinned.

---

## C. Reading and truth

### C1 — Models report the injected block in preference to the raw history

| | |
|---|---|
| Databases | the nine exp3 groups (`exp3_{llama,qwen,mistral}_{sem,swap,abs}.sqlite`) |
| Command | `for f in exp3_*.sqlite; do python analysis/04_donor_echo.py --db "$f" --arm 3c; done` |
| Expected | `DONOR_ECHO` **0.940-1.000** in every group; `CORRECT` 0.000-0.060; `OFF_BY_ONE` ~0.000 (rules out an arithmetic account); `OTHER` ~0.000 (rules out a parser fault). 64,000 probed turns, flat across turn index |
| Not optional | turn 0 is excluded (`donor_degenerate = 1`); at score 0 with no last move the donor *is* the true state and including those rows manufactures echo evidence |

### C2 — With reading held constant, the truth of the state barely matters

| | |
|---|---|
| Databases | `exp3_llama_sem.sqlite`, `exp3_qwen_sem.sqlite` |
| Command | `python analysis/06_evidence.py --out EVIDENCE.md --csv EVIDENCE_cells.csv`, then read the arm-3 and arm-3c defect rates for those groups; interval version: `python analysis/08_decomposition_ci.py --out DECOMPOSITION.md` |
| Expected | llama_sem allc `0.097` (3c) vs `0.085` (3); llama_sem tft `0.124` vs `0.100`; qwen_sem allc `0.045` vs `0.051`; qwen_sem tft `0.306` vs `0.068` — the one documented exception, resolved by C3 |

### C3 — Qwen-vs-TFT retaliates against a betrayal that did not occur

| | |
|---|---|
| Databases | `exp6_qwen_sem_logit.sqlite` (the measurement); `exp3_qwen_sem.sqlite` (the earlier inference) |
| Command | `python analysis/13_exp6_fields.py --db exp6_qwen_sem_logit.sqlite --out EXP6_FIELDS_qwen.json` |
| Read | `jq '.[0].contrasts["content_score|tft"].excl_t0, .[0].contrasts["content_move|tft"].excl_t0, .[0].contrasts["content_donor|tft"].excl_t0' EXP6_FIELDS_qwen.json` |
| Expected | score-only `+0.0138` CI `[+0.0059, +0.0217]`; move-only **`-0.4049`** CI `[-0.4160, -0.3938]`; whole-donor `-0.2016`. Flipping one word moves defection 29x more than shifting the score by 15 points, and further than replacing the entire block |
| Why exp3 could not do this | exp3 persisted only `turns.donor_agent_score`. `turns.displayed_opponent_last` was added in exp6; see `DATA.md`, availability matrix |

### C4 — Numeric fields, not their content, drive Qwen under abstract labels

| | |
|---|---|
| Database | `exp3_qwen_abs.sqlite` |
| Command | `python analysis/02_episode_level.py --db exp3_qwen_abs.sqlite --out ep_exp3_qwen_abs.json` |
| Expected | `ate_true\|allc` `+0.745`, `ate_true\|tft` `+0.738`, both `p < 1e-4`; arm 3c `0.754` vs arm 3 `0.771` (false numbers ~ true numbers); opponent-invariant to three decimals; off-task clean, so not a readout artefact |
| Scope | SUPPORTED, **NARROW** — one model, one framing. Say so in the sentence |

### C5 — The opponent's last move dominates; the cumulative score does not

| | |
|---|---|
| Databases | `exp6_{llama,qwen,mistral}_sem_logit.sqlite` |
| Command | `python analysis/13_exp6_fields.py --glob 'exp6_*_logit.sqlite' --out EXP6_FIELDS.json` |
| Read | the SUMMARY table it prints, or `jq '.[] \| {group, s: .contrasts["content_score|tft"].excl_t0.diff, m: .contrasts["content_move|tft"].excl_t0.diff}' EXP6_FIELDS.json` |

Expected (turn 0 excluded from both arms — at turn 0 there is no last move and
3m is byte-identical to 3):

| group | opp | `3-3s` | `3-3m` | ratio |
|---|---|---|---|---|
| qwen | tft | `+0.0138` | `-0.4049` | 29x |
| qwen | allc | `+0.0048` ns | `-0.2839` | 59x |
| llama | allc | `-0.0227` | `-0.0931` | 4.1x |
| llama | tft | `-0.0317` | `-0.0891` | 2.8x |
| mistral | tft | `+0.0001` ns | `-0.0141` | — |
| mistral | allc | `-0.0004` | `-0.0115` | 29x |

**Four of six score contrasts exclude zero.** Write "dominates", never "does
nothing" — the OVERCLAIM is named in `CLAIMS.md` C5 because it was drafted in
the wrong form once already.

### C6 — The model reads the false state perfectly and acts on it barely

| | |
|---|---|
| Databases | `exp6_{llama,qwen,mistral}_sem_logit.sqlite` |
| Command | `python analysis/13_exp6_fields.py --glob 'exp6_*_logit.sqlite' --out EXP6_FIELDS.json` (Part 5, the echo test) |
| Read | `jq '.[] \| {group, echo: .contrasts.echo_3s}' EXP6_FIELDS.json` — the echo block is written into the `contrasts` object by `analyse()`, not at the top level |
| Expected | `{"n": 10000, "shown": 10000, "true": 0, "neither": 0}` per group: **100.0% matched the displayed lie, 0 matched the true score**, in all three models — 30,000 probes, unanimous. Read beside C5's `3-3s` of 0.5-3.2pp |
| Corollary to check too | arm 3s scores CPR **0.000** in every group (`EVIDENCE.md`, CPR by arm). CPR in a falsifying arm is a belief measure, not a validity gate; gate on arm 3 alone, where it is 1.000 |

### C7 — The last-move effect does not survive chain-of-thought in every model

| | |
|---|---|
| Databases | `exp6_*_logit.sqlite` and `exp6_*_scratchpad.sqlite` |
| Command | `python analysis/13_exp6_fields.py --glob 'exp6_*.sqlite' --out EXP6_FIELDS.json` |
| Read | `content_move\|<opp>.excl_t0.diff` for each group, logit vs scratchpad |
| Expected | qwen allc `-0.2839` -> `-0.1447`; qwen tft `-0.4049` -> `-0.0520`; mistral allc `-0.0115` -> `-0.0328`; mistral tft `-0.0141` -> `-0.0344`; **llama allc `-0.0931` -> `+0.0126`** and llama tft `-0.0891` -> `-0.0155` |
| The sentence | present under LOGIT in three of three models, under CoT in two of three. Name the readout and name llama. The literature uses CoT |

### C8 — Arm 3c is a weak instrument, and vs ALLC a zero-dose control

| | |
|---|---|
| Databases | `exp6_*_logit.sqlite` |
| Command | `python analysis/13_exp6_fields.py --glob 'exp6_*_logit.sqlite' --out EXP6_FIELDS.json` (Part 1, manipulation integrity) |
| Read | `jq '.[] \| {group, f: .falsification}' EXP6_FIELDS.json` |
| Expected | `3m` falsification rate `1.0000` (19,000 of 19,000) in both opponents, all models; `3c` vs **allc** `0.0000` in all three models; `3c` vs tft `0.2988` llama / `0.1392` mistral / `0.3765` qwen. Effect at zero dose: `-0.0140` / `+0.0000` ns / `+0.0128` |
| Method note | the rate is verified by self-join of `turns.displayed_opponent_last` against the previous turn's `opponent_action`, so the check cannot share a bug with the writer |
| Still open | the per-falsified-row overshoot — rescaled by dose, 3c exceeds 3m by 26% (llama) and 32% (qwen). Limitations, not Results |

---

## D. Presentation

### D1 — Inserting a state block changes behaviour more than its content does

| | |
|---|---|
| Databases | `exp3_{llama,qwen,mistral}_sem.sqlite` |
| Command | step 1 of `scripts/reproduce.sh` |
| Read | `jq '.contrasts["3b_minus_1|allc"].diff, .contrasts["3b_minus_1|tft"].diff' ep_exp3_*_sem.json` |
| Expected | llama `-0.181 / -0.192`; qwen `+0.039 / +0.038`; mistral `0.000` (floor — 99.8% of episodes never defect, so report the floor, not the estimate) |
| The sentence | always name the model. "The container effect exists in LLMs" is the OVERCLAIM |

### D2 — The container effect is lexical

| | |
|---|---|
| Databases | `exp3_{llama,qwen}_sem.sqlite` vs `exp3_{llama,qwen}_abs.sqlite` |
| Command | `python analysis/07_cross_experiment.py --csv EVIDENCE_cells.csv --out CROSS_EXPERIMENT.md` (lexical test table) |
| Expected | llama allc `-0.181 [-0.192, -0.169]` semantic vs `+0.007 [-0.003, +0.016] ns` abstract; llama tft `-0.192` vs `+0.029`; qwen allc `+0.039` vs `-0.055`; qwen tft `+0.038` vs `-0.065`. In llama it disappears; **in qwen it reverses sign** |
| Also | baselines move: llama defects 0.28-0.31 under Cooperate/Defect and 0.71-0.74 under X/Y |
| Status | this was a **pre-specified** falsification test of exp2's lexical account, and the account survived it |

### D3 — A matched placebo is required or two large effects cancel

| | |
|---|---|
| Database | `exp4_qwen_sem_scratchpad.sqlite` |
| Command | `python analysis/02_episode_level.py --db exp4_qwen_sem_scratchpad.sqlite --out ep_exp4_qwen_sem_scratchpad.json` |
| Read | `jq '.contrasts["3b_minus_1|allc"], .contrasts["ate_true|allc"], .contrasts["3_minus_1|allc"]' ep_exp4_qwen_sem_scratchpad.json` |
| Expected | perturbation `+0.1934 [+0.1741, +0.2126]` p<1e-4; ATE_true `-0.2135 [-0.2279, -0.1989]` p<1e-4; **ATE_naive `-0.0202 [-0.0397, -0.0000]` p=0.046**. Two ~20-point effects of opposite sign cancelling to near zero |

---

## E. Readout and reasoning

### E1 — Defection rises sharply from constrained readout to chain-of-thought

| | |
|---|---|
| Databases | `exp4_*_sem_logit.sqlite`, `exp4_*_sem_scratchpad.sqlite`, `exp5_*_sem_minimal.sqlite` |
| Command | `python analysis/07_cross_experiment.py --csv EVIDENCE_cells.csv --out CROSS_EXPERIMENT.md` (readout ladder) |
| Expected | P(D given arm 3b), semantic: llama `0.102 / 0.579 / 0.639`; qwen `0.042 / 0.597 / 0.597`; mistral `0.000 / 0.335 / 0.406` across LOGIT / CoT guided / CoT minimal |
| Citation rule | **never cite exp4 alone.** exp4's scratchpad instruction names the finite horizon; the claim is licensed only jointly with exp5's control. `CROSS_EXPERIMENT.md` marks the ladder confounded in place |

### E2 — The rise is not caused by naming the finite horizon

| | |
|---|---|
| Databases | `exp4_*_sem_scratchpad.sqlite` (guided) vs `exp5_*_sem_minimal.sqlite` (minimal) |
| Command | as E1 |
| Expected | removing "how many rounds remain" leaves defection unchanged or **higher** in all six cells. Averages over arms: llama `0.603 -> 0.694`, qwen `0.529 -> 0.689`, mistral `0.369 -> 0.407`. Arm 1: llama `0.731 -> 0.557`, qwen `0.650 -> 0.404` — the horizon effect runs *backwards* relative to backward induction |
| Instrument gate | the exp4 wording is pinned by `tests/test_guided_still_names_the_horizon` and the default variant by `tests/test_default_is_guided_so_exp4_reproduces`. Run `python -m pytest tests/test_instruction_by_readout.py -q` |

### E3 — exp5 controls salience, not availability

| | |
|---|---|
| Databases | every database |
| Command | `for f in *.sqlite; do echo "$f"; sqlite3 "$f" "SELECT DISTINCT horizon_mode, horizon FROM episodes;"; done` |
| Expected | `known\|20` everywhere. `HorizonMode.STOCHASTIC` is implemented (`cdx/config.py`, `cdx/game.py`, gated by `tests/test_stochastic_horizon.py`, 6 tests) and **was never run** |
| The sentence | the rules section already states the game lasts exactly 20 rounds, so the horizon was never hidden — the instruction only made it salient. "Backward induction is falsified" is an OVERCLAIM |

### E4 — Reasoning attenuates the container effect by ~60% without abolishing it

| | |
|---|---|
| Databases | `exp4_llama_sem_logit.sqlite`, `exp5_llama_sem_minimal.sqlite`, `exp4_llama_sem_scratchpad.sqlite` |
| Command | step 1 of `scripts/reproduce.sh` |
| Read | `jq '.contrasts["3b_minus_1|allc"], .contrasts["3b_minus_1|tft"]' ep_exp4_llama_sem_logit.json ep_exp5_llama_sem_minimal.json ep_exp4_llama_sem_scratchpad.json` |
| Expected | LOGIT `-0.2218 [-0.2387, -0.2052]` / `-0.2069 [-0.2253, -0.1880]`; CoT minimal `-0.0920 [-0.1054, -0.0788]` / `-0.0747 [-0.0869, -0.0620]`; CoT guided `+0.0215 [+0.0039, +0.0390]` / `+0.0161 [-0.0012, +0.0334]`. **No CI overlap between any pair** |
| Scope | SUPPORTED, NARROW — llama only; the other two had no effect to attenuate |

### E5 — Under neutral reasoning the container effect appears in all three models

| | |
|---|---|
| Databases | `exp4_*_sem_logit.sqlite` vs `exp5_*_sem_minimal.sqlite` |
| Command | step 1 of `scripts/reproduce.sh` |
| Read | `jq '.contrasts["3b_minus_1|allc"].diff' ep_exp4_*_sem_logit.json ep_exp5_*_sem_minimal.json` |
| Expected | llama `-0.222 -> -0.092` ***; qwen **`+0.033 -> -0.053`** ***; mistral `0.000 -> -0.021` ***. Qwen's sign reverses; five of six point estimates negative, four significant |
| Status | new in exp5 — **not** implied by exp4's guided cells |

---

## F, G, H — not command-checkable

`CLAIMS.md` sections F (Retracted), G (Not tested / Limitations) and H
(Recommended spine) contain no new estimand.

- **F** is an audit trail. Each retracted claim's replacement is checkable
  through the row above that supersedes it: the `.ljust()` artefact through C1,
  "exp1 was a false positive" through B1, "CoT kills the container effect"
  through E4, Qwen's conditional state effect through A3/E5, "inflation below
  1.0x marks a dead cell" through B3.
- **G** items are absences. G1 (stochastic horizon) is checkable as E3's SQL;
  G3 (`OpponentPolicy.LLM` raises in `cdx/game.py`) as
  `python -c "from cdx.game import build_opponent"` plus reading the raise;
  G7 is closed by C3 for exp6 and open for exp2-exp5 (see `DATA.md`).
- **G8 is the one open analysis, not an absence.** "Does the treatment effect
  differ by model?" — ATE_true vs ALLC is `+0.074 / +0.044 / +0.029` across
  llama / qwen / mistral in exp5, and no bootstrap difference-of-differences has
  been run. It decides whether the paper reports a pooled effect or three case
  studies.

**G8 now has a script.** `analysis/14_reviewer_responses.py` check B is the
bootstrap difference-of-differences, pairwise and joint:

```bash
python analysis/14_reviewer_responses.py --out REVIEWER_RESPONSES.json
```

It needs no GPU — every number comes from a database already in the repository.

> TODO (author): `analysis/14_reviewer_responses.py` landed after `CLAIMS.md`
> was last written and is **not** reflected there. Two of its checks change what
> may be said, not just what is known:
>
> - **Check A, revealed-opponent stratification.** Along a cooperative
>   trajectory TFT and ALLC are observationally identical, so until the agent
>   defects, no arm of this study contains one bit distinguishing them — the
>   pre-registered test was partly running on turns where it could not have
>   succeeded. Depending on the base rate and the revealed-stratum result, A1
>   either strengthens considerably or acquires a scope limit. The script is
>   explicit that the stratification is post-treatment and therefore descriptive,
>   not causal.
> - **Check B** settles G8: pooled effect, or three case studies.
>
> Run it, then add rows here and statuses in `CLAIMS.md` for whatever it
> returns — before drafting Results. This row is a placeholder for numbers that
> do not exist yet, and is marked as one so it cannot be mistaken for a result.

---

## What to do when a number disagrees

Do not adjust the claim to the output. In order:

1. Confirm the database is the one named here (`SELECT run_id, argv FROM run_meta;`).
2. Confirm the interval is episode-level and 10,000 resamples, seed 20260811 —
   turn-level intervals differ by 0.62x-3.75x and are the most common cause of
   a mismatch.
3. Confirm turn 0 is excluded where the row says so (arms 3m and 3c).
4. Confirm the stack: B5 says a 4pp difference in the perturbation contrast is
   *expected* between the exp1-exp3 and exp4-exp6 stacks.
5. If it still disagrees, that is a finding. Record it in `EXPERIMENTS.md` as a
   new entry — append-only — rather than editing the number that was published.
