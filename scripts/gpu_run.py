#!/usr/bin/env python3
"""BATCHED GPU RUN — the paid session.

Advances every episode in a cell in LOCKSTEP so vLLM sees one large batch per
turn instead of one prompt at a time. That is where the speedup lives; an
unbatched GPU run is only a few times faster than the laptop and wastes money.

DURABILITY
    Every turn is written to SQLite as it happens, and completed cells are
    recorded so a restart skips them. A crash or a preemption costs one cell, not
    the session. This matters more on rented hardware than anywhere else.

BUDGET
    --budget-minutes stops cleanly and reports, so a forgotten instance cannot
    silently drain the balance. Always set it.

    python scripts/gpu_run.py --verify
    python scripts/gpu_run.py --episodes 500 --budget-minutes 45
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.analysis import ProportionDiff, min_detectable_effect
from cdx.config import (
    Action,
    Arm,
    ExperimentConfig,
    Framing,
    GameConfig,
    ModelConfig,
    OpponentPolicy,
    ReadoutMode,
    ScaffoldConfig,
)
from cdx.db import EpisodeRecord, Store, TurnRecord, encode_top_tokens
from cdx.donor import DonorStats, select_donor
from cdx.game import Game, build_opponent, replay
from cdx.optimal import episode_regret, predicted_defection_direction, solve
from cdx.probe import (
    PROBE_SUITE,
    PROBE_SUITE_HASH,
    ProbeMethod,
    ProbeResult,
    render_replay_probe,
    score_answer,
)
from cdx.runner import (
    DEFAULT_SCRATCHPAD_PROMPT,
    SCRATCHPAD_PROMPTS,
    instruction_for,
)
from cdx.scaffold import (
    DEFAULT_STATE_TEMPLATE,
    TEMPLATE_DENSITY_TOLERANCE,
    HISTORY_HEADER,
    SCORE_FALSIFICATION,
    STATE_HEADER,
    STATE_TEMPLATES,
    PromptAssembler,
    ScaffoldBuilder,
    falsified_view,
    move_was_falsified,
)
from cdx.seeding import EpisodeKey, purpose_rng

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gpu_run")

RULE = "=" * 74
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_ARMS = ["1", "3", "3b"]
DEFAULT_OPPONENTS = ["tft", "allc"]


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, minutes: float) -> None:
        self.limit = minutes * 60
        self.start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def remaining(self) -> float:
        return self.limit - self.elapsed

    def check(self) -> None:
        if self.remaining <= 0:
            raise BudgetExceeded(f"budget of {self.limit/60:.0f} min exhausted")


def environment() -> dict:
    """Everything needed to reproduce this run on other hardware."""
    import platform
    import subprocess

    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "unavailable"

    def _sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=10).stdout.strip()[:200]
        except Exception:
            return "unavailable"

    gpu = _sh("nvidia-smi --query-gpu=name --format=csv,noheader")
    return {
        "gpu_name": gpu.splitlines()[0] if gpu else "unknown",
        "gpu_count": len(gpu.splitlines()) if gpu else 0,
        "driver": _sh("nvidia-smi --query-gpu=driver_version "
                      "--format=csv,noheader").splitlines()[:1] and
                  _sh("nvidia-smi --query-gpu=driver_version "
                      "--format=csv,noheader").splitlines()[0] or "unknown",
        "vllm_version": _v("vllm"),
        "torch_version": _v("torch"),
        "transformers_version": _v("transformers"),
        "python_version": platform.python_version(),
        "git_commit": _sh("git rev-parse HEAD") or "not-a-git-repo",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--probe-every", type=int, default=4)
    ap.add_argument("--framing", default=Framing.SEMANTIC.value)
    ap.add_argument("--objective", default="self_interest",
                    choices=["none", "self_interest", "joint"])
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
    ap.add_argument("--opponents", nargs="*", default=DEFAULT_OPPONENTS)
    ap.add_argument("--swap-labels", action="store_true")
    ap.add_argument("--readout", default=ReadoutMode.LOGIT.value,
                    choices=[m.value for m in ReadoutMode],
                    help="LOGIT reads the action distribution from the next "
                         "token with no reasoning space. SCRATCHPAD lets the "
                         "model generate reasoning first, then reads the "
                         "action from the continuation. Prior work in this "
                         "literature uses the latter, so a result that holds "
                         "only under LOGIT does not reach it.")
    ap.add_argument("--max-scratchpad-tokens", type=int, default=128,
                    help="Reasoning budget per decision. Runtime is roughly "
                         "linear in this: at 192 a 12-cell N=1600 sweep is ~5h "
                         "on an A100. 128 is ample for a 20-round dilemma. "
                         "Ignored unless --readout scratchpad.")
    ap.add_argument("--scratchpad-prompt", default=DEFAULT_SCRATCHPAD_PROMPT,
                    choices=sorted(SCRATCHPAD_PROMPTS),
                    help="Which reasoning instruction to use. GUIDED (exp4, the "
                         "default) names the state, the opponent and how many "
                         "rounds remain - that last clause hands the model the "
                         "backward-induction argument for defecting, so the "
                         "LOGIT-vs-SCRATCHPAD comparison it produces is "
                         "confounded. MINIMAL asks only for step-by-step "
                         "thought and names nothing, isolating the effect of "
                         "reasoning from the effect of what the instruction "
                         "points at. Ignored unless --readout scratchpad.")
    ap.add_argument("--no-history", action="store_true",
                    help="Render the prompt WITHOUT the [HISTORY] section, so "
                         "the injected [STATE] block is the only source of "
                         "state. Every experiment so far shipped the raw log "
                         "one section below the block, which makes arms 3c/3s/"
                         "3m contradiction manipulations rather than "
                         "false-state ones: 'the model discounts a locally "
                         "refuted claim, more so when it is cheap to check' "
                         "explains the whole exp6 pattern including the "
                         "score/move asymmetry. This flag removes the "
                         "refutation and nothing else. It also turns arm 1 into "
                         "a genuine state-deprivation condition and stops arm-3 "
                         "CPR being a copy task. OFF by default: exp1-exp6 must "
                         "reproduce byte-identically from HEAD.")
    ap.add_argument("--state-template", default=DEFAULT_STATE_TEMPLATE,
                    choices=sorted(STATE_TEMPLATES),
                    help="Which [STATE] rendering to inject. exp1-exp7 all ran "
                         "on 'original' and it is the default, so omitting this "
                         "reproduces them byte-for-byte. The other three are "
                         "exp8's template family: same four fields carrying the "
                         "same information, different field LABELS "
                         "('reworded'), different field ORDER ('*_permuted'), "
                         "or both. A placebo-controlled ablation method that "
                         "has only ever been run on one prompt string is one "
                         "observation, not an instrument; this is the factor "
                         "that makes it more than one. NOT a ScaffoldConfig "
                         "field - that would rewrite config_fingerprint on "
                         "every historical row - so the template is carried by "
                         "run_id, by run_meta.config_json and by "
                         "turn_details.prompt_full, exactly as the scratchpad "
                         "variant and --no-history are.")
    ap.add_argument("--insertion-index", type=int, default=1,
                    choices=[0, 1, 2],
                    help="Where the [STATE] block is inserted. 1 = after the "
                         "rules and before [HISTORY], which is what every "
                         "experiment to date used. 2 = after [HISTORY] and "
                         "before the instruction; illegal with --no-history, "
                         "because there is no second seam then and the block "
                         "would silently land after the instruction instead. "
                         "cdx/config.py's own note says lost-in-the-middle "
                         "effects produce >30% swings from position alone, and "
                         "no driver has ever tested that on this prompt. This "
                         "IS a config field and moves config_fingerprint, "
                         "which is safe: historical rows all carry 1.")
    ap.add_argument("--logprobs-top-k", type=int, default=None,
                    help="Top-K logprobs requested per decision. Defaults to "
                         "cdx.backends_vllm.LOGPROBS_TOP_K (20, matching exp3). "
                         "Some vLLM builds cap this unless the engine is told "
                         "otherwise; 0.27 rejected 60 outright. Recorded in "
                         "run_meta, so changing it between runs is visible.")
    ap.add_argument("--full-prompt-episodes", type=int, default=3,
                    help="Store the COMPLETE decoded prompt for episodes with "
                         "id below this. prompt_preview truncates the middle, "
                         "where the [STATE] block lives once history grows. 0 "
                         "disables; every episode would add gigabytes.")
    ap.add_argument("--budget-minutes", type=float, default=45.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--run-id", default="gpu")
    ap.add_argument("--db", default="results.sqlite")
    ap.add_argument("--out", default="gpu_results.json")
    ap.add_argument("--verify", action="store_true",
                    help="Instrument check plus a 2-episode run, then exit.")
    args = ap.parse_args()

    if args.no_history and args.insertion_index > 1:
        print("  ABORT: --insertion-index 2 needs [HISTORY] to sit after. With "
              "--no-history there is no second seam and the block would land "
              "after the instruction. Refused before the model is loaded.")
        return 2

    if args.verify:
        args.episodes, args.budget_minutes = 2, 10.0
        args.run_id, args.db, args.out = "verify", "verify.sqlite", "verify.json"

    from cdx.backends_vllm import LOGPROBS_TOP_K, VLLMBackend

    print(f"\n{RULE}\nSTEP 1  load {args.model}\n{RULE}")
    t0 = time.time()
    backend = VLLMBackend(
        ModelConfig(model_id=args.model, dtype="bfloat16",
                    max_scratchpad_tokens=args.max_scratchpad_tokens),
        swap_labels=args.swap_labels,
        logprobs_top_k=args.logprobs_top_k or LOGPROBS_TOP_K,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    framing = Framing(args.framing)
    game_config = GameConfig(horizon=args.horizon)
    experiment = ExperimentConfig(
        run_id=args.run_id, game=game_config, probe_text_hash=PROBE_SUITE_HASH,
        scaffold=ScaffoldConfig(objective=args.objective,
                                swap_action_labels=args.swap_labels,
                                insertion_index=args.insertion_index),
    )
    builder = ScaffoldBuilder(backend.tokenizer, experiment.scaffold,
                              state_template=args.state_template)
    assembler = PromptAssembler(backend.tokenizer, experiment.scaffold)
    store = Store(args.db)

    print(f"\n{RULE}\nSTEP 2  verify instrument\n{RULE}")
    print(f"  framing {framing.value}   objective {args.objective}"
          f"{'   LABELS SWAPPED' if args.swap_labels else ''}")
    print(f"  readout {args.readout}"
          + (f"   scratchpad budget {args.max_scratchpad_tokens} tokens"
             if args.readout == ReadoutMode.SCRATCHPAD.value else ""))
    if args.readout == ReadoutMode.SCRATCHPAD.value:
        # Printed verbatim because the exact wording is the confound: exp4's
        # GUIDED prompt names the horizon, which is the backward-induction
        # argument for defecting. Anyone reading a log must be able to see
        # which instruction produced the numbers underneath it.
        print(f"  scratchpad prompt: {args.scratchpad_prompt}")
        print(f"    {instruction_for(framing, ReadoutMode.SCRATCHPAD, args.scratchpad_prompt).strip()!r}")
        print("  NOTE: the action is read from the continuation after the")
        print("        generated reasoning. Watch off-task: if reasoning")
        print("        pushes action mass down the way abstract framing did")
        print("        for Mistral, the cells are unreadable regardless of")
        print("        what the defection rate says.")
    for action in (Action.COOPERATE, Action.DEFECT):
        ids = backend._action_token_ids[framing][action]
        print(f"  {action.value}: {ids} -> "
              f"{[backend.tokenizer.decode([i]) for i in ids]}")
    print(f"  filler {builder.filler_text!r} -> "
          f"{backend.tokenizer.encode(builder.filler_text)}")
    # The three exp8 factors, printed together and on the real tokeniser. The
    # parity target is a property of (tokeniser, template): it is NOT expected
    # to match the original template's 34/39/45, and a run that quietly
    # inherited the wrong template would be visible here as the wrong number.
    print(f"  template    {builder.template_name}  "
          f"fields {' | '.join(builder.template.labels)}")
    print(f"  parity target {builder.block_tokens} tokens "
          f"(derived from this template alone)")
    print(f"  position    insertion_index={args.insertion_index} -> "
          f"{'after the rules, before [HISTORY]' if args.insertion_index == 1 else ('after [HISTORY]' if args.insertion_index == 2 else 'before the rules')}")
    print(f"  probe hash  {PROBE_SUITE_HASH[:32]}...")
    print(f"  config      {experiment.fingerprint()[:32]}...")
    _verify_history_condition(builder, assembler, backend, experiment,
                              game_config, framing, args)
    _verify_template_density(builder, game_config, framing, args)
    env = environment()
    store.write_run_meta(
        args.run_id,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_id=args.model, model_revision=backend.model_config.revision,
        dtype=backend.model_config.dtype, probe_hash=PROBE_SUITE_HASH,
        config_json=json.dumps({k: str(v) for k, v in vars(args).items()}),
        argv=" ".join(sys.argv), **env,
    )
    print(f"  database    {Path(args.db).resolve()}")
    print(f"  gpu         {env['gpu_name']} x{env['gpu_count']}  driver {env['driver']}")
    print(f"  versions    vllm={env['vllm_version']} torch={env['torch_version']}")
    print(f"  git commit  {env['git_commit'][:12]}")
    if env["git_commit"] == "not-a-git-repo":
        print("  WARNING: not a git repo. The pre-registration timestamp is your")
        print("           evidence that probes were frozen before data. Commit first.")
    print("  NOTE: if the action tokens do not match what the prompt asks for,")
    print("        every downstream number is meaningless. Stop and fix.")

    # ---- resume ---------------------------------------------------------
    results: dict[str, dict] = {}
    out_path = Path(args.out)
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
            print(f"\n  RESUMING: {len(results)} cell(s) already complete "
                  f"-> {sorted(results)}")
        except json.JSONDecodeError:
            print(f"\n  WARNING: {args.out} unreadable; starting fresh")

    cells = [(Arm(a), OpponentPolicy(o)) for a in args.arms for o in args.opponents]
    budget = Budget(args.budget_minutes)

    print(f"\n{RULE}\nSTEP 3  {len(cells)} cells x {args.episodes} episodes"
          f"   budget {args.budget_minutes:.0f} min\n{RULE}")

    try:
        for i, (arm, opponent) in enumerate(cells, start=1):
            name = f"{arm.value}|{opponent.value}"
            if name in results:
                print(f"\n  cell {i}/{len(cells)}: {name} — already done, skipping")
                continue
            budget.check()
            print(f"\n  cell {i}/{len(cells)}: arm={arm.value} vs {opponent.value}"
                  f"   ({budget.remaining/60:.1f} min left)")
            results[name] = run_cell(
                backend, builder, assembler, store, experiment, game_config,
                framing, arm, opponent, args, budget,
            )
            # Persist after EVERY cell, so a crash costs at most one cell.
            out_path.write_text(json.dumps(results, indent=2))
    except BudgetExceeded as exc:
        print(f"\n  STOPPED: {exc}")
    except KeyboardInterrupt:
        print("\n  INTERRUPTED by user")
    finally:
        out_path.write_text(json.dumps(results, indent=2))
        try:
            store._conn.execute(
                "UPDATE run_meta SET finished_at=? WHERE run_id=?",
                (datetime.now(timezone.utc).isoformat(timespec='seconds'), args.run_id))
        except Exception:
            pass
        store.close()

    report(results, game_config, args, budget)
    print(f"\n  json  -> {out_path.resolve()}")
    print(f"  sqlite-> {Path(args.db).resolve()}")
    print("  REMEMBER TO TERMINATE THE INSTANCE.")
    return 0


def _verify_template_density(builder, game_config, framing, args) -> None:
    """Prove the TEMPLATE factor is not a filler-fraction factor, on the real
    tokeniser, before any GPU time is spent.

    exp8's `A` is defended by an algebraic argument: arms 3s and 3m are
    byte-identical but for one line, so block width cancels in their
    difference. That argument is about WIDTH. It says nothing about DENSITY -
    the fraction of the parity target that is real text rather than filler. If
    `reworded` padded to 60% text where `original` pads to 77%, the T contrast
    would carry a whitespace difference the algebra does not remove, and a
    template effect would be indistinguishable from a padding effect.

    Under CharTokenizer the templates match to within 3.2 points and the test
    suite pins that. But a BPE vocabulary is free to split "Your cumulative
    points" and "Opponent score" with quite different efficiency, and no CPU
    test can see it. This is the only place the property is checked against the
    tokeniser the run will actually use.

    Compared per block type at MATCHED states, because treatment density moves
    with the turn index (~6 points across the horizon) while the cross-template
    gap does not.
    """
    if builder.template_name == DEFAULT_STATE_TEMPLATE:
        print(f"  density     {builder.template_name} is the default "
              f"- no cross-template comparison to make")
        return

    reference = ScaffoldBuilder(builder.tokenizer, builder.config,
                                state_template=DEFAULT_STATE_TEMPLATE)
    key = EpisodeKey(f"{args.run_id}-density", 0, Arm.TREATMENT, args.model,
                     ReadoutMode(args.readout), OpponentPolicy.TFT)
    rounds = [r for r in (1, 5, 15) if r < game_config.horizon] or [1]

    worst = (0.0, "")
    for n_rounds in rounds:
        state = replay(game_config, build_opponent(OpponentPolicy.TFT, key),
                       [Action.COOPERATE] * n_rounds)
        for block in ("treatment_text", "nondiagnostic_text", "syntactic_text"):
            d_ref = len(reference.tokenizer.encode(
                getattr(reference, block)(state, framing))) / reference.block_tokens
            d_new = len(builder.tokenizer.encode(
                getattr(builder, block)(state, framing))) / builder.block_tokens
            gap = abs(d_new - d_ref)
            if gap > worst[0]:
                worst = (gap, f"{block} at {n_rounds} rounds: "
                              f"{builder.template_name} {d_new:.1%} vs "
                              f"{DEFAULT_STATE_TEMPLATE} {d_ref:.1%}")

    print(f"  density     worst cross-template gap {worst[0]:.1%} "
          f"(tolerance {TEMPLATE_DENSITY_TOLERANCE:.0%})  [{worst[1]}]")
    if worst[0] > TEMPLATE_DENSITY_TOLERANCE:
        # WARNS, DOES NOT ABORT - and the distinction is deliberate.
        #
        # This check is NOT one of PREREGISTRATION_EXP8.md section 7's eleven
        # gates. It was added after the pre-registration was frozen, so it has
        # no standing to block a registered factor. A post-hoc check that kills
        # a registered condition is a researcher degree of freedom wearing a
        # gate's clothes.
        #
        # What it does instead is MEASURE and RECORD. The gap is real: under
        # Llama-3.1 the reworded template's treatment block carries ~19% filler
        # against the original's ~6%, because the reworded syntactic body is the
        # longest of its three and drags the per-template parity target up. It
        # cannot be tuned away without resizing placebo bodies to hit density
        # targets across three different BPE vocabularies AFTER seeing that the
        # untuned version failed, which is instrument p-hacking.
        #
        # So the number goes in the log, and from the log into the paper, as a
        # measured limitation of the T factor. O and P are unaffected -
        # original_permuted matches original to 0.0% on every block type.
        print(f"\n  {'!' * 66}")
        print(f"  DENSITY WARNING - the T factor is confounded in this group.")
        print(f"  {builder.template_name!r} differs from {DEFAULT_STATE_TEMPLATE!r} "
              f"by {worst[0]:.1%} in filler fraction "
              f"(tolerance {TEMPLATE_DENSITY_TOLERANCE:.0%}).")
        print(f"  {worst[1]}")
        print(f"  Any T contrast reading this group carries a padding difference")
        print(f"  the token-parity argument does not remove. O and P contrasts")
        print(f"  within a single template are unaffected. REPORT THIS NUMBER.")
        print(f"  {'!' * 66}\n")


def _verify_history_condition(builder, assembler, backend, experiment,
                              game_config, framing, args) -> None:
    """Prove the history condition BEFORE any GPU time is spent on it.

    A --no-history run that silently still renders [HISTORY] is a null by
    construction, and it looks exactly like a real null in every downstream
    number. That failure mode has already cost this project three smoke runs;
    the fix is to render two prompts here, on the real tokeniser, and read them.

    Two independent properties are checked, because either alone can pass while
    the condition is broken:

      1. the header is present iff it is supposed to be;
      2. prompt LENGTH is turn-invariant iff history is off. History is the only
         section that grows with the turn index - rules, block (padded to the
         parity target) and instruction are all constant - so this is a check
         the whole run can be audited against later from `turns.prompt_tokens`,
         on every row rather than on the three episodes that store prompt_full.
    """
    include = not args.no_history
    readout = ReadoutMode(args.readout)
    key = EpisodeKey(f"{args.run_id}-selfcheck", 0, Arm.TREATMENT, args.model,
                     readout, OpponentPolicy.TFT)
    suffix = instruction_for(framing, readout, args.scratchpad_prompt)

    lengths, texts, rendered = [], [], []
    for n_rounds in (1, 5):
        state = replay(game_config, build_opponent(OpponentPolicy.TFT, key),
                       [Action.COOPERATE] * n_rounds)
        _, block = builder.build_pair(Arm.TREATMENT, state, framing)
        ids = assembler.assemble(
            game_config=game_config, state=state, framing=framing, block=block,
            instruction_suffix=suffix, include_history=include)
        lengths.append(len(ids))
        texts.append(backend.tokenizer.decode(list(ids)))
        rendered.append((state, block, list(ids)))

    print(f"  history     {'RENDERED' if include else 'REMOVED'}"
          f"   prompt {lengths[0]}/{lengths[1]} tokens at 1/5 rounds")

    for text in texts:
        if (HISTORY_HEADER in text) is not include:
            raise SystemExit(
                f"  ABORT: --no-history={not include} but {HISTORY_HEADER!r} "
                f"{'is' if not include else 'is not'} in the rendered prompt. "
                f"The condition did not apply and the run would be a null by "
                f"construction.")
    if include and lengths[0] == lengths[1]:
        raise SystemExit(
            "  ABORT: prompt length did not grow between round 1 and round 5 "
            "with history ON. The history section is not rendering rounds.")
    if not include and lengths[0] != lengths[1]:
        raise SystemExit(
            f"  ABORT: prompt length moved {lengths[0]} -> {lengths[1]} with "
            f"history OFF. Something else in the prompt still tracks the turn "
            f"index, so `turns.prompt_tokens` cannot audit this run.")
    if not include:
        # Cheap and worth saying out loud: with history gone, arm 1 carries no
        # state at all. That is the point (it becomes a real deprivation
        # condition) but it also means CPR in arm 1 should collapse, and a CPR
        # that does NOT collapse is evidence the model is reconstructing state
        # from somewhere this check has not found.
        print("  NOTE: arm 1 now contains NO state. Expect CPR(1) at the "
              "guessing floor;")
        print("        a high CPR(1) means state is leaking from another "
              "section.")

    _verify_template_and_position(builder, assembler, backend, game_config,
                                  framing, args, include, rendered)


def _verify_template_and_position(builder, assembler, backend, game_config,
                                  framing, args, include, rendered) -> None:
    """The exp8 pre-flight. Same contract as the history check above: prove the
    manipulation applied BEFORE renting anything, by reading a prompt rendered
    on the real tokeniser.

    Three properties, each of which can fail silently and each of which would
    make the run a null or a confound rather than an error:

      TEMPLATE  the active template's labels are all present and NO other
                registered template's labels are. A run that inherited the
                default template while its run_id said 'reworded' is a
                perfectly clean replication of exp6 mislabelled as a
                generalisation test.

      ORDER     the labels appear in the declared order. Permuting a tuple and
                forgetting to thread it through renders the canonical order
                with a permuted name on it.

      POSITION  the block's token IDs sit at the byte offset implied by
                insertion_index, computed from the section lengths rather than
                searched for. `assemble` inserts raw IDs and never re-encodes,
                so an exact slice match is available and anything weaker would
                pass while the block drifted.
    """
    template = builder.template
    own = set(template.labels)
    foreign = {lab for t in STATE_TEMPLATES.values() for lab in t.labels} - own

    for text in texts_of(rendered, backend):
        for label in template.labels:
            if f"{label}:" not in text:
                raise SystemExit(
                    f"  ABORT: template {template.name!r} declares field "
                    f"{label!r} but it is not in the rendered prompt. The "
                    f"template did not apply.")
        for label in sorted(foreign):
            if f"{label}:" in text:
                raise SystemExit(
                    f"  ABORT: template {template.name!r} is active but the "
                    f"foreign label {label!r} is in the rendered prompt. Two "
                    f"templates are rendering at once.")
        block_at = text.index(STATE_HEADER)
        positions = [text.index(f"{label}:", block_at) for label in template.labels]
        if positions != sorted(positions):
            raise SystemExit(
                f"  ABORT: template {template.name!r} declares field order "
                f"{template.field_order} but the prompt renders "
                f"{template.labels} out of order at offsets {positions}.")

    want_idx = assembler.block_section_index(include_history=include)
    for state, block, ids in rendered:
        rules = backend.tokenizer.encode(assembler._rules(game_config, framing))
        start = 0
        if want_idx >= 1:
            start += len(rules)
        if want_idx >= 2:
            start += len(backend.tokenizer.encode(
                assembler._history_section(state, framing)))
        got = ids[start:start + len(block.token_ids)]
        if got != list(block.token_ids):
            raise SystemExit(
                f"  ABORT: insertion_index={args.insertion_index} should place "
                f"the block at token offset {start}, but the {len(block.token_ids)} "
                f"tokens there are not the block. Position is the confound "
                f"cdx/config.py records as worth >30% swings on its own; a run "
                f"with the block somewhere unintended measures placement.")

    if include:
        for text in texts_of(rendered, backend):
            if args.insertion_index == 2 and text.index(STATE_HEADER) < text.index(HISTORY_HEADER):
                raise SystemExit(
                    "  ABORT: insertion_index=2 but [STATE] renders BEFORE "
                    "[HISTORY].")
            if args.insertion_index == 1 and text.index(STATE_HEADER) > text.index(HISTORY_HEADER):
                raise SystemExit(
                    "  ABORT: insertion_index=1 but [STATE] renders AFTER "
                    "[HISTORY].")

    where = {0: "before the rules",
             1: "after the rules, before [HISTORY]",
             2: "after [HISTORY], before the instruction"}[args.insertion_index]
    print(f"  template    OK  {template.name}  order "
          f"{' -> '.join(template.labels)}")
    print(f"  position    OK  block at section {want_idx} ({where}), "
          f"token offsets verified on 2 prompts")


def texts_of(rendered, backend):
    return [backend.tokenizer.decode(ids) for _, _, ids in rendered]


def run_cell(backend, builder, assembler, store, experiment, game_config,
             framing, arm, opponent, args, budget) -> dict:
    """Advance every episode in this cell in lockstep, one batched call per turn."""
    # Readout is part of the episode key, so seeds differ between LOGIT and
    # SCRATCHPAD runs. That is correct: they are different conditions, not two
    # views of the same episode.
    readout = ReadoutMode(args.readout)
    keys = [
        EpisodeKey(experiment.run_id, e, arm, args.model, readout, opponent)
        for e in range(args.episodes)
    ]
    games = [Game(game_config, build_opponent(opponent, k), k) for k in keys]
    turn_rows: dict[int, list[TurnRecord]] = defaultdict(list)
    donor_stats = DonorStats()

    agg = {"defect": [0, 0], "cpr": [0, 0], "off_task": [0, 0],
           "per_kind": defaultdict(lambda: [0, 0]),
           "by_turn": defaultdict(lambda: [0, 0])}

    for turn in range(game_config.horizon):
        budget.check()
        live = [(idx, k, g) for idx, (k, g) in enumerate(zip(keys, games))
                if g.should_continue()]
        if not live:
            break

        # Arm 3c renders the treatment template from ANOTHER episode's state.
        # Episodes advance in lockstep, so the donor pool is exactly the other
        # live episodes at this turn. Selection is seeded from the episode key
        # on its own RNG purpose, so it is reproducible and cannot shift the
        # actions it is meant to leave untouched.
        live_states = [g.state for _, _, g in live]

        prompts, seeds, blocks, donor_info = [], [], [], []
        for pos, (_, key, game) in enumerate(live):
            block = None
            donor_score, degenerate, shown_last = None, None, None
            if arm.injects_block:
                donor = None
                if arm is Arm.PLACEBO_STALE:
                    donor, degenerate = select_donor(
                        live_states, pos,
                        purpose_rng(key, f"donor{turn}"),
                        experiment.scaffold.max_donor_draws,
                    )
                    donor_stats.record(turn, degenerate)
                    # Persisted per row so the scaffold-echo question can be
                    # answered offline: a probe answer equal to THIS number
                    # rather than the true score proves the model read the
                    # block. Aggregates cannot show that.
                    donor_score = donor.agent_score if donor is not None else None
                _, block = builder.build_pair(arm, game.state, framing, donor=donor)

                # What the block ASSERTED, read off the same view that rendered
                # it rather than recomputed here. Recomputing is how exp1's
                # zero-padding defect survived: the log said one thing and the
                # prompt said another, and nothing compared them.
                #
                # Recorded only for arms that assert something false. A NULL
                # means "this arm told the truth", not "not recorded".
                if arm.falsifies_field:
                    if arm is Arm.PLACEBO_STALE:
                        view = donor
                    elif arm is Arm.PLACEBO_MOVE:
                        view = falsified_view(game.state, flip_move=True)
                    else:                                  # PLACEBO_SCORE
                        view = falsified_view(
                            game.state, score_offset=SCORE_FALSIFICATION)
                        donor_score = view.agent_score
                    if view is not None:
                        last = view.last_opponent_action()
                        shown_last = last.value if last is not None else None
                    # Arm 3m at turn 0 has no move to flip, so the block equals
                    # arm 3 and the row carries no falsification. Reuse
                    # donor_degenerate rather than adding a second flag - the
                    # analysis exclusion is identical.
                    if arm is Arm.PLACEBO_MOVE:
                        degenerate = int(not move_was_falsified(game.state))
            blocks.append(block)
            donor_info.append((donor_score,
                               None if degenerate is None else int(degenerate),
                               shown_last))
            prompts.append(assembler.assemble(
                game_config=game_config, state=game.state, framing=framing,
                block=block,
                instruction_suffix=instruction_for(
                    framing, readout, args.scratchpad_prompt),
                # Default True, so this is inert for exp1-exp6. The probe
                # prompts built in _probe_turn are these prompts plus a suffix,
                # so the condition carries into CPR automatically - which is
                # required: a CPR measured with the history restored would be
                # measuring a different prompt than the decision was.
                include_history=not args.no_history))
            seeds.append(purpose_rng(key, f"turn{turn}").getrandbits(63))

        decisions = _batched(
            lambda p, s: backend.decide_batch(p, readout, s, framing),
            prompts, seeds, args.batch_size)

        # Print one real scratchpad per cell. A defection rate cannot tell you
        # whether the model reasoned or emitted boilerplate; this can, and it
        # costs one line of output per cell.
        if turn == 0 and decisions and decisions[0].scratchpad:
            sample = " ".join(decisions[0].scratchpad.split())[:400]
            print(f"    scratchpad[0]: {sample}", flush=True)

        cpr_marks, cpr_detail, cpr_raw = {}, {}, {}
        if turn % max(args.probe_every, 1) == 0:
            cpr_marks, cpr_detail, cpr_raw = _probe_turn(
                backend, live, prompts, seeds, framing, agg, turn, args)

        for pos, ((idx, key, game), decision, block) in enumerate(
                zip(live, decisions, blocks)):
            agg["off_task"][1] += 1
            agg["off_task"][0] += int(decision.is_off_task)

            # Step FIRST, then record from the returned Turn. Asking the opponent
            # for its move separately would call it twice, which advances the
            # Q-learner's RNG and desynchronises the game from what was recorded.
            step = game.step(decision.action)
            opt_seq = _optimal_sequence(opponent, game_config)

            turn_rows[idx].append(TurnRecord(
                turn=turn,
                agent_action=step.agent_action,
                opponent_action=step.opponent_action,
                agent_payoff=step.agent_payoff,
                optimal_action=opt_seq[turn] if opt_seq and turn < len(opt_seq) else None,
                turn_regret=None,
                logit_mass_c=decision.logit_mass_cooperate,
                logit_mass_d=decision.logit_mass_defect,
                action_mass_total=decision.action_mass_total,
                logit_gap=decision.logit_gap,
                scaffold_tokens=block.n_tokens if block else None,
                scaffold_pad=block.pad_tokens_added if block else None,
                cpr_score=cpr_marks.get(pos),
                cpr_method=ProbeMethod.REPLAY.value if cpr_marks else None,
                cpr_own_score=cpr_detail.get(pos, {}).get("own_score"),
                cpr_opponent_last=cpr_detail.get(pos, {}).get("opponent_last"),
                cpr_rounds_played=cpr_detail.get(pos, {}).get("rounds_played"),
                turn_regret_calc=None,
                action_tokens_found=None,
                prompt_tokens=len(prompts[pos]),
                donor_agent_score=donor_info[pos][0],
                donor_degenerate=donor_info[pos][1],
                displayed_opponent_last=donor_info[pos][2],
                top_tokens=encode_top_tokens(decision.top_tokens) if decision.top_tokens else None,
                scratchpad=decision.scratchpad,
                # Raw probe text. Without it a CPR of 0 is undiagnosable later,
                # which is exactly the hole that cost three debugging rounds on
                # the laptop.
                probe_answers=json.dumps(cpr_raw.get(pos)) if cpr_raw else None,
                prompt_preview=_preview(backend, prompts[pos]),
                # Complete prompt for the first few episodes only. The preview
                # truncates the middle, which is where [STATE] sits once the
                # history grows - so on later turns it cannot show whether the
                # block rendered. Every prompt would be gigabytes; a bounded
                # sample is free and answers the question.
                prompt_full=(backend.tokenizer.decode(list(prompts[pos]))
                             if idx < args.full_prompt_episodes else None),
            ))
            agg["defect"][0] += int(decision.action is Action.DEFECT)
            agg["defect"][1] += 1

        print(f"    turn {turn:2d}  {len(live):5d} live  "
              f"defect {agg['defect'][0]/max(agg['defect'][1],1):.3f}  "
              f"cpr {agg['cpr'][0]/max(agg['cpr'][1],1):.3f}  "
              f"off-task {agg['off_task'][0]/max(agg['off_task'][1],1):.3f}  "
              f"({budget.remaining/60:.1f}m left)", flush=True)

    # ---- persist every episode -------------------------------------------
    trajectories, regrets = set(), []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for idx, (key, game) in enumerate(zip(keys, games)):
        trajectories.add("".join(a.value for a in game.state.agent_history))
        try:
            regrets.append(episode_regret(opponent, game_config,
                                          game.state.agent_history))
        except ValueError:
            pass
        store.write_episode(
            EpisodeRecord(
                key=key, model_revision=backend.model_config.revision,
                framing=framing, horizon_mode=game_config.horizon_mode,
                horizon=game_config.horizon,
                temperature=backend.model_config.temperature,
                config_fingerprint=experiment.fingerprint(),
                prompt_hash=PROBE_SUITE_HASH[:32],
                n_turns=len(game.state.turns),
                agent_score=game.state.agent_score,
                opponent_score=game.state.opponent_score,
                defection_count=sum(a is Action.DEFECT for a in game.state.agent_history),
                episode_regret=regrets[-1] if regrets else None,
            ),
            turn_rows[idx], now,
        )

    return {
        "defect_rate": agg["defect"][0] / max(agg["defect"][1], 1),
        "n_turns": agg["defect"][1],
        "n_episodes": args.episodes,
        "distinct_trajectories": len(trajectories),
        "cpr": agg["cpr"][0] / max(agg["cpr"][1], 1),
        "cpr_n": agg["cpr"][1],
        "per_kind": {k.value: v[0] / max(v[1], 1) for k, v in agg["per_kind"].items()},
        "by_turn": {str(t): v[0] / max(v[1], 1) for t, v in sorted(agg["by_turn"].items())},
        "off_task_rate": agg["off_task"][0] / max(agg["off_task"][1], 1),
        "mean_regret": sum(regrets) / len(regrets) if regrets else None,
        # Arm 3c only. A cell that fell back to an identical donor on most
        # turns did not test stale state on those turns, and its effect
        # estimate is diluted by exactly that fraction. Must be reported.
        **(donor_stats.summary() if donor_stats.total else {}),
    }


def _optimal_sequence(opponent, game_config):
    try:
        return solve(opponent, game_config).sequence
    except ValueError:
        return None


def _preview(backend, prompt_ids) -> str:
    """First and last 300 characters of what the model actually saw."""
    text = backend.tokenizer.decode(list(prompt_ids))
    return text if len(text) <= 640 else text[:300] + "\n...[snip]...\n" + text[-300:]


def _probe_turn(backend, live, prompts, seeds, framing, agg, turn, args):
    """All probes for all live episodes, batched per probe kind."""
    marks_by_pos = [dict() for _ in live]
    raw_by_pos = [dict() for _ in live]
    for spec in PROBE_SUITE:
        suffix = backend.tokenizer.encode(render_replay_probe(spec))
        probe_prompts = [list(p) + suffix for p in prompts]
        answers = _batched(backend.probe_batch, probe_prompts, seeds, args.batch_size)
        for pos, ((_, _, game), answer) in enumerate(zip(live, answers)):
            mark = score_answer(spec, answer, game.state, framing)
            marks_by_pos[pos][spec.kind] = mark
            raw_by_pos[pos][spec.kind.value] = {
                "got": answer[:120],
                "want": spec.truth(game.state, framing),
                "mark": mark,
            }
            agg["per_kind"][spec.kind][0] += mark
            agg["per_kind"][spec.kind][1] += 1

    cpr, detail, raw = {}, {}, {}
    for pos, marks in enumerate(marks_by_pos):
        passed = ProbeResult(ProbeMethod.REPLAY, marks).cpr
        cpr[pos] = passed
        detail[pos] = {k.value: v for k, v in marks.items()}
        raw[pos] = raw_by_pos[pos]
        agg["cpr"][0] += passed
        agg["cpr"][1] += 1
        agg["by_turn"][turn][0] += passed
        agg["by_turn"][turn][1] += 1
    return cpr, detail, raw


def _batched(fn, prompts, seeds, batch_size):
    """Chunk so a huge cell cannot exhaust GPU memory in one submission."""
    out = []
    for i in range(0, len(prompts), batch_size):
        out.extend(fn(prompts[i:i + batch_size], seeds[i:i + batch_size]))
    return out


def report(results, game_config, args, budget) -> None:
    print(f"\n{RULE}\nRESULTS  ({budget.elapsed/60:.1f} min elapsed)\n{RULE}")
    if not results:
        print("  no cells completed")
        return

    print(f"\n{'arm':6}{'opponent':10}{'defect':>9}{'cpr':>8}{'regret':>9}"
          f"{'distinct':>10}{'off-task':>10}")
    for name, r in sorted(results.items()):
        arm, opp = name.split("|")
        regret = f"{r['mean_regret']:.1f}" if r["mean_regret"] is not None else "n/a"
        print(f"{arm:6}{opp:10}{r['defect_rate']:>9.3f}{r['cpr']:>8.3f}{regret:>9}"
              f"{r['distinct_trajectories']:>10}{r['off_task_rate']:>10.4f}")

    # ---- instrument gates ------------------------------------------------
    print(f"\n{RULE}\nINSTRUMENT GATES\n{RULE}")
    bad = []
    for name, r in sorted(results.items()):
        if r["off_task_rate"] > 0.10:
            bad.append(f"{name}: off-task {r['off_task_rate']:.3f} > 0.10")
        if r["distinct_trajectories"] < max(2, r["n_episodes"] // 4):
            bad.append(f"{name}: only {r['distinct_trajectories']} distinct "
                       f"trajectories from {r['n_episodes']} episodes")
    for name, r in sorted(results.items()):
        rate = r.get("donor_degenerate_rate")
        if rate is not None and rate > 0.25:
            bad.append(
                f"{name}: donor identical to true state on {rate:.0%} of turns; "
                f"Arm 3c is diluted by that fraction"
            )
    print("  all gates pass" if not bad else "\n".join(f"  FAIL  {b}" for b in bad))
    if bad:
        print("\n  Results below are NOT interpretable until these are fixed.")

    for name, r in sorted(results.items()):
        if "donor_by_turn" in r:
            print(f"\n  {name} donor degeneracy by turn "
                  f"(share of episodes with no distinct donor):")
            print("    " + "  ".join(
                f"t{t}={v:.0%}" for t, v in r["donor_by_turn"].items()))
            print("    Turn 0 is always 100%: every episode starts at score 0")
            print("    with no last move, so no distinct donor can exist.")

    # ---- manipulation check ---------------------------------------------
    print(f"\n{RULE}\nMANIPULATION CHECK  CPR(3) - CPR(3b)\n{RULE}")
    for opp in args.opponents:
        t, c = results.get(f"3|{opp}"), results.get(f"3b|{opp}")
        if t and c:
            eff = t["cpr"] - c["cpr"]
            flag = "" if eff >= 0.05 else "   <-- scaffold not improving comprehension"
            print(f"  vs {opp:6} {t['cpr']:.3f} - {c['cpr']:.3f} = {eff:+.3f}{flag}")

    # ---- state deprivation ------------------------------------------------
    if getattr(args, "no_history", False):
        print(f"\n{RULE}\nSTATE DEPRIVATION CHECK  CPR(3) - CPR(1)   [--no-history]\n{RULE}")
        print("  With [HISTORY] gone, arm 1 carries NO state and arm 3 carries")
        print("  all of it, so this gap is the DOSE of the removal. Near zero")
        print("  means either the model reconstructs state from the rules")
        print("  section or the probes are answerable without state - in either")
        print("  case nothing below distinguishes the block from the raw log.")
        for opp in args.opponents:
            t, b = results.get(f"3|{opp}"), results.get(f"1|{opp}")
            if t and b:
                gap = t["cpr"] - b["cpr"]
                flag = "" if gap >= 0.30 else "   <-- removal did not bite"
                print(f"  vs {opp:6} {t['cpr']:.3f} - {b['cpr']:.3f} = {gap:+.3f}{flag}")
        print("  NOTE: CPR scores against the TRUE state, so arms 3c/3s/3m are")
        print("        expected at ~0.000 here. That is the manipulation")
        print("        working, not a gate failure (CLAIMS.md C6).")

    # ---- primary contrast ------------------------------------------------
    print(f"\n{RULE}\nPRIMARY CONTRAST  ATE_true = P(defect|3) - P(defect|3b)\n{RULE}")
    print(f"{'opponent':10}{'ATE_naive':>12}{'ATE_true':>11}{'95% CI':>20}"
          f"{'p':>9}{'pred':>7}{'obs':>7}")
    flip = {}
    for opp in args.opponents:
        need = [f"1|{opp}", f"3|{opp}", f"3b|{opp}"]
        if not all(k in results for k in need):
            continue
        a1, a3, a3b = (results[k] for k in need)
        d = ProportionDiff(a3["defect_rate"], a3b["defect_rate"],
                           a3["n_episodes"], a3b["n_episodes"])
        pred = predicted_defection_direction(OpponentPolicy(opp), game_config)
        obs = "down" if d.diff < 0 else "up"
        flip[opp] = (d, pred, obs)
        lo, hi = d.ci95
        print(f"{opp:10}{a3['defect_rate']-a1['defect_rate']:>+12.3f}{d.diff:>+11.3f}"
              f"{f'[{lo:+.3f},{hi:+.3f}]':>20}{d.p_value:>9.4f}{pred:>7}{obs:>7}"
              f"{'' if pred == obs else '   <-- MISMATCH'}")

    print(f"\n{RULE}\nSIGN-FLIP TEST (pre-registered)\n{RULE}")
    if len(flip) < 2:
        print("  INCOMPLETE - need both a retaliator and a pushover cell")
        return
    dt, pt, ot = flip.get("tft", (None, None, None))
    da, pa, oa = flip.get("allc", (None, None, None))
    if dt is None or da is None:
        print("  INCOMPLETE - expected tft and allc")
        return
    sig = dt.significant and da.significant
    opposite = dt.diff < 0 < da.diff
    matched = pt == ot and pa == oa
    if sig and opposite and matched:
        print("  HOLDS - both significant, opposite signs, both as predicted")
    elif not sig:
        print(f"  UNDERPOWERED - p(tft)={dt.p_value:.4f} p(allc)={da.p_value:.4f}")
        print(f"  MDE at n={args.episodes}: {min_detectable_effect(args.episodes):.3f}")
    elif not opposite:
        print("  FAILS - signs do not oppose; consistent with a prompt artifact")
    else:
        print("  FAILS - signs oppose but do not match the DP prediction")


if __name__ == "__main__":
    raise SystemExit(main())
