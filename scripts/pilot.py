#!/usr/bin/env python3
"""LOCAL PILOT — the go/no-go gate before renting a GPU.

Runs real episodes on Apple Silicon via MLX and reports the one number that
decides whether the project is alive: the comprehension pass rate.

WHAT THIS CAN AND CANNOT TELL YOU
---------------------------------
CAN:    CPR. Measured per turn, so even 15 episodes gives hundreds of
        observations - ample to separate "CPR is 30%" from "CPR is 95%".

CANNOT: the ATE. That needs N=1,600 per arm. At N=15 the minimum detectable
        effect is ~51 pp. Read CPR, ignore the ATE.

NEVER:  produce numbers for the paper. This runs 4-bit quantised; quantisation
        is a confound the design controls for. Production is bf16 on rented
        hardware.

Probing happens on a BRANCH of the same prompt. The probe forward pass never
re-enters the game, so it cannot influence the action it measures.

    python scripts/pilot.py --episodes 15 --arms 1 --trace 1
    python scripts/pilot.py --episodes 15 --arms 1 3 --probe-every 2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.analysis import min_detectable_effect, wilson_interval
from cdx.config import (
    Action,
    Arm,
    ExperimentConfig,
    ScaffoldConfig,
    Framing,
    GameConfig,
    OpponentPolicy,
    ReadoutMode,
)
from cdx.game import Game, build_opponent
from cdx.optimal import episode_regret, solve
from cdx.probe import (
    PROBE_SUITE,
    PROBE_SUITE_HASH,
    ProbeKind,
    ProbeMethod,
    ProbeResult,
    render_replay_probe,
    score_answer,
)
from cdx.runner import _INSTRUCTION
from cdx.scaffold import PromptAssembler, ScaffoldBuilder
from cdx.seeding import EpisodeKey, purpose_rng

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

ARMS = [Arm.BASELINE, Arm.TREATMENT, Arm.PLACEBO_NONDIAGNOSTIC]
OPPONENTS = [OpponentPolicy.TFT, OpponentPolicy.ALLC]
CPR_GO_THRESHOLD = 0.90
OFF_TASK_ABORT = 0.5
RULE = "=" * 74


class Counter:
    """Running tallies with a live view, so problems surface during the run
    rather than twenty minutes later in the final report."""

    def __init__(self) -> None:
        self.defect = defaultdict(lambda: [0, 0])
        self.cpr = defaultdict(lambda: [0, 0])
        self.per_kind = defaultdict(lambda: [0, 0])
        self.by_turn = defaultdict(lambda: [0, 0])
        self.off_task = [0, 0]
        self.regrets = defaultdict(list)
        self.decisions = 0
        # Distinct action trajectories per cell. If this is 1 while N is 15,
        # every episode is a copy and the sample size is a fiction.
        self.trajectories = defaultdict(set)

    @property
    def off_task_rate(self) -> float:
        return self.off_task[0] / max(self.off_task[1], 1)

    @property
    def cpr_rate(self) -> float:
        p = sum(v[0] for v in self.cpr.values())
        n = sum(v[1] for v in self.cpr.values())
        return p / max(n, 1)

    @property
    def cpr_counts(self) -> tuple[int, int]:
        return (
            sum(v[0] for v in self.cpr.values()),
            sum(v[1] for v in self.cpr.values()),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--model", default=None)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--framing", default=Framing.SEMANTIC.value)
    parser.add_argument("--probe-every", type=int, default=2,
                        help="Probe every Nth turn. Probing costs ~3x an action "
                             "decision, so this is the main runtime lever.")
    parser.add_argument("--arms", nargs="*", default=[a.value for a in ARMS])
    parser.add_argument("--opponents", nargs="*",
                        default=[o.value for o in OPPONENTS],
                        help="Subset of opponents. ALLC alone is the decisive "
                             "cell: optimal play there REQUIRES defection.")
    parser.add_argument("--objective", default="self_interest",
                        choices=["none", "self_interest", "joint"],
                        help="Whether the prompt states a goal. 'none' was the "
                             "original behaviour and produced 0%% defection.")
    parser.add_argument("--swap-labels", action="store_true",
                        help="Invert which label means which action. If choices "
                             "follow the label rather than the meaning, the model "
                             "has position bias, not a strategy.")
    parser.add_argument("--trace", type=int, default=1,
                        help="Print full turn-by-turn detail for the first N "
                             "episodes of each cell. 0 disables.")
    parser.add_argument("--no-abort", action="store_true",
                        help="Do not stop early on a high off-task rate.")
    args = parser.parse_args()
    selected_arms = [Arm(a) for a in args.arms]
    selected_opponents = [OpponentPolicy(o) for o in args.opponents]
    framing = Framing(args.framing)

    from cdx.backends_mlx import DEFAULT_MLX_MODEL, MLXBackend
    from cdx.config import ModelConfig

    model_id = args.model or DEFAULT_MLX_MODEL
    print(f"\n{RULE}\nSTEP 1  load model\n{RULE}")
    print(f"  model    {model_id}")
    t0 = time.time()
    backend = MLXBackend(ModelConfig(model_id=model_id),
                         swap_labels=args.swap_labels)
    print(f"  loaded   {time.time() - t0:.1f}s")

    game_config = GameConfig(horizon=args.horizon)
    experiment = ExperimentConfig(
        run_id="pilot", game=game_config, probe_text_hash=PROBE_SUITE_HASH,
        scaffold=ScaffoldConfig(swap_action_labels=args.swap_labels,
                                objective=args.objective),
    )
    builder = ScaffoldBuilder(backend.tokenizer, experiment.scaffold)
    assembler = PromptAssembler(backend.tokenizer, experiment.scaffold)

    print(f"\n{RULE}\nSTEP 2  verify instrument\n{RULE}")
    print(f"  framing          {framing.value}"
          f"{'  LABELS SWAPPED' if args.swap_labels else ''}")
    print(f"  objective        {args.objective}")
    for action in (Action.COOPERATE, Action.DEFECT):
        ids = backend._action_token_ids[framing][action]
        decoded = [repr(backend.tokenizer.decode([i])) for i in ids]
        print(f"  {action.value} action tokens   {ids} -> {', '.join(decoded)}")
    print(f"  scaffold filler  {backend.tokenizer.encode(builder.filler_text)}"
          f" ({builder.filler_text!r})")
    print(f"  probe hash       {PROBE_SUITE_HASH[:24]}...")
    print("  NOTE: if the action tokens above do not match what the prompt asks")
    print("        for, every downstream number is meaningless.")

    cells = [(arm, opp) for arm in selected_arms for opp in selected_opponents]
    est = len(cells) * args.episodes * args.horizon * (1 + 3 / max(args.probe_every, 1))
    print(f"\n{RULE}\nSTEP 3  run episodes\n{RULE}")
    print(f"  cells     {len(cells)}  ({', '.join(f'{a.value}/{o.value}' for a, o in cells)})")
    print(f"  episodes  {args.episodes} per cell x {args.horizon} turns")
    print(f"  calls     ~{est:,.0f}   trace={args.trace}  probe-every={args.probe_every}")

    counts = Counter()
    started = time.time()

    for cell_i, (arm, opponent) in enumerate(cells, start=1):
        print(f"\n  --- cell {cell_i}/{len(cells)}: arm={arm.value} opponent={opponent.value} ---")
        for episode_id in range(args.episodes):
            tracing = episode_id < args.trace
            _run_episode(
                backend, builder, assembler, counts, game_config, framing,
                arm, opponent, episode_id, model_id, args, tracing,
            )

            done = sum(counts.defect[k][1] for k in counts.defect)
            rate = counts.decisions / max(time.time() - started, 1e-9)
            p, n = counts.cpr_counts
            print(
                f"    ep {episode_id + 1:>3}/{args.episodes}  "
                f"CPR {p}/{n} ({counts.cpr_rate:.2f})  "
                f"off-task {counts.off_task_rate:.2f}  "
                f"{rate:.2f} dec/s",
                flush=True,
            )

            if (not args.no_abort and counts.off_task[1] >= args.horizon
                    and counts.off_task_rate > OFF_TASK_ABORT):
                print(f"\n  ABORTING: off-task rate {counts.off_task_rate:.2f} "
                      f"exceeds {OFF_TASK_ABORT}.")
                print("  The readout is not seeing valid action tokens. Continuing")
                print("  would waste time producing meaningless numbers.")
                print("  Re-run with --no-abort to collect the full run anyway.")
                _report(model_id, args, counts, game_config, time.time() - started)
                return 1

    _report(model_id, args, counts, game_config, time.time() - started)
    return 0


def _run_episode(backend, builder, assembler, counts, game_config, framing,
                 arm, opponent, episode_id, model_id, args, tracing) -> None:
    key = EpisodeKey(
        run_id="pilot", episode_id=episode_id, arm=arm, model_id=model_id,
        readout_mode=ReadoutMode.LOGIT, opponent=opponent,
    )
    game = Game(game_config, build_opponent(opponent, key), key)

    if tracing:
        print(f"\n    +-- TRACE episode {episode_id} "
              f"(arm={arm.value} vs {opponent.value}) " + "-" * 18)

    while game.should_continue():
        turn = game.state.turn_index
        block = None
        if arm.injects_block:
            _, block = builder.build_pair(arm, game.state, framing)

        prompt_ids = assembler.assemble(
            game_config=game_config, state=game.state, framing=framing,
            block=block, instruction_suffix=_INSTRUCTION[framing],
        )
        seed = purpose_rng(key, f"turn{turn}").getrandbits(63)
        decision = backend.decide(prompt_ids, ReadoutMode.LOGIT, seed, framing=framing)
        counts.decisions += 1
        counts.off_task[1] += 1
        counts.off_task[0] += int(decision.is_off_task)

        if tracing:
            last = game.state.last_opponent_action()
            print(f"    | t{turn:<2} state    score {game.state.agent_score}-"
                  f"{game.state.opponent_score}, opp last "
                  f"{last.value if last else '-'}, prompt {len(prompt_ids)} tok")
            flag = "  <-- OFF-TASK" if decision.is_off_task else ""
            print(f"    |     action   {decision.action.value}  "
                  f"P(C)={decision.logit_mass_cooperate:.3f} "
                  f"P(D)={decision.logit_mass_defect:.3f} "
                  f"mass={decision.action_mass_total:.3f}{flag}")
            if decision.is_off_task:
                top = ", ".join(f"{t!r}:{p:.2f}" for t, p in decision.top_tokens[:3])
                print(f"    |     model wanted: {top}")

        if turn % max(args.probe_every, 1) == 0:
            marks: dict[ProbeKind, int] = {}
            for spec in PROBE_SUITE:
                probe_ids = list(prompt_ids) + backend.tokenizer.encode(
                    render_replay_probe(spec)
                )
                answer = backend.probe(probe_ids, seed)
                mark = score_answer(spec, answer, game.state, framing)
                marks[spec.kind] = mark
                counts.per_kind[spec.kind][0] += mark
                counts.per_kind[spec.kind][1] += 1
                if tracing:
                    want = spec.truth(game.state, framing)
                    print(f"    |     probe    {spec.kind.value:15}"
                          f"want={want!r:12} got={answer[:32]!r:36}"
                          f"{'OK' if mark else 'WRONG'}")

            passed = ProbeResult(ProbeMethod.REPLAY, marks).cpr
            counts.cpr[(arm.value, opponent.value)][0] += passed
            counts.cpr[(arm.value, opponent.value)][1] += 1
            counts.by_turn[turn][0] += passed
            counts.by_turn[turn][1] += 1
            if tracing:
                print(f"    |     CPR      {'PASS' if passed else 'FAIL'}")

        game.step(decision.action)
        d = counts.defect[(arm.value, opponent.value)]
        d[0] += int(decision.action is Action.DEFECT)
        d[1] += 1

    counts.trajectories[(arm.value, opponent.value)].add(
        "".join(a.value for a in game.state.agent_history)
    )

    try:
        counts.regrets[(arm.value, opponent.value)].append(
            episode_regret(opponent, game_config, game.state.agent_history)
        )
    except ValueError:
        pass

    if tracing:
        defections = sum(a is Action.DEFECT for a in game.state.agent_history)
        print(f"    +-- end: score {game.state.agent_score}, "
              f"defected {defections}/{len(game.state.turns)}\n")


def _report(model_id, args, counts, game_config, elapsed) -> None:
    print(f"\n{RULE}\nSTEP 4  results — {model_id}")
    print(f"{args.episodes} episodes/cell, horizon {args.horizon}, "
          f"elapsed {elapsed/60:.1f} min\n{RULE}")

    p, n = counts.cpr_counts
    lo, hi = wilson_interval(p, n)
    print("\nCOMPREHENSION PASS RATE (the number that matters)")
    print(f"  overall  {counts.cpr_rate:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   n={n} turns")

    print("\n  by probe component (which part fails?):")
    for kind, (kp, kn) in counts.per_kind.items():
        klo, khi = wilson_interval(kp, kn)
        print(f"    {kind.value:16}{kp/max(kn,1):.3f}  [{klo:.3f}, {khi:.3f}]  n={kn}")

    print("\n  by turn index (gradual decay, or a suspicious cliff?):")
    for turn in sorted(counts.by_turn):
        tp, tn = counts.by_turn[turn]
        print(f"    turn {turn:2d}  {tp/max(tn,1):.3f}   n={tn}")

    print("\nCPR BY ARM (the manipulation check)")
    print(f"  {'arm':6}{'opponent':10}{'CPR':>8}{'95% CI':>18}{'n':>7}")
    by_arm = defaultdict(lambda: [0, 0])
    for (arm, opp), (p_, n_) in sorted(counts.cpr.items()):
        lo_, hi_ = wilson_interval(p_, n_)
        print(f"  {arm:6}{opp:10}{p_/max(n_,1):>8.3f}"
              f"{f'[{lo_:.3f},{hi_:.3f}]':>18}{n_:>7}")
        by_arm[arm][0] += p_; by_arm[arm][1] += n_
    if len(by_arm) > 1:
        print(f"\n  pooled per arm:")
        for arm in sorted(by_arm):
            p_, n_ = by_arm[arm]
            print(f"    arm {arm:4} {p_/max(n_,1):.3f}  (n={n_})")
        t = by_arm.get("3"); c = by_arm.get("3b")
        if t and c:
            eff = t[0]/max(t[1],1) - c[0]/max(c[1],1)
            print(f"\n  MANIPULATION EFFECT  CPR(3) - CPR(3b) = {eff:+.3f}")
            if eff < 0.05:
                print("    WARNING: the scaffold is not measurably improving")
                print("    comprehension. ATE_true is then not measuring")
                print("    comprehension repair, whatever else it measures.")

    print("\nDEFECTION RATE (underpowered — context only)")
    print(f"  {'arm':6}{'opponent':10}{'defect':>8}{'mean regret':>13}")
    for (arm, opp), (d, dn) in sorted(counts.defect.items()):
        reg = counts.regrets.get((arm, opp), [])
        reg_str = f"{sum(reg)/len(reg):.1f}" if reg else "n/a"
        print(f"  {arm:6}{opp:10}{d/max(dn,1):>8.3f}{reg_str:>13}")

    print("\n  optimal for reference:")
    for opp in [OpponentPolicy(o) for o in args.opponents]:
        o = solve(opp, game_config)
        print(f"    vs {opp.value:6} optimal={o.value:4} "
              f"optimal defect rate={o.defection_rate:.2f}")

    print("\nSAMPLE INDEPENDENCE (are episodes actually distinct?)")
    print(f"  {'arm':6}{'opponent':10}{'distinct':>10}{'episodes':>10}{'effective N':>13}")
    degenerate = []
    for (arm, opp), traj in sorted(counts.trajectories.items()):
        n_ep = args.episodes
        frac = len(traj) / max(n_ep, 1)
        print(f"  {arm:6}{opp:10}{len(traj):>10}{n_ep:>10}{frac:>12.0%}")
        if len(traj) < max(2, n_ep // 4):
            degenerate.append((arm, opp, len(traj)))
    if degenerate:
        print("\n  WARNING: episodes are near-identical. Confidence intervals")
        print("  computed over turns are FICTION - the effective sample size is")
        print("  the number of distinct trajectories, not the number of turns.")

    print("\nDIAGNOSTICS")
    print(f"  off-task rate  {counts.off_task_rate:.4f}  "
          f"(turns where the model put <10% mass on any valid action)")
    print(f"  MDE at n={args.episodes}: {min_detectable_effect(args.episodes):.3f} "
          f"— the ATE above is NOT interpretable")

    print(f"\n{RULE}\nVERDICT\n{RULE}")

    # INSTRUMENT VALIDITY FIRST. A high off-task rate means the readout never saw
    # the action tokens, so every action is the tie-break firing rather than a
    # decision. The first pilot run printed GO on data where off-task was 1.0000;
    # that ordering bug is why this check now comes first.
    if counts.off_task_rate > OFF_TASK_ABORT:
        print(f"  INVALID — off-task rate {counts.off_task_rate:.4f}")
        print("  The model never placed meaningful probability on a valid action")
        print("  token. Every action is the tie-break, not a decision, and the CPR")
        print("  reading cannot be trusted either.")
        print("\n  Check in order:")
        print("    1. STEP 2 above — do the action tokens match what the prompt asks for?")
        print("    2. Is the chat template applied? Instruct models need it.")
        print("    3. Read the TRACE 'model wanted:' lines — what IS it emitting?")
        print("\n  DO NOT RENT A GPU ON THIS RUN.")
        return

    if hi < CPR_GO_THRESHOLD:
        print(f"  GO. CPR upper bound {hi:.3f} < {CPR_GO_THRESHOLD}")
        print("  A real comprehension deficit exists, so the scaffold intervention")
        print("  has room to act. Next: --arms 1 3 to confirm the scaffold repairs")
        print("  it, then rent the GPU.")
    elif lo > CPR_GO_THRESHOLD:
        print(f"  CAUTION. CPR lower bound {lo:.3f} > {CPR_GO_THRESHOLD}")
        print("  These models track state well at this horizon, so the headline")
        print("  effect is likely small. Lengthen the horizon, harden the probes,")
        print("  or pivot to the protocol framing BEFORE spending money.")
    else:
        print(f"  INCONCLUSIVE. CI [{lo:.3f}, {hi:.3f}] straddles {CPR_GO_THRESHOLD}")
        print("  Raise --episodes and re-run. Cheap compared to a GPU hour.")

    cliff = _detect_cliff(counts.by_turn)
    if cliff is not None:
        print(f"\n  NOTE: CPR drops sharply at turn {cliff}. Real comprehension decay")
        print("  is gradual; a clean step is usually an instrument artifact — check")
        print("  whether probe answers are being truncated before the number appears.")


def _detect_cliff(by_turn) -> int | None:
    turns = sorted(by_turn)
    for a, b in zip(turns, turns[1:]):
        ra = by_turn[a][0] / max(by_turn[a][1], 1)
        rb = by_turn[b][0] / max(by_turn[b][1], 1)
        if ra > 0.8 and rb < 0.2:
            return b
    return None


if __name__ == "__main__":
    raise SystemExit(main())
