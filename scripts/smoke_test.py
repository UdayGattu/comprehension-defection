#!/usr/bin/env python3
"""End-to-end smoke test on the dummy backend. No GPU, no model weights.

Exercises the full pipeline including the sign-flip analysis, so the plumbing is
proven before any money is spent. The numbers it prints are meaningless — the
dummy backend is a stub, not a model. What is being verified is that every stage
runs, persists, and can be resumed.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.backends import CharTokenizer, DummyBackend
from cdx.config import (
    Action,
    Arm,
    ExperimentConfig,
    Framing,
    GameConfig,
    OpponentPolicy,
    ReadoutMode,
)
from cdx.db import Store
from cdx.optimal import summary_table
from cdx.runner import Cell, Runner

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

N_EPISODES = 25
ARMS = [Arm.BASELINE, Arm.TREATMENT, Arm.PLACEBO_NONDIAGNOSTIC, Arm.PLACEBO_STALE]
OPPONENTS = [OpponentPolicy.TFT, OpponentPolicy.ALLC]


def main() -> int:
    print("\n--- optimal play (exact DP) ---")
    print(f"{'opponent':9}{'optimal':>9}{'ALLD':>7}{'regret':>8}{'predicted':>11}")
    for row in summary_table(GameConfig()):
        print(
            f"{row['opponent']:9}{row['optimal']:>9}{row['alld']:>7}"
            f"{row['regret_of_alld']:>8}{row['predicted_direction']:>11}"
        )

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "smoke.db")
        experiment = ExperimentConfig(run_id="smoke")
        runner = Runner(experiment, DummyBackend(), CharTokenizer(), store)
        runner.seed_donor_pool()

        cells = [
            Cell(arm, opp, ReadoutMode.LOGIT, Framing.SEMANTIC, N_EPISODES)
            for arm in ARMS
            for opp in OPPONENTS
        ]
        stats = runner.run_cells(cells, model_id="dummy/deterministic")
        print(f"\n--- runner: {stats} ---")

        print("\n--- defection rate by arm and opponent (dummy backend) ---")
        rows = store._conn.execute(
            """
            SELECT opponent_policy, arm,
                   AVG(CAST(agent_action = 'D' AS FLOAT)) AS defect_rate,
                   COUNT(*) AS n
            FROM turns GROUP BY opponent_policy, arm ORDER BY opponent_policy, arm
            """
        ).fetchall()
        print(f"{'opponent':10}{'arm':6}{'defect':>9}{'turns':>8}")
        for opp, arm, rate, n in rows:
            print(f"{opp:10}{arm:6}{rate:>9.3f}{n:>8}")

        print("\n--- ATE vs matched placebo (headline contrast) ---")
        for opp in ("tft", "allc"):
            got = {
                arm: rate
                for o, arm, rate, _ in rows
                if o == opp and arm in ("3", "3b")
            }
            if {"3", "3b"} <= got.keys():
                ate = got["3"] - got["3b"]
                print(f"  vs {opp:5} ATE_true = {ate:+.3f}")

        print("\n--- off-task and reproducibility diagnostics ---")
        diag = store._conn.execute(
            "SELECT AVG(CAST(action_mass_total < 0.1 AS FLOAT)), "
            "       AVG(CAST(logit_gap < 1e-4 AS FLOAT)), COUNT(*) FROM turns"
        ).fetchone()
        print(f"  off-task rate        {diag[0]:.4f}")
        print(f"  near-tie decisions   {diag[1]:.4f}  (reproducibility-fragile share)")
        print(f"  total decisions      {diag[2]}")

        store.close()

    print("\nSmoke test complete. Numbers above are from a stub backend and mean nothing.")
    print("What is verified: engine, parity, persistence, resume, analysis path.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
