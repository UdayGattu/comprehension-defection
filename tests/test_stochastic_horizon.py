"""Stochastic termination consumes exactly one draw per turn.

THE BUG THIS EXISTS TO PREVENT
    should_continue() used to draw from the termination RNG on every call, so
    the answer depended on how many times it was asked. Two callers ask per
    turn - step() at the end of a turn, and the batched runner when it rebuilds
    its live set at the start of the next. Every turn therefore consumed TWO
    draws, and the realised continuation probability was gamma^2 = 0.81 instead
    of the configured 0.90. Expected episode length collapsed from ~10 turns to
    ~5.3.

    Nothing caught it because nothing ran STOCHASTIC end to end. Phase 2 is
    specified to use stochastic termination precisely so backward induction
    does not trivially imply defection, so this would have corrupted the run
    that matters most - and the corruption is invisible: shorter episodes look
    like a modelling choice, not a bug.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdx.config import (
    Action,
    Arm,
    GameConfig,
    HorizonMode,
    OpponentPolicy,
    ReadoutMode,
)
from cdx.game import Game, build_opponent
from cdx.seeding import EpisodeKey

GAMMA = 0.9
HORIZON = 200          # well above the stochastic mean, so the cap rarely binds
EPISODES = 3000


def make(episode: int, mode: HorizonMode, horizon: int = HORIZON) -> Game:
    key = EpisodeKey("t", episode, Arm.BASELINE, "m", ReadoutMode.LOGIT,
                     OpponentPolicy.ALLC)
    cfg = GameConfig(horizon=horizon, horizon_mode=mode,
                     continuation_probability=GAMMA)
    return Game(cfg, build_opponent(OpponentPolicy.ALLC, key), key)


def play_out(game: Game) -> int:
    """Drive an episode exactly as the batched runner does: ask
    should_continue() to build the live set, then step."""
    turns = 0
    while game.should_continue():
        game.step(Action.COOPERATE)
        turns += 1
        if turns > HORIZON * 2:
            pytest.fail("episode failed to terminate")
    return turns


# --- purity ----------------------------------------------------------------


def test_should_continue_is_idempotent():
    """Asking twice must not change the answer, or the number of callers
    silently changes the experiment."""
    game = make(0, HorizonMode.STOCHASTIC)
    game.step(Action.COOPERATE)
    answers = [game.should_continue() for _ in range(20)]
    assert len(set(answers)) == 1


def test_extra_calls_do_not_shorten_episodes():
    """The exact shape of the bug: an extra caller consumed an extra draw."""
    a = play_out(make(7, HorizonMode.STOCHASTIC))

    game = make(7, HorizonMode.STOCHASTIC)
    turns = 0
    while game.should_continue():
        game.should_continue()      # a second caller, as the runner is
        game.should_continue()      # and a third
        game.step(Action.COOPERATE)
        turns += 1
    assert turns == a


# --- distribution ----------------------------------------------------------


def test_mean_episode_length_matches_the_configured_gamma():
    """Geometric with continuation gamma: E[turns] = 1 / (1 - gamma) = 10.

    Under the double-draw bug this came out near 1/(1-0.81) = 5.3, so the
    tolerance below is far tighter than the error it guards against.
    """
    lengths = [play_out(make(e, HorizonMode.STOCHASTIC)) for e in range(EPISODES)]
    mean = statistics.fmean(lengths)
    assert 9.0 <= mean <= 11.0, (
        f"mean episode length {mean:.2f}, expected ~10 for gamma={GAMMA}. "
        f"~5.3 indicates two draws per turn."
    )


def test_hard_cap_binds():
    """The cap exists so an episode cannot run unboundedly."""
    lengths = [play_out(make(e, HorizonMode.STOCHASTIC, horizon=5))
               for e in range(200)]
    assert max(lengths) <= 5


# --- the mode that is actually in use --------------------------------------


def test_known_horizon_is_unchanged():
    """Session 1 runs KNOWN. This fix must not touch it."""
    for episode in range(20):
        assert play_out(make(episode, HorizonMode.KNOWN, horizon=20)) == 20


def test_terminated_games_stay_terminated():
    game = make(3, HorizonMode.STOCHASTIC, horizon=3)
    play_out(game)
    assert game.state.terminated
    assert not game.should_continue()
    with pytest.raises(RuntimeError):
        game.step(Action.COOPERATE)