"""Comprehension-controlled measurement of defection in language agents.

Package layout:
    config.py    typed configuration; the only place magic numbers may live
    seeding.py   hash-derived episode seeds (no RNG state serialisation)
    game.py      deterministic IPD engine and scripted opponents
    optimal.py   exact optimal play, regret, sign-flip predictions
    scaffold.py  prompt assembly and token-ID-level parity enforcement
    backends.py  inference backends and the decision record
    db.py        SQLite persistence with crash-safe resume
    runner.py    episode loop
"""

__version__ = "0.1.0"
