"""Tests for src/strategies.py, using small hand-made player pools."""

import numpy as np
import pandas as pd
import pytest

import config
from src.draft import Team, run_draft
from src.strategies import (
    POSITIONS,
    build_strategies,
    make_strategy,
    position_demand,
    replacement_levels,
    strategy_scores,
    vorp,
)


def ladder_pool(players_per_position=30):
    """At each position, single-position players worth 3000, 2990, 2980, ... (10 apart)."""
    rows = []
    for position in POSITIONS:
        for rank in range(players_per_position):
            rows.append({"POSITIONS": [position], "PROJ_MID": 3000 - 10 * rank})
    pool = pd.DataFrame(rows)
    pool["PLAYER_ID"] = range(len(pool))
    return pool


def strategy_pool(num_players=300, seed=0):
    """A fake draft pool with every projection column the strategies read."""
    rng = np.random.default_rng(seed)
    listings = [["PG", "SG"], ["SG", "SF"], ["SF", "PF"], ["PF", "C"], ["C"]]
    middle = rng.gamma(3, 700, num_players)
    return pd.DataFrame({
        "PLAYER_ID": range(num_players),
        "POSITIONS": [listings[i] for i in rng.choice(len(listings), num_players)],
        "PROJ_LAST_SEASON": middle * rng.normal(1, 0.2, num_players),
        "PROJ_AVG3": middle * rng.normal(1, 0.15, num_players),
        "PROJ_FLOOR": middle * rng.uniform(0.6, 0.95, num_players),
        "PROJ_MID": middle,
        "PROJ_CEILING": middle * rng.uniform(1.05, 1.4, num_players),
        "ACTUAL_PTS": middle * rng.normal(1, 0.3, num_players),
    })


def test_demand_covers_every_starting_slot():
    demand = position_demand()
    assert demand.sum() == pytest.approx(config.NUM_TEAMS * len(config.STARTING_SLOTS))
    assert demand["C"] == pytest.approx(14.4)      # 12 from C, 2.4 from UTIL
    assert demand["PG"] == pytest.approx(20.4)     # 12 from PG, 6 from G, 2.4 from UTIL


def test_demand_replacement_levels(monkeypatch):
    monkeypatch.setattr(config, "REPLACEMENT_METHOD", "demand")
    levels = replacement_levels(ladder_pool(), "PROJ_MID")
    assert levels["PG"] == 3000 - 10 * 20          # 21st-best guard (demand 20)
    assert levels["C"] == 3000 - 10 * 14           # 15th-best center (demand 14)


def test_fixed_replacement_levels(monkeypatch):
    monkeypatch.setattr(config, "REPLACEMENT_METHOD", "fixed")
    levels = replacement_levels(ladder_pool(), "PROJ_MID")
    assert (levels == 3000 - 10 * config.REPLACEMENT_RANK).all()


def test_vorp_uses_the_players_best_position(monkeypatch):
    monkeypatch.setattr(config, "REPLACEMENT_METHOD", "demand")
    pool = ladder_pool()
    pool.loc[len(pool)] = {"POSITIONS": ["PG", "C"], "PROJ_MID": 3100, "PLAYER_ID": 999}
    levels = replacement_levels(pool, "PROJ_MID")
    best_level = min(levels["PG"], levels["C"])
    assert vorp(pool, "PROJ_MID").iloc[-1] == 3100 - best_level


def test_raw_strategy_scores_are_the_projection():
    pool = strategy_pool()
    scores = strategy_scores(pool)
    pd.testing.assert_series_equal(scores["best_available"], pool["PROJ_MID"], check_names=False)
    pd.testing.assert_series_equal(scores["naive_recency"], pool["PROJ_LAST_SEASON"], check_names=False)


def test_strategies_never_use_actual_points():
    # Scramble actual points; no strategy's scores may change
    pool = strategy_pool()
    scrambled = pool.copy()
    scrambled["ACTUAL_PTS"] = np.random.default_rng(1).permutation(scrambled["ACTUAL_PTS"].to_numpy()) * 10
    pd.testing.assert_frame_equal(strategy_scores(pool), strategy_scores(scrambled))


def test_strategy_takes_the_best_allowed_player():
    pool = strategy_pool()
    scores = strategy_scores(pool)
    pick = make_strategy(scores, "best_available")
    allowed = pool.drop(index=scores["best_available"].idxmax())   # top player already taken
    expected = pool.loc[scores["best_available"].drop(index=scores["best_available"].idxmax()).idxmax(), "PLAYER_ID"]
    assert pick(None, allowed) == expected


def test_every_strategy_drafts_legally():
    pool = strategy_pool()
    seats = list(config.STRATEGIES) * 2    # 12 teams, two per strategy
    teams = [Team(f"Team {i + 1}", name) for i, name in enumerate(seats)]
    picks = run_draft(teams, pool, build_strategies(pool))
    assert len(picks) == config.NUM_TEAMS * config.ROSTER_SIZE
    assert picks["PLAYER_ID"].is_unique
    assert all(team.is_full() for team in teams)
    assert set(picks["STRATEGY"]) == set(config.STRATEGIES)