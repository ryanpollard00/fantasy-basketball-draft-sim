"""Tests for src/simulate.py, using a small fake player pool."""

import numpy as np
import pandas as pd
import pytest

import config
from src.simulate import FIELD_TEAMS, field_strategies, simulate_season


def make_pool(num_players=250, seed=0):
    rng = np.random.default_rng(seed)
    listings = [["PG", "SG"], ["SG", "SF"], ["SF", "PF"], ["PF", "C"], ["C"]]
    middle = rng.gamma(3, 700, num_players)
    return pd.DataFrame({
        "PLAYER_ID": range(num_players),
        "POSITIONS": [listings[i] for i in rng.choice(len(listings), num_players)],
        "PROJ_LAST_SEASON": middle * rng.normal(1, 0.2, num_players),
        "PROJ_MID": middle,
        "ACTUAL_PTS": middle * rng.normal(1, 0.3, num_players),
    })


@pytest.fixture
def two_strategies(monkeypatch):
    """Two strategies that rank identically, so any difference in results would be a bug."""
    monkeypatch.setattr(config, "STRATEGIES", {
        "model_a": {"value": "PROJ_MID", "ranking": "raw"},
        "model_b": {"value": "PROJ_MID", "ranking": "raw"},
    })


def test_field_has_eleven_opponents_who_disagree():
    pool = make_pool()
    field = field_strategies(pool, np.random.default_rng(0))
    assert len(field) == FIELD_TEAMS
    # Among closely ranked mid-tier players, noisy boards shouldn't all pick the same one
    mid_tier = pool.sort_values("PROJ_LAST_SEASON", ascending=False).iloc[100:150]
    choices = {pick(None, mid_tier) for pick in field.values()}
    assert len(choices) > 1


def test_same_seed_gives_the_same_field():
    pool = make_pool()
    first = field_strategies(pool, np.random.default_rng(5))
    second = field_strategies(pool, np.random.default_rng(5))
    for name in first:
        assert first[name](None, pool) == second[name](None, pool)


def test_every_seat_is_used_equally(two_strategies):
    results = simulate_season(make_pool(), "2025-26", num_drafts=config.NUM_TEAMS)
    assert len(results) == config.NUM_TEAMS * 2
    for _, rows in results.groupby("STRATEGY"):
        assert sorted(rows["DRAFT_POSITION"]) == list(range(1, config.NUM_TEAMS + 1))
    assert results["FINISH"].between(1, config.NUM_TEAMS).all()


def test_strategies_face_identical_leagues(two_strategies):
    # Two strategies with the same board must get exactly the same results in every draft
    results = simulate_season(make_pool(), "2025-26", num_drafts=6)
    a = results[results["STRATEGY"] == "model_a"].reset_index(drop=True)
    b = results[results["STRATEGY"] == "model_b"].reset_index(drop=True)
    pd.testing.assert_series_equal(a["SCORE"], b["SCORE"])
    pd.testing.assert_series_equal(a["EDGE_PCT"], b["EDGE_PCT"])


def test_results_are_reproducible(two_strategies):
    first = simulate_season(make_pool(), "2025-26", num_drafts=4)
    second = simulate_season(make_pool(), "2025-26", num_drafts=4)
    pd.testing.assert_frame_equal(first, second)