"""Tests for src/lineup.py and the walk-forward season split."""

import numpy as np
import pandas as pd
import pytest

import config
from src.draft import Team, best_projection, run_draft
from src.features import split_by_season
from src.lineup import best_lineup, score_draft

GUARD, WING, FORWARD, BIG, CENTER = ["PG", "SG"], ["SG", "SF"], ["SF", "PF"], ["PF", "C"], ["C"]


def make_roster(players):
    """A roster table from (positions, points) pairs."""
    return pd.DataFrame({
        "POSITIONS": [positions for positions, _ in players],
        "ACTUAL_PTS": [points for _, points in players],
    })


def test_best_eight_start_when_positions_allow():
    roster = make_roster([
        (GUARD, 5000), (GUARD, 4000), (GUARD, 3000), (GUARD, 100),
        (WING, 4500),
        (FORWARD, 4200), (FORWARD, 3500), (FORWARD, 200),
        (BIG, 3800),
        (CENTER, 3600), (CENTER, 300), (CENTER, 250),
    ])
    slots, total = best_lineup(roster)
    assert total == 5000 + 4500 + 4200 + 4000 + 3800 + 3600 + 3500 + 3000
    assert set(slots[roster["ACTUAL_PTS"] < 1000]) == {"BENCH"}


def test_only_center_must_start_even_when_weak():
    roster = make_roster(
        [(GUARD, 4000 - i * 100) for i in range(6)]
        + [(FORWARD, 3000 - i * 100) for i in range(5)]
        + [(CENTER, 500)]
    )
    slots, _ = best_lineup(roster)
    assert slots.iloc[-1] == "C"    # the weak center starts; no one else can play C


def test_roster_that_cannot_fill_the_lineup_raises_error():
    roster = make_roster([(GUARD, 3000)] * 12)    # no one can play SF, PF, F or C
    with pytest.raises(ValueError):
        best_lineup(roster)


def test_drafted_rosters_get_legal_lineups_and_standings():
    rng = np.random.default_rng(0)
    listings = [GUARD, WING, FORWARD, BIG, CENTER]
    pool = pd.DataFrame({
        "PLAYER_ID": range(300),
        "POSITIONS": [listings[i] for i in rng.choice(len(listings), 300)],
        "PROJ_MID": rng.gamma(3, 700, 300),
    })
    pool["ACTUAL_PTS"] = pool["PROJ_MID"] * rng.normal(1, 0.3, 300)
    teams = [Team(f"Team {i + 1}", "test") for i in range(config.NUM_TEAMS)]
    picks = run_draft(teams, pool, {"test": best_projection})

    for _, roster in picks.merge(pool, on="PLAYER_ID").groupby("TEAM"):
        slots, _ = best_lineup(roster)
        starters = slots[slots != "BENCH"]
        assert sorted(starters) == sorted(config.STARTING_SLOTS)
        for row, slot in starters.items():
            assert config.SLOT_ELIGIBILITY[slot] & set(roster.loc[row, "POSITIONS"])

    standings = score_draft(picks, pool)
    assert len(standings) == config.NUM_TEAMS
    assert standings["FINISH"].tolist() == sorted(standings["FINISH"])
    assert sorted(standings["DRAFT_POSITION"]) == list(range(1, config.NUM_TEAMS + 1))
    assert standings["SCORE"].is_monotonic_decreasing


def test_walk_forward_split_only_trains_on_earlier_seasons():
    table = pd.DataFrame({"SEASON": config.TRAIN_TARGET_SEASONS + [config.TEST_SEASON]})
    for season in config.BACKTEST_SEASONS:
        train, test = split_by_season(table, season)
        assert set(test["SEASON"]) == {season}
        assert all(config.SEASONS.index(s) < config.SEASONS.index(season) for s in train["SEASON"])