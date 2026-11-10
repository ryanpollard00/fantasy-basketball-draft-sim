""" Tests for src/features.py, using small hand-made tables instead of the database."""

import pandas as pd
import pytest

import config
from src.features import (
    BASELINE_COLUMNS,
    MODEL_FEATURES,
    TARGET,
    add_history,
    add_rate_stats,
    add_season_number,
    add_weighted_averages,
    build_model_table,
    split_by_season,
)

# Typical values for every column the feature code reads; each test overrides the specific values that it is checking and leaves the other values as the default ones
DEFAULTS = {
    "PLAYER_NAME": "Player", "POSITIONS": ["PG", "SG"], "FANTASY_PTS": 1000, "FANTASY_PPG": 20.0,
    "GP_SHARE": 0.6, "MPG": 25.0, "MIN": 1500, "WIN_PCT": 0.5, "AGE": 25,
}


def make_base(rows):
    """Build a base table from a list of partial rows, filling the other columns with DEFAULTS."""
    return pd.DataFrame([{**DEFAULTS, **row} for row in rows])


def history_for(rows, season):
    """Run the history and weighted-average steps and return the one row for `season`."""
    df = add_weighted_averages(add_history(add_season_number(make_base(rows))))
    return df[df["SEASON"] == season].iloc[0]


def test_games_played_share_is_capped_at_one():
    # A traded player can appear in 84 games while his final team played 82
    seasons = pd.DataFrame({"SEASON": ["2024-25"], "TEAM_ID": [1], "GP": [84], "MIN": [2520]})
    teams = pd.DataFrame({"SEASON": ["2024-25"], "TEAM_ID": [1], "TEAM_GAMES": [82], "WIN_PCT": [0.5]})
    assert add_rate_stats(seasons, teams)["GP_SHARE"].iloc[0] == 1


def test_last_season_lines_up_with_next_season():
    row = history_for([
        {"PLAYER_ID": 1, "SEASON": "2023-24", "FANTASY_PPG": 30.0},
        {"PLAYER_ID": 1, "SEASON": "2024-25", "FANTASY_PPG": 35.0},
    ], "2024-25")
    assert row["FANTASY_PPG_LAG1"] == 30.0


def test_missed_season_leaves_a_gap():
    # Played 2022-23, sat out 2023-24, returned in 2024-25
    row = history_for([
        {"PLAYER_ID": 1, "SEASON": "2022-23", "FANTASY_PPG": 30.0},
        {"PLAYER_ID": 1, "SEASON": "2024-25", "FANTASY_PPG": 35.0},
    ], "2024-25")
    assert pd.isna(row["FANTASY_PPG_LAG1"])
    assert row["FANTASY_PPG_LAG2"] == 30.0


def test_weighted_average_with_full_history():
    row = history_for([
        {"PLAYER_ID": 1, "SEASON": "2021-22", "FANTASY_PTS": 1000},
        {"PLAYER_ID": 1, "SEASON": "2022-23", "FANTASY_PTS": 2000},
        {"PLAYER_ID": 1, "SEASON": "2023-24", "FANTASY_PTS": 3000},
        {"PLAYER_ID": 1, "SEASON": "2024-25", "FANTASY_PTS": 0},
    ], "2024-25")
    # 0.5 x 3000 + 0.3 x 2000 + 0.2 x 1000 = 2300
    assert row["FANTASY_PTS_AVG3"] == pytest.approx(2300)
    assert row["SEASONS_OF_HISTORY"] == 3


def test_weighted_average_rescales_for_short_history():
    row = history_for([
        {"PLAYER_ID": 1, "SEASON": "2022-23", "FANTASY_PTS": 2000},
        {"PLAYER_ID": 1, "SEASON": "2023-24", "FANTASY_PTS": 3000},
        {"PLAYER_ID": 1, "SEASON": "2024-25", "FANTASY_PTS": 0},
    ], "2024-25")
    # (0.5 x 3000 + 0.3 x 2000) / 0.8 = 2625
    assert row["FANTASY_PTS_AVG3"] == pytest.approx(2625)


def test_rookies_and_low_minute_players_are_excluded():
    base = make_base([
        {"PLAYER_ID": 1, "SEASON": "2024-25", "MIN": 1500},   # qualified last season: kept
        {"PLAYER_ID": 1, "SEASON": "2025-26"},
        {"PLAYER_ID": 2, "SEASON": "2024-25", "MIN": 300},    # under 500 minutes: dropped
        {"PLAYER_ID": 2, "SEASON": "2025-26"},
        {"PLAYER_ID": 3, "SEASON": "2025-26"},                # rookie, no last season: dropped
    ])
    assert build_model_table(base)["PLAYER_ID"].tolist() == [1]


def test_target_season_stats_never_reach_the_inputs():
    """Leakage check: multiply every 2025-26 stat and make sure that the 2025-26 inputs don't move"""
    base = make_base([
        {"PLAYER_ID": player, "SEASON": season, "FANTASY_PTS": 1000 + 100 * i + player, "AGE": 20 + i}
        for player in range(1, 4)
        for i, season in enumerate(config.SEASONS)
    ])
    changed = base.copy()
    in_test_season = changed["SEASON"] == config.TEST_SEASON
    stat_columns = ["FANTASY_PTS", "FANTASY_PPG", "GP_SHARE", "MPG", "MIN", "WIN_PCT", "AGE"]
    changed.loc[in_test_season, stat_columns] *= 100

    _, original_test = split_by_season(build_model_table(base))
    _, changed_test = split_by_season(build_model_table(changed))

    inputs = MODEL_FEATURES + BASELINE_COLUMNS
    pd.testing.assert_frame_equal(original_test[inputs], changed_test[inputs])
    assert (original_test[TARGET] != changed_test[TARGET]).all()


def test_split_keeps_seasons_apart():
    table = pd.DataFrame({"SEASON": config.TRAIN_TARGET_SEASONS + [config.TEST_SEASON]})
    train, test = split_by_season(table)
    assert set(train["SEASON"]) == set(config.TRAIN_TARGET_SEASONS)
    assert set(test["SEASON"]) == {config.TEST_SEASON}