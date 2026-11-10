"""Tests for src/scoring.py, using small hand-made DataFrames instead of the database."""

import pandas as pd
import pytest

from src.scoring import add_fantasy_points, add_missed_shots, add_per_game, add_positions


def make_player(**stats):
    """Build a one-row DataFrame for one player season. Any stat not given is 0."""
    columns = ["PLAYER_ID", "GP", "W", "PTS", "REB", "AST", "STL", "BLK", "TOV",
               "FG3M", "FGM", "FGA", "FTM", "FTA"]
    row = {column: 0 for column in columns}
    row.update(stats)
    return pd.DataFrame([row])


def score(df):
    """Run the scoring steps in the same order as the real pipeline."""
    return add_fantasy_points(add_missed_shots(df))


def test_missed_shots():
    df = add_missed_shots(make_player(FGM=4, FGA=10, FTM=3, FTA=5))
    assert df["FG_MISS"].iloc[0] == 6
    assert df["FT_MISS"].iloc[0] == 2


def test_fantasy_points_match_hand_calculation():
    # 20 pts, 10 reb, 5 ast, 2 stl, 1 blk, 3 tov, 2 threes, 8-15 FG, 2-4 FT, 1 win
    df = score(make_player(PTS=20, REB=10, AST=5, STL=2, BLK=1, TOV=3,
                           FG3M=2, FGM=8, FGA=15, FTM=2, FTA=4, W=1))
    # 20 + 10 + 10 + 8 + 4 - 6 + 2 + 16 - 7 + 2 - 2 + 1 = 58
    assert df["FANTASY_PTS"].iloc[0] == 58


def test_each_win_adds_one_point():
    no_win = score(make_player(PTS=20, W=0))["FANTASY_PTS"].iloc[0]
    one_win = score(make_player(PTS=20, W=1))["FANTASY_PTS"].iloc[0]
    assert one_win - no_win == 1


def test_points_per_game():
    df = make_player(GP=4)
    df["FANTASY_PTS"] = 100
    assert add_per_game(df)["FANTASY_PPG"].iloc[0] == 25


def test_positions_are_translated():
    seasons = make_player(PLAYER_ID=1)
    players = pd.DataFrame({"PLAYER_ID": [1], "POSITION": ["G-F"]})
    assert add_positions(seasons, players)["POSITIONS"].iloc[0] == ["SG", "SF"]


def test_unknown_position_becomes_empty_list():
    seasons = make_player(PLAYER_ID=1)
    players = pd.DataFrame({"PLAYER_ID": [1], "POSITION": ["X"]})
    assert add_positions(seasons, players)["POSITIONS"].iloc[0] == []


def test_duplicate_player_raises_error():
    seasons = make_player(PLAYER_ID=1)
    players = pd.DataFrame({"PLAYER_ID": [1, 1], "POSITION": ["G", "F"]})
    with pytest.raises(pd.errors.MergeError):
        add_positions(seasons, players)
