"""
Calculate fantasy points and fantasy position eligibility for each player season.

Run from the project root to print a preview:
    python -m src.scoring
"""

import sqlite3

import pandas as pd

import config


def load_player_seasons():
    """Load every player's season totals from the database."""
    conn = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql("SELECT * FROM player_seasons", conn)
    conn.close()
    return df

def add_missed_shots(df):
    """Add missed field goals and free throws, which are scored but not provided by the API."""
    df = df.copy()
    df["FG_MISS"] = df["FGA"] - df["FGM"]
    df["FT_MISS"] = df["FTA"] - df["FTM"]
    return df

def add_fantasy_points(df):
    """Add season fantasy points using the scoring rules in config.SCORING."""
    df = df.copy()
    df["FANTASY_PTS"] = 0
    for stat, points in config.SCORING.items():
        df["FANTASY_PTS"] += df[stat] * points
    return df

def add_per_game(df):
    """Add fantasy points per game, which compares players fairly across missed games and shortened seasons."""
    df = df.copy()
    df["FANTASY_PPG"] = df["FANTASY_PTS"] / df["GP"]
    return df

def load_players():
    """Load each player's NBA position listing from the database."""
    conn = sqlite3.connect(config.DB_PATH)
    players = pd.read_sql("SELECT PLAYER_ID, POSITION FROM players", conn)
    conn.close()
    return players

def add_positions(df, players):
    """Add each player's fantasy positions, translated from the NBA's G/F/C listing."""
    df = df.merge(players, on="PLAYER_ID", how="left", validate="many_to_one")
    df["POSITIONS"] = df["POSITION"].map(config.NBA_POSITION_MAP)
    df["POSITIONS"] = df["POSITIONS"].apply(lambda p: p if isinstance(p, list) else [])
    return df


if __name__ == "__main__":
    df = load_player_seasons()
    df = add_missed_shots(df)
    df = add_fantasy_points(df)
    df = add_per_game(df)
    df = add_positions(df, load_players())
    print(df["POSITION"].value_counts(dropna=False))
    missing = df[df["POSITIONS"].str.len() == 0]
    print(f"\nPlayer seasons with no fantasy position: {len(missing)}")
    print(missing[["PLAYER_NAME", "SEASON", "MIN", "POSITION"]].head(10).to_string(index=False))