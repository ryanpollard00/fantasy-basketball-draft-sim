"""
Download NBA data with nba_api and save it to SQLite.

Run from the project's main folder (with the venv active):
    python -m src.data_pull

Creates three tables in the database at config.DB_PATH:
    player_seasons  one row per player per season (season totals)
    team_seasons    one row per team per season (wins, losses, win %)
    players         one row per player (position, draft info)

Seasons already in the database are skipped, so rerunning is fast.
To re-download everything, delete the database file and run again.
"""

import sqlite3
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguestandingsv3, playerindex

import config

# Columns kept from each table (the stats used for fantasy scoring and model features)
PLAYER_SEASON_COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE",
    "GP", "W", "L", "MIN",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PTS",
]
TEAM_SEASON_COLUMNS = {
    "TeamID": "TEAM_ID", "TeamCity": "TEAM_CITY", "TeamName": "TEAM_NAME",
    "WINS": "WINS", "LOSSES": "LOSSES", "WinPCT": "WIN_PCT",
}
PLAYER_COLUMNS = {
    "PERSON_ID": "PLAYER_ID", "PLAYER_FIRST_NAME": "FIRST_NAME", "PLAYER_LAST_NAME": "LAST_NAME",
    "POSITION": "POSITION", "DRAFT_YEAR": "DRAFT_YEAR", "DRAFT_NUMBER": "DRAFT_NUMBER",
    "FROM_YEAR": "FROM_YEAR", "TO_YEAR": "TO_YEAR",
}


def call_with_retries(fetch, label):
    """Run one API call. Retry with a growing wait if it fails, then pause before the next call."""
    for attempt in range(1, config.API_RETRIES + 1):
        try:
            df = fetch()
            time.sleep(config.API_PAUSE_SECONDS)
            return df
        except Exception as error:
            wait = config.API_PAUSE_SECONDS * attempt * 4
            print(f"  {label}: attempt {attempt} failed ({type(error).__name__}). Retrying in {wait:.0f}s...")
            time.sleep(wait)
    raise RuntimeError(f"{label}: failed after {config.API_RETRIES} attempts. Wait a few minutes and rerun.")


def fetch_player_seasons(season):
    """Season totals for every player who played in the regular season."""
    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        timeout=60,
    )
    df = response.get_data_frames()[0][PLAYER_SEASON_COLUMNS].copy()
    df.insert(0, "SEASON", season)
    return df


def fetch_team_seasons(season):
    """Wins, losses and win % for every team."""
    response = leaguestandingsv3.LeagueStandingsV3(
        season=season,
        season_type="Regular Season",
        timeout=60,
    )
    df = response.get_data_frames()[0][list(TEAM_SEASON_COLUMNS)].rename(columns=TEAM_SEASON_COLUMNS)
    df.insert(0, "SEASON", season)
    return df


def fetch_players():
    """Position and draft info for every player, including retired ones."""
    response = playerindex.PlayerIndex(
        season=config.TEST_SEASON,
        historical_nullable="1",
        timeout=60,
    )
    return response.get_data_frames()[0][list(PLAYER_COLUMNS)].rename(columns=PLAYER_COLUMNS)


def saved_seasons(conn, table):
    """Seasons already stored in a table (empty if the table doesn't exist yet)."""
    try:
        return set(pd.read_sql(f"SELECT DISTINCT SEASON FROM {table}", conn)["SEASON"])
    except pd.errors.DatabaseError:
        return set()


def table_exists(conn, table):
    query = "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?"
    return conn.execute(query, (table,)).fetchone() is not None


def main():
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)

    for table, fetch in [("player_seasons", fetch_player_seasons), ("team_seasons", fetch_team_seasons)]:
        done = saved_seasons(conn, table)
        for season in config.SEASONS:
            if season in done:
                print(f"{table} {season}: already saved, skipping")
                continue
            df = call_with_retries(lambda: fetch(season), f"{table} {season}")
            df.to_sql(table, conn, if_exists="append", index=False)
            print(f"{table} {season}: saved {len(df)} rows")

    if table_exists(conn, "players"):
        print("players: already saved, skipping")
    else:
        df = call_with_retries(fetch_players, "players")
        df.to_sql("players", conn, if_exists="replace", index=False)
        print(f"players: saved {len(df)} rows")

    # Print row counts per season as a sanity check on the download
    summary = pd.read_sql(
        "SELECT SEASON, COUNT(*) AS players, SUM(GP) AS games_played FROM player_seasons GROUP BY SEASON",
        conn,
    )
    print("\nPlayers per season:")
    print(summary.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
