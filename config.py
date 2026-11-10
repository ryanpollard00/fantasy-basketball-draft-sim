"""
Project settings for the fantasy basketball draft simulator.
 
All tunable settings live here, so scoring, league format and model
parameters can be changed in one place without editing the code.
"""
from pathlib import Path
# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
SEASONS = [
    "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]
TRAIN_TARGET_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
TEST_SEASON = "2025-26"          # backtest season: projected, then scored on actual results

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "fantasy.db"   # SQLite database file (not committed to GitHub)
API_PAUSE_SECONDS = 1.5          # pause between nba_api calls (stats.nba.com rate-limits)
API_RETRIES = 3                  # retries if a call times out
MIN_MINUTES = 500                # drop players with fewer season minutes than this

# ---------------------------------------------------------------------
# Scoring (points league)
# Keys match the column names returned by nba_api's LeagueDashPlayerStats.
# FG_MISS and FT_MISS are calculated: FGA - FGM and FTA - FTM.
# W = wins in games the player played.
# ---------------------------------------------------------------------
SCORING = {
    "PTS": 1,
    "REB": 1,
    "AST": 2,
    "STL": 4,
    "BLK": 4,
    "TOV": -2,
    "FG3M": 1,
    "FGM": 2,
    "FG_MISS": -1,
    "FTM": 1,
    "FT_MISS": -1,
    "W": 1,
}

# ---------------------------------------------------------------------
# League and roster
# ---------------------------------------------------------------------
NUM_TEAMS = 12
TEAMS_PER_STRATEGY = 2           # 6 strategies x 2 = 12 teams

STARTING_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
BENCH_SLOTS = 4
ROSTER_SIZE = len(STARTING_SLOTS) + BENCH_SLOTS   # 12 players = 12 draft rounds

ALL_POSITIONS = {"PG", "SG", "SF", "PF", "C"}
SLOT_ELIGIBILITY = {
    "PG": {"PG"},
    "SG": {"SG"},
    "SF": {"SF"},
    "PF": {"PF"},
    "C": {"C"},
    "G": {"PG", "SG"},
    "F": {"SF", "PF"},
    "UTIL": ALL_POSITIONS,
    "BENCH": ALL_POSITIONS,
}

# Draft rule: a pick goes into the first open slot in this order that it fits
# (most restrictive first, bench last).
SLOT_PRIORITY = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "BENCH"]

# The NBA's official listings only use G, F and C. This maps them to fantasy
# positions (an approximation; see Limitations in the README).
NBA_POSITION_MAP = {
    "G": ["PG", "SG"],
    "F": ["SF", "PF"],
    "C": ["C"],
    "G-F": ["SG", "SF"],
    "F-G": ["SG", "SF"],
    "F-C": ["PF", "C"],
    "C-F": ["PF", "C"],
}

# ---------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------
LONG_TERM_WEIGHTS = [0.5, 0.3, 0.2]   # most recent season first (strategy 2)

RF_PARAMS = {
    "n_estimators": 300,          # number of trees; their spread gives floor/ceiling
    "min_samples_leaf": 5,
    "max_features": 0.5,          # chosen by cross-validation
    "random_state": 42,
    "n_jobs": -1,
}
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,        # chosen by cross-validation
    "max_depth": 2,               # chosen by cross-validation
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

FLOOR_PERCENTILE = 25
CEILING_PERCENTILE = 75

# ---------------------------------------------------------------------
# Draft strategy and simulation
# ---------------------------------------------------------------------
# VORP replacement level
# "data":  take the top NUM_TEAMS x len(STARTING_SLOTS) players by projection (96),
#          count how many play each position, and use the next player at that position
# "fixed": use REPLACEMENT_RANK at every position (for the sensitivity check)
REPLACEMENT_METHOD = "data"
REPLACEMENT_RANK = 24            # only used when REPLACEMENT_METHOD = "fixed"

N_SIMULATIONS = 500
RANDOM_SEED = 42

# Hyperparameter search grids (time-based cross-validation on the training seasons)
RF_PARAM_GRID = {
    "min_samples_leaf": [5, 10, 20],
    "max_features": [0.33, 0.5, 1.0],
}
XGB_PARAM_GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05],
    "n_estimators": [150, 300, 600],
}