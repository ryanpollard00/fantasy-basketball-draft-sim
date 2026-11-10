"""
Train, evaluate and apply the fantasy point projection models.
"""

import sqlite3
import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

import config
from src.features import MODEL_FEATURES, TARGET, build_model_table, split_by_season


def evaluate(actual, predicted):
    """Mean absolute error and root mean squared error, both in fantasy points."""
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    }


def train_random_forest(train):
    """Fit a Random Forest on the training rows using config.RF_PARAMS."""
    model = RandomForestRegressor(**config.RF_PARAMS)
    model.fit(train[MODEL_FEATURES], train[TARGET])
    return model


def train_xgboost(train):
    """Fit an XGBoost model on the training rows using config.XGB_PARAMS."""
    model = XGBRegressor(**config.XGB_PARAMS)
    model.fit(train[MODEL_FEATURES], train[TARGET])
    return model


def season_splits(train):
    """Expanding-window splits: fit on earlier training seasons, validate on the next one."""
    seasons = config.TRAIN_TARGET_SEASONS
    splits = []
    for i in range(1, len(seasons)):
        fit_rows = np.flatnonzero(train["SEASON"].isin(seasons[:i]))
        validation_rows = np.flatnonzero(train["SEASON"] == seasons[i])
        splits.append((fit_rows, validation_rows))
    return splits


def tune(model, param_grid, train):
    """Try every combination in param_grid with season-based cross-validation.

    Returns the fitted search; its best model is refit on all training rows.
    """
    search = GridSearchCV(
        model,
        param_grid,
        cv=season_splits(train),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(train[MODEL_FEATURES], train[TARGET])
    return search


def tune_random_forest(train):
    return tune(RandomForestRegressor(**config.RF_PARAMS), config.RF_PARAM_GRID, train)


def tune_xgboost(train):
    return tune(XGBRegressor(**config.XGB_PARAMS), config.XGB_PARAM_GRID, train)


def project_with_range(rf, rows):
    """Floor, middle and ceiling projections from a fitted Random Forest.

    The middle is the forest's average prediction; the floor and ceiling are
    percentiles of the individual trees' predictions (config.FLOOR_PERCENTILE
    and config.CEILING_PERCENTILE).
    """
    features = rows[MODEL_FEATURES].to_numpy()
    tree_predictions = np.stack([tree.predict(features) for tree in rf.estimators_])
    return pd.DataFrame(
        {
            "PROJ_FLOOR": np.percentile(tree_predictions, config.FLOOR_PERCENTILE, axis=0),
            "PROJ_MID": rf.predict(rows[MODEL_FEATURES]),
            "PROJ_CEILING": np.percentile(tree_predictions, config.CEILING_PERCENTILE, axis=0),
        },
        index=rows.index,
    )


def scale_range(ranges, factor):
    """Stretch each floor and ceiling away from the middle projection by `factor` (floors stop at 0)."""
    scaled = ranges.copy()
    scaled["PROJ_FLOOR"] = (ranges["PROJ_MID"] - factor * (ranges["PROJ_MID"] - ranges["PROJ_FLOOR"])).clip(lower=0)
    scaled["PROJ_CEILING"] = ranges["PROJ_MID"] + factor * (ranges["PROJ_CEILING"] - ranges["PROJ_MID"])
    return scaled


def range_coverage(actual, ranges):
    """Share of actual results that fall between the floor and the ceiling."""
    return actual.between(ranges["PROJ_FLOOR"], ranges["PROJ_CEILING"]).mean()


def calibrate_range_factor(train):
    """Find the stretch factor that makes floor-to-ceiling ranges hold the intended share of results.

    Uses the same season splits as tuning: fit on earlier training seasons, project the
    next one, and pick the factor whose coverage is closest to the percentile gap
    (IQR = 50%). The test season is never used.
    """
    target_coverage = (config.CEILING_PERCENTILE - config.FLOOR_PERCENTILE) / 100
    actuals, ranges = [], []
    for fit_rows, validation_rows in season_splits(train):
        rf = train_random_forest(train.iloc[fit_rows])
        validation = train.iloc[validation_rows]
        actuals.append(validation[TARGET])
        ranges.append(project_with_range(rf, validation))
    actual = pd.concat(actuals)
    ranges = pd.concat(ranges)

    factors = np.arange(1.0, 5.01, 0.05)
    coverages = np.array([range_coverage(actual, scale_range(ranges, f)) for f in factors])
    return float(factors[np.argmin(np.abs(coverages - target_coverage))])


def build_projections():
    """Train on the training seasons and project every player in the test season.

    One row per player in the draft pool: positions, the two naive projections
    (strategies 1 and 2), the calibrated floor, middle and ceiling (strategies 3-6),
    and actual points, which are used only to score teams after the draft.
    """
    train, test = split_by_season(build_model_table())
    rf = train_random_forest(train)
    ranges = scale_range(project_with_range(rf, test), calibrate_range_factor(train))
    projections = test[["PLAYER_ID", "PLAYER_NAME", "POSITIONS", "FANTASY_PTS_LAG1", "FANTASY_PTS_AVG3"]].join(ranges)
    projections["ACTUAL_PTS"] = test[TARGET]
    return projections.rename(columns={"FANTASY_PTS_LAG1": "PROJ_LAST_SEASON", "FANTASY_PTS_AVG3": "PROJ_AVG3"})


def save_projections(projections, db_path=None):
    """Save projections to SQLite, storing each player's positions as text such as "PG,SG,etc"."""
    to_save = projections.copy()
    to_save["POSITIONS"] = to_save["POSITIONS"].str.join(",")
    conn = sqlite3.connect(db_path or config.DB_PATH)
    to_save.to_sql("projections", conn, if_exists="replace", index=False)
    conn.close()


def load_projections(db_path=None):
    """Load saved projections, turning positions back into lists."""
    conn = sqlite3.connect(db_path or config.DB_PATH)
    projections = pd.read_sql("SELECT * FROM projections", conn)
    conn.close()
    projections["POSITIONS"] = projections["POSITIONS"].str.split(",")
    return projections


if __name__ == "__main__":
    projections = build_projections()
    save_projections(projections)
    print(f"Saved {len(projections)} projections for {config.TEST_SEASON}\n")
    columns = ["PLAYER_NAME", "POSITIONS", "PROJ_LAST_SEASON", "PROJ_AVG3", "PROJ_FLOOR", "PROJ_MID", "PROJ_CEILING"]
    print(projections.nlargest(10, "PROJ_MID")[columns].round(0).to_string(index=False))