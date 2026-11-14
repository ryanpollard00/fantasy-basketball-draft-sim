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


def train_random_forest(train, params=None):
    """Fit a Random Forest on the training rows (config.RF_PARAMS unless params are given)."""
    model = RandomForestRegressor(**(params or config.RF_PARAMS))
    model.fit(train[MODEL_FEATURES], train[TARGET])
    return model


def train_xgboost(train):
    """Fit an XGBoost model on the training rows using config.XGB_PARAMS."""
    model = XGBRegressor(**config.XGB_PARAMS)
    model.fit(train[MODEL_FEATURES], train[TARGET])
    return model


def season_splits(train):
    """Expanding-window splits: fit on earlier training seasons, validate on the next one."""
    seasons = sorted(train["SEASON"].unique())
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


def calibrate_range_factor(train, params=None):
    """Find the stretch factor that makes floor-to-ceiling ranges hold the intended share of results.

    Uses the same season splits as tuning: fit on earlier training seasons, project the
    next one, and pick the factor whose coverage is closest to the percentile gap
    (IQR = 50%). The test season is never used.
    """
    target_coverage = (config.CEILING_PERCENTILE - config.FLOOR_PERCENTILE) / 100
    actuals, ranges = [], []
    for fit_rows, validation_rows in season_splits(train):
        rf = train_random_forest(train.iloc[fit_rows], params)
        validation = train.iloc[validation_rows]
        actuals.append(validation[TARGET])
        ranges.append(project_with_range(rf, validation))
    actual = pd.concat(actuals)
    ranges = pd.concat(ranges)

    factors = np.arange(1.0, 5.01, 0.05)
    coverages = np.array([range_coverage(actual, scale_range(ranges, f)) for f in factors])
    return float(factors[np.argmin(np.abs(coverages - target_coverage))])


def build_projections(season=None):
    """Project every player in one backtest season, using only earlier seasons.

    Tuning and floor/ceiling calibration are redone on that season's training rows, so
    nothing from the backtest season or later can warp its projections. Returns the
    projections and a description of how they were made.
    """
    season = season or config.TEST_SEASON
    train, test = split_by_season(build_model_table(), season)
    search = tune_random_forest(train)
    params = {**config.RF_PARAMS, **search.best_params_}
    rf = search.best_estimator_
    range_factor = calibrate_range_factor(train, params)

    ranges = scale_range(project_with_range(rf, test), range_factor)
    projections = test[["PLAYER_ID", "PLAYER_NAME", "SEASON", "POSITIONS", "FANTASY_PTS_LAG1", "FANTASY_PTS_AVG3"]].join(ranges)
    projections["ACTUAL_PTS"] = test[TARGET]
    projections = projections.rename(columns={"FANTASY_PTS_LAG1": "PROJ_LAST_SEASON", "FANTASY_PTS_AVG3": "PROJ_AVG3"})

    details = {
        "season": season,
        "training_seasons": sorted(train["SEASON"].unique()),
        "best_params": search.best_params_,
        "range_factor": range_factor,
    }
    return projections, details


def build_all_projections():
    """Projections for every season in config.BACKTEST_SEASONS, combined into one table."""
    results = [build_projections(season) for season in config.BACKTEST_SEASONS]
    projections = pd.concat([projections for projections, _ in results], ignore_index=True)
    details = [details for _, details in results]
    return projections, details


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
    projections, all_details = build_all_projections()
    save_projections(projections)
    print(f"Saved {len(projections)} projections for {len(all_details)} backtest seasons\n")

    for details in all_details:
        season_rows = projections[projections["SEASON"] == details["season"]]
        model_mae = evaluate(season_rows["ACTUAL_PTS"], season_rows["PROJ_MID"])["MAE"]
        baseline_mae = evaluate(season_rows["ACTUAL_PTS"], season_rows["PROJ_LAST_SEASON"])["MAE"]
        coverage = range_coverage(season_rows["ACTUAL_PTS"], season_rows)
        print(f"{details['season']}: trained on {', '.join(details['training_seasons'])}")
        print(f"    {len(season_rows)} players | best settings {details['best_params']} | stretch factor {details['range_factor']:.2f}")
        print(f"    MAE: model {model_mae:,.0f} vs. last season {baseline_mae:,.0f} | floor-to-ceiling coverage {coverage:.0%}")