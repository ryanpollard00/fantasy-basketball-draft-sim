"""Tests for src/model.py, using small hand-made tables and a temporary database."""

import numpy as np
import pandas as pd
import pytest

import config
from src.features import MODEL_FEATURES, TARGET
from src.model import (
    load_projections,
    project_with_range,
    range_coverage,
    save_projections,
    scale_range,
    season_splits,
    train_random_forest,
)


def test_scale_range_stretches_both_sides():
    ranges = pd.DataFrame({"PROJ_FLOOR": [900.0], "PROJ_MID": [1000.0], "PROJ_CEILING": [1200.0]})
    scaled = scale_range(ranges, 2)
    assert scaled["PROJ_FLOOR"].iloc[0] == 800       # 100 below the middle becomes 200 below
    assert scaled["PROJ_CEILING"].iloc[0] == 1400    # 200 above the middle becomes 400 above
    assert scaled["PROJ_MID"].iloc[0] == 1000        # the middle never moves


def test_scaled_floor_never_goes_below_zero():
    ranges = pd.DataFrame({"PROJ_FLOOR": [100.0], "PROJ_MID": [300.0], "PROJ_CEILING": [500.0]})
    assert scale_range(ranges, 3)["PROJ_FLOOR"].iloc[0] == 0


def test_range_coverage():
    ranges = pd.DataFrame({"PROJ_FLOOR": [0, 0, 0, 0], "PROJ_CEILING": [10, 10, 10, 10]})
    actual = pd.Series([5, 10, 11, -1])   # inside, on the edge, above, below
    assert range_coverage(actual, ranges) == 0.5


def test_season_splits_only_validate_on_later_seasons():
    train = pd.DataFrame({"SEASON": np.repeat(config.TRAIN_TARGET_SEASONS, 3)})
    splits = season_splits(train)
    assert len(splits) == len(config.TRAIN_TARGET_SEASONS) - 1
    for fit_rows, validation_rows in splits:
        fit_seasons = set(train["SEASON"].iloc[fit_rows])
        validation_seasons = set(train["SEASON"].iloc[validation_rows])
        assert len(validation_seasons) == 1
        assert max(fit_seasons) < min(validation_seasons)   # season labels sort in time order


def test_floor_middle_ceiling_are_in_order():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.normal(size=(200, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    train[TARGET] = 2000 + 500 * train[MODEL_FEATURES[0]] + rng.normal(0, 300, 200)
    ranges = project_with_range(train_random_forest(train), train.head(20))
    assert (ranges["PROJ_FLOOR"] <= ranges["PROJ_MID"]).all()
    assert (ranges["PROJ_MID"] <= ranges["PROJ_CEILING"]).all()


def test_projections_survive_a_save_and_load(tmp_path):
    projections = pd.DataFrame({
        "PLAYER_ID": [1, 2],
        "PLAYER_NAME": ["Guard", "Center"],
        "POSITIONS": [["PG", "SG"], ["C"]],
        "PROJ_MID": [3000.0, 2500.0],
    })
    db_path = tmp_path / "test.db"
    save_projections(projections, db_path)
    pd.testing.assert_frame_equal(load_projections(db_path), projections)