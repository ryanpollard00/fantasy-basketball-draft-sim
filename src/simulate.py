"""
Monte Carlo simulation: each strategy drafts as one team against 11 human-like opponents.

The human-like opponents rank players by last season's fantasy points with its own
random noise in every draft, so they disagree with each other the way real drafters do.

Run from the project root (takes several minutes):
    python -m src.simulate
"""

import sqlite3
import time

import numpy as np
import pandas as pd

import config
from src.draft import Team, run_draft
from src.lineup import score_draft
from src.model import load_projections
from src.strategies import build_strategies, make_strategy

FIELD_TEAMS = config.NUM_TEAMS - 1


def field_strategies(players, rng):
    """Pick functions for the 11 opponents in one draft.

    Each opponent ranks players by config.FIELD_VALUE times its own random factor per player
    (mean 1, standard deviation config.FIELD_NOISE), so opponents disagree about who is next.
    """
    noise = rng.normal(1, config.FIELD_NOISE, size=(len(players), FIELD_TEAMS))
    boards = pd.DataFrame(
        players[config.FIELD_VALUE].to_numpy()[:, None] * noise,
        index=players.index,
        columns=[f"field_{i + 1}" for i in range(FIELD_TEAMS)],
    )
    return {name: make_strategy(boards, name) for name in boards.columns}


def simulate_season(players, season, num_drafts=None):
    """Draft every strategy against the noisy field num_drafts times in one season.

    Draft d puts the test team in draft spot d % 12, so every seat is used equally, and uses the
    same random field for every strategy. This way, strategies are compared on identical leagues.
    Returns one row per strategy per draft, describing the test team.
    """
    num_drafts = num_drafts or config.N_SIMULATIONS
    test_strategies = build_strategies(players)
    rows = []
    for draft_number in range(num_drafts):
        test_seat = draft_number % config.NUM_TEAMS
        field = field_strategies(players, np.random.default_rng([config.RANDOM_SEED, draft_number]))
        strategies = {**test_strategies, **field}

        for test_strategy in config.STRATEGIES:
            field_names = iter(field)
            teams = [
                Team("Test team", test_strategy) if seat == test_seat else Team(f"Field {seat + 1}", next(field_names))
                for seat in range(config.NUM_TEAMS)
            ]
            standings = score_draft(run_draft(teams, players, strategies), players)
            test_team = standings[standings["TEAM"] == "Test team"].iloc[0]
            field_average = standings.loc[standings["TEAM"] != "Test team", "SCORE"].mean()
            rows.append({
                "SEASON": season,
                "DRAFT": draft_number + 1,
                "STRATEGY": test_strategy,
                "DRAFT_POSITION": test_seat + 1,
                "SCORE": test_team["SCORE"],
                "FINISH": test_team["FINISH"],
                "EDGE_PCT": (test_team["SCORE"] / field_average - 1) * 100,
            })
        if (draft_number + 1) % 40 == 0:
            print(f"  {season}: {draft_number + 1} of {num_drafts} leagues done")
    return pd.DataFrame(rows)


def simulate_all_seasons(num_drafts=None):
    """Run the simulation for every season in config.BACKTEST_SEASONS."""
    projections = load_projections()
    results = []
    for season in config.BACKTEST_SEASONS:
        start = time.time()
        players = projections[projections["SEASON"] == season]
        results.append(simulate_season(players, season, num_drafts))
        print(f"{season} finished in {(time.time() - start) / 60:.1f} minutes")
    return pd.concat(results, ignore_index=True)


def save_results(results, db_path=None):
    """Save simulation results to SQLite, replacing any earlier run."""
    conn = sqlite3.connect(db_path or config.DB_PATH)
    results.to_sql("simulation_results", conn, if_exists="replace", index=False)
    conn.close()


def load_results(db_path=None):
    """Load saved simulation results."""
    conn = sqlite3.connect(db_path or config.DB_PATH)
    results = pd.read_sql("SELECT * FROM simulation_results", conn)
    conn.close()
    return results


def summarize(results, value):
    """Average of `value` for each strategy, by season and across all seasons."""
    summary = results.pivot_table(index="STRATEGY", columns="SEASON", values=value, aggfunc="mean")
    summary["ALL SEASONS"] = results.groupby("STRATEGY")[value].mean()
    return summary


if __name__ == "__main__":
    results = simulate_all_seasons()
    save_results(results)
    print(f"\nSaved {len(results):,} results\n")

    print("Average finish against the field (1 = best of 12):")
    print(summarize(results, "FINISH").sort_values("ALL SEASONS").round(2).to_string())
    print("\nPoints above the field's average team (%):")
    print(summarize(results, "EDGE_PCT").sort_values("ALL SEASONS", ascending=False).round(1).to_string())