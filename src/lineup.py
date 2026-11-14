"""
Lineup optimization: the best starting lineup a drafted roster can field.

Run from the project root to print a preview:
    python -m src.lineup
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import config

INELIGIBLE = -1e9   # value for a player in a slot he can't fill, so the solver never chooses it


def best_lineup(roster, points_column="ACTUAL_PTS"):
    """Assign a roster's players to slots so the 8 starters score as many points as possible.

    roster: one row per player, with POSITIONS and points_column.
    Returns each player's assigned slot (a Series matching the roster's index) and the
    starters' total points. Bench slots are worth 0, so only starters count.
    """
    slots = list(config.STARTING_SLOTS) + ["BENCH"] * config.BENCH_SLOTS
    value = np.full((len(roster), len(slots)), INELIGIBLE)
    for i, (positions, points) in enumerate(zip(roster["POSITIONS"], roster[points_column])):
        for j, slot in enumerate(slots):
            if slot == "BENCH":
                value[i, j] = 0
            elif config.SLOT_ELIGIBILITY[slot] & set(positions):
                value[i, j] = points

    rows, columns = linear_sum_assignment(value, maximize=True)
    if (value[rows, columns] == INELIGIBLE).any():
        raise ValueError("This roster can't fill every starting slot with an eligible player")

    assigned = pd.Series([slots[c] for c in columns], index=roster.index[rows])
    return assigned, value[rows, columns].sum()


def score_draft(picks, players, points_column="ACTUAL_PTS"):
    """Score of every team's optimized lineup and finishing place in the league (1 = most points) in one draft.

    picks: the table returned by run_draft. players: the draft pool, with POSITIONS
    and points_column.
    """
    drafted = picks.merge(players[["PLAYER_ID", "POSITIONS", points_column]], on="PLAYER_ID")
    rows = []
    for team_name, roster in drafted.groupby("TEAM", sort=False):
        _, score = best_lineup(roster, points_column)
        rows.append({
            "TEAM": team_name,
            "STRATEGY": roster["STRATEGY"].iloc[0],
            "DRAFT_POSITION": roster["PICK"].min(),
            "SCORE": score,
        })
    results = pd.DataFrame(rows)
    results["FINISH"] = results["SCORE"].rank(ascending=False, method="min").astype(int)
    return results.sort_values("FINISH").reset_index(drop=True)


if __name__ == "__main__":
    from src.draft import Team, best_projection, run_draft
    from src.model import load_projections

    players = load_projections()
    players = players[players["SEASON"] == config.TEST_SEASON]
    teams = [Team(f"Team {i + 1}", "test") for i in range(config.NUM_TEAMS)]
    raw_picks = run_draft(teams, players, {"test": best_projection})
    picks = raw_picks.merge(players[["PLAYER_ID", "PLAYER_NAME", "POSITIONS", "ACTUAL_PTS"]], on="PLAYER_ID")

    roster = picks[picks["TEAM"] == "Team 1"].copy()
    roster["BEST_SLOT"], total = best_lineup(roster)
    draft_slot_total = roster.loc[roster["SLOT"] != "BENCH", "ACTUAL_PTS"].sum()

    print(roster[["PLAYER_NAME", "POSITIONS", "ACTUAL_PTS", "SLOT", "BEST_SLOT"]].to_string(index=False))
    print(f"\nStarters' points, draft slots:     {draft_slot_total:,.0f}")
    print(f"Starters' points, best lineup:     {total:,.0f}")
    print("\nStandings for this draft:")
    print(score_draft(raw_picks, players).round(0).to_string(index=False))