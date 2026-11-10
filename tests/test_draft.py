"""Tests for src/draft.py, using small hand-made player pools."""

import numpy as np
import pandas as pd
import pytest

import config
from src.draft import Team, best_projection, run_draft, snake_order

GUARD = ["PG", "SG"]
CENTER = ["C"]


def make_pool(num_players=300, seed=0):
    """Fake draft pool with a realistic mix of positions."""
    rng = np.random.default_rng(seed)
    listings = [["PG", "SG"], ["SG", "SF"], ["SF", "PF"], ["PF", "C"], ["C"]]
    choices = rng.choice(len(listings), num_players, p=[0.35, 0.15, 0.25, 0.13, 0.12])
    return pd.DataFrame({
        "PLAYER_ID": range(num_players),
        "POSITIONS": [listings[i] for i in choices],
        "PROJ_MID": rng.gamma(3, 700, num_players),
    })


def fill(team, slots):
    """Fill the given slots with placeholder players."""
    for player_id, slot in enumerate(slots):
        team.add_player(player_id, slot)


def test_new_team_has_every_slot_open():
    team = Team("Team 1", "test")
    assert len(team.open_slots) == config.ROSTER_SIZE
    assert not team.is_full()


def test_guards_fill_the_most_restrictive_slot_first():
    team = Team("Team 1", "test")
    slots = []
    for player_id in range(4):
        slot = team.slot_for(GUARD)
        slots.append(slot)
        team.add_player(player_id, slot)
    assert slots == ["PG", "SG", "G", "UTIL"]


def test_bench_stays_closed_until_starters_are_filled():
    team = Team("Team 1", "test")
    fill(team, ["PG", "SG", "G", "UTIL"])
    assert team.slot_for(GUARD) is None     # every guard-eligible starting slot is full
    assert team.can_draft(CENTER)           # but a center still fits


def test_bench_opens_once_starters_are_filled():
    team = Team("Team 1", "test")
    fill(team, config.STARTING_SLOTS)
    assert team.slot_for(GUARD) == "BENCH"


def test_snake_order_reverses_each_round():
    assert snake_order(3, 2) == [0, 1, 2, 2, 1, 0]
    order = snake_order(config.NUM_TEAMS, config.ROSTER_SIZE)
    for position in range(config.NUM_TEAMS):
        assert order.count(position) == config.ROSTER_SIZE


def test_full_draft_produces_legal_rosters():
    players = make_pool()
    teams = [Team(f"Team {i + 1}", "test") for i in range(config.NUM_TEAMS)]
    picks = run_draft(teams, players, {"test": best_projection})

    assert len(picks) == config.NUM_TEAMS * config.ROSTER_SIZE
    assert picks["PLAYER_ID"].is_unique
    assert all(team.is_full() for team in teams)

    positions = picks.merge(players, on="PLAYER_ID")
    for _, pick in positions.iterrows():
        assert config.SLOT_ELIGIBILITY[pick["SLOT"]] & set(pick["POSITIONS"])

    # Within each team, no bench pick comes before the last starter
    for _, team_picks in picks.groupby("TEAM"):
        starter_picks = team_picks.loc[team_picks["SLOT"] != "BENCH", "PICK"]
        bench_picks = team_picks.loc[team_picks["SLOT"] == "BENCH", "PICK"]
        assert starter_picks.max() < bench_picks.min()


def test_illegal_pick_stops_the_draft():
    players = make_pool()
    teams = [Team(f"Team {i + 1}", "test") for i in range(config.NUM_TEAMS)]

    def always_player_zero(team, allowed):
        return 0   # legal on the first pick, already taken on the second

    with pytest.raises(ValueError):
        run_draft(teams, players, {"test": always_player_zero})