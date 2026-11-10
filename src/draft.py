"""
Snake draft engine: teams, roster slots and the draft loop.

Run from the project root to print a preview:
    python -m src.draft
"""

import pandas as pd

import config


class Team:
    """Holds one fantasy team's draft strategy, players, and roster slots still open."""

    def __init__(self, name, strategy_name):
        self.name = name
        self.strategy_name = strategy_name
        self.open_slots = list(config.STARTING_SLOTS) + ["BENCH"] * config.BENCH_SLOTS
        self.roster = []   # (slot, player_id) pairs, in draft order

    def add_player(self, player_id, slot):
        """Put a drafted player into one of the team's open slots."""
        self.open_slots.remove(slot)
        self.roster.append((slot, player_id))

    def slot_for(self, positions):
        """The most restrictive open slot a player fits, or none if he fits none.
        """
        starters_filled = all(slot == "BENCH" for slot in self.open_slots)
        for slot in config.SLOT_PRIORITY:
            if slot not in self.open_slots:
                continue
            if slot == "BENCH" and not starters_filled:
                continue
            if config.SLOT_ELIGIBILITY[slot] & set(positions):
                return slot
        return None

    def can_draft(self, positions):
        """True if the player fits one of the team's open slots under the roster rules."""
        return self.slot_for(positions) is not None

    def is_full(self):
        """True once every roster slot is filled."""
        return len(self.open_slots) == 0


def snake_order(num_teams, num_rounds):
    """Draft positions (0 to num_teams - 1) in pick order, reversing direction every round."""
    order = []
    for round_number in range(num_rounds):
        positions = list(range(num_teams))
        if round_number % 2 == 1:
            positions.reverse()
        order.extend(positions)
    return order


def run_draft(teams, players, strategies):
    """Run a snake draft and return every pick in order.

    teams: Team objects in draft-position order (teams[0] picks first in round 1).
    players: one row per player in the pool, with PLAYER_ID and POSITIONS columns.
    strategies: maps each team's strategy_name to a function(team, allowed) that returns
        a PLAYER_ID, where `allowed` holds only the players that team may draft.
    """
    available = players.copy()
    picks = []
    for pick_number, position in enumerate(snake_order(len(teams), config.ROSTER_SIZE), start=1):
        team = teams[position]
        allowed = available[available["POSITIONS"].apply(team.can_draft)]
        player_id = strategies[team.strategy_name](team, allowed)
        if player_id not in set(allowed["PLAYER_ID"]):
            raise ValueError(f"{team.name} chose player {player_id}, who is not available to it")

        positions = allowed.loc[allowed["PLAYER_ID"] == player_id, "POSITIONS"].iloc[0]
        slot = team.slot_for(positions)
        team.add_player(player_id, slot)
        available = available[available["PLAYER_ID"] != player_id]
        picks.append({"PICK": pick_number, "TEAM": team.name, "STRATEGY": team.strategy_name,
                      "PLAYER_ID": player_id, "SLOT": slot})
    return pd.DataFrame(picks)


def best_projection(team, allowed):
    """Placeholder strategy for testing the engine: take the highest middle projection from the random forest model."""
    return allowed.loc[allowed["PROJ_MID"].idxmax(), "PLAYER_ID"]


if __name__ == "__main__":
    from src.model import load_projections

    players = load_projections()
    teams = [Team(f"Team {i + 1}", "test") for i in range(config.NUM_TEAMS)]
    picks = run_draft(teams, players, {"test": best_projection})
    picks = picks.merge(players[["PLAYER_ID", "PLAYER_NAME", "POSITIONS"]], on="PLAYER_ID")

    print(picks.head(15).to_string(index=False))
    print("\nTeam 1's roster:")
    print(picks[picks["TEAM"] == "Team 1"][["PICK", "PLAYER_NAME", "POSITIONS", "SLOT"]].to_string(index=False))
    print("\nEvery team full:", all(team.is_full() for team in teams))
    print("Different players drafted:", picks["PLAYER_ID"].nunique(), "of", len(picks), "picks")