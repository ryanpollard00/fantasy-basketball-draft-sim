"""
The six draft strategies, including which number each one uses to rank players, and how it ranks them.

Run from the project root to print a preview:
    python -m src.strategies
"""

import pandas as pd

import config

POSITIONS = ["PG", "SG", "SF", "PF", "C"]


def eligible(players, position):
    """True for each player who can play `position`."""
    return players["POSITIONS"].apply(lambda positions: position in positions)


def position_demand():
    """Starting slots each position fills across the league.

    Each position gets its own slot, plus an even share of every flexible slot (G, F, UTIL)
    that accepts it. The totals add up to every starting slot in the league (96). Further explanation found in config.py.
    """
    demand = pd.Series(0.0, index=POSITIONS)
    for slot in config.STARTING_SLOTS:
        accepted = config.SLOT_ELIGIBILITY[slot]
        for position in accepted:
            demand[position] += config.NUM_TEAMS / len(accepted)
    return demand


def replacement_levels(players, value_column):
    """Each position's replacement level: the value of the best player there who misses the cut.

    "demand" method: the cut is the position's demand (about 20 per guard and forward
    position, 14 at center). "fixed" method: config.REPLACEMENT_RANK at every position.
    """
    if config.REPLACEMENT_METHOD == "demand":
        cutoffs = position_demand().round().astype(int)
    else:
        cutoffs = pd.Series(config.REPLACEMENT_RANK, index=POSITIONS)

    levels = {}
    for position in POSITIONS:
        values = players.loc[eligible(players, position), value_column].sort_values(ascending=False)
        levels[position] = values.iloc[min(cutoffs[position], len(values) - 1)]
    return pd.Series(levels)


def vorp(players, value_column):
    """Value over replacement player for every player.

    Each player is measured against his best position (the one with the lowest
    replacement level), which gives him the highest VORP.
    """
    levels = replacement_levels(players, value_column)
    best_level = players["POSITIONS"].apply(lambda positions: levels[positions].min())
    return players[value_column] - best_level


def strategy_scores(players):
    """Every strategy's score for every player, one column per strategy.

    Scores are calculated once from the full pool before the draft. Each strategy reads
    only its own projection column, never actual points.
    """
    scores = pd.DataFrame(index=players.index)
    for name, settings in config.STRATEGIES.items():
        if settings["ranking"] == "vorp":
            scores[name] = vorp(players, settings["value"])
        else:
            scores[name] = players[settings["value"]]
    return scores


def make_strategy(scores, name):
    """A pick function for the draft engine: take the allowed player with the highest score."""
    def pick(team, allowed):
        best_row = scores.loc[allowed.index, name].idxmax()
        return allowed.loc[best_row, "PLAYER_ID"]
    return pick


def build_strategies(players):
    """Pick functions for all six strategies, keyed by name, ready for run_draft."""
    scores = strategy_scores(players)
    return {name: make_strategy(scores, name) for name in config.STRATEGIES}


if __name__ == "__main__":
    import numpy as np

    from src.draft import Team, run_draft
    from src.model import load_projections

    players = load_projections()
    rng = np.random.default_rng(config.RANDOM_SEED)
    seats = rng.permutation(list(config.STRATEGIES) * config.TEAMS_PER_STRATEGY).tolist()
    teams = [Team(f"Team {i + 1}", name) for i, name in enumerate(seats)]

    picks = run_draft(teams, players, build_strategies(players))
    picks = picks.merge(players[["PLAYER_ID", "PLAYER_NAME", "POSITIONS"]], on="PLAYER_ID")

    print("First four picks by team:")
    for team in teams:
        first_four = ", ".join(picks.loc[picks["TEAM"] == team.name, "PLAYER_NAME"].head(4))
        print(f"  {team.name} ({team.strategy_name}): {first_four}")

    print("\nPlayers drafted at each position, by strategy (two-position players count at both):")
    position_mix = picks.explode("POSITIONS").groupby(["STRATEGY", "POSITIONS"]).size().unstack(fill_value=0)
    print(position_mix[["PG", "SG", "SF", "PF", "C"]].to_string())