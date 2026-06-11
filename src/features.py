"""Feature engineering: walks chronologically through history, computing for
every match the features available BEFORE kickoff (no leakage).

Features per match (home-perspective):
    elo_home, elo_away, elo_diff           - Elo rating updated chronologically
    rank_home, rank_away, rank_diff        - FIFA ranking (latest snapshot <= match date)
    points_home, points_away, points_diff  - FIFA ranking points
    form_home, form_away                   - average goal difference over last 5 matches
    winrate_home, winrate_away             - win rate over last 10 matches
    rest_home, rest_away                   - days since each team's last match (capped)
    h2h_home_winrate                       - home team's win rate vs this opponent (last 5)
    neutral, importance, is_world_cup

The Elo state at the end of the historical sweep is exported so the simulator
can use the same numbers going into 2026.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

INITIAL_ELO = 1500.0
ELO_HOME_ADV = 65.0  # standard "home field" bump used in the prediction step


def expected_score(elo_a: float, elo_b: float, home_adv: float = 0.0) -> float:
    """Logistic expectation, classic Elo formula."""
    return 1.0 / (1.0 + 10 ** (-((elo_a + home_adv) - elo_b) / 400.0))


def k_factor(importance: float, goal_diff: int) -> float:
    """K grows with match importance and a goal-difference multiplier (FIFA-style)."""
    margin_mult = 1.0
    if abs(goal_diff) == 2:
        margin_mult = 1.5
    elif abs(goal_diff) >= 3:
        margin_mult = (11.0 + abs(goal_diff)) / 8.0
    return importance * margin_mult


@dataclass
class TeamState:
    elo: float = INITIAL_ELO
    last_played: pd.Timestamp | None = None
    recent_results: deque = field(default_factory=lambda: deque(maxlen=10))  # 1/0.5/0
    recent_gd: deque = field(default_factory=lambda: deque(maxlen=5))


def _latest_ranking(rankings: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """Return rankings indexed by snapshot date for fast as-of lookup.
    If a team appears twice on the same date (rare, due to renames merging
    via aliases), keep the first occurrence."""
    grouped = {}
    for d, g in rankings.groupby("rank_date"):
        g = g.drop_duplicates(subset=["team"], keep="first")
        grouped[d] = g.set_index("team")
    return grouped


def _ranking_as_of(
    snapshots: list[pd.Timestamp],
    by_date: dict[pd.Timestamp, pd.DataFrame],
    target: pd.Timestamp,
    team: str,
) -> tuple[float, float]:
    """Find the most recent ranking snapshot on or before `target`."""
    idx = np.searchsorted(snapshots, target, side="right") - 1
    while idx >= 0:
        snap = by_date[snapshots[idx]]
        if team in snap.index:
            row = snap.loc[team]
            return float(row["rank"]), float(row["total_points"])
        idx -= 1
    return 200.0, 0.0  # unranked default


def build_features(matches: pd.DataFrame, rankings: pd.DataFrame
                   ) -> tuple[pd.DataFrame, dict[str, TeamState], dict[tuple[str, str], deque]]:
    """Stream through `matches` chronologically, producing one feature row per match
    plus the final team-state dict (Elo etc.) used by the simulator."""
    states: dict[str, TeamState] = defaultdict(TeamState)
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=5))

    rank_snapshots = sorted(rankings["rank_date"].unique())
    rank_by_date = _latest_ranking(rankings)

    rows = []
    for r in matches.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if h is None or a is None:
            continue
        sh, sa = states[h], states[a]

        rank_h, pts_h = _ranking_as_of(rank_snapshots, rank_by_date, r.date, h)
        rank_a, pts_a = _ranking_as_of(rank_snapshots, rank_by_date, r.date, a)

        rest_h = (r.date - sh.last_played).days if sh.last_played is not None else 30
        rest_a = (r.date - sa.last_played).days if sa.last_played is not None else 30
        rest_h = min(rest_h, 60)
        rest_a = min(rest_a, 60)

        form_h = float(np.mean(sh.recent_gd)) if sh.recent_gd else 0.0
        form_a = float(np.mean(sa.recent_gd)) if sa.recent_gd else 0.0
        wr_h = float(np.mean(sh.recent_results)) if sh.recent_results else 0.5
        wr_a = float(np.mean(sa.recent_results)) if sa.recent_results else 0.5

        h2h_key = tuple(sorted([h, a]))
        h2h_hist = h2h[h2h_key]
        h2h_home_wr = (
            float(np.mean([1.0 if w == h else 0.5 if w is None else 0.0 for w in h2h_hist]))
            if h2h_hist
            else 0.5
        )

        is_wc = 1 if "fifa world cup" in str(r.tournament).lower() and "qual" not in str(r.tournament).lower() else 0
        neutral = 1 if bool(r.neutral) else 0

        home_adv = 0.0 if neutral else ELO_HOME_ADV
        expected_h = expected_score(sh.elo, sa.elo, home_adv)

        rows.append({
            "date": r.date,
            "home_team": h,
            "away_team": a,
            "outcome": r.outcome,
            "elo_home": sh.elo,
            "elo_away": sa.elo,
            "elo_diff": sh.elo - sa.elo + home_adv,
            "rank_home": rank_h,
            "rank_away": rank_a,
            "rank_diff": rank_a - rank_h,  # positive => home higher-ranked
            "points_home": pts_h,
            "points_away": pts_a,
            "points_diff": pts_h - pts_a,
            "form_home": form_h,
            "form_away": form_a,
            "winrate_home": wr_h,
            "winrate_away": wr_a,
            "rest_home": rest_h,
            "rest_away": rest_a,
            "h2h_home_winrate": h2h_home_wr,
            "neutral": neutral,
            "importance": r.importance,
            "is_world_cup": is_wc,
            "expected_home_score": expected_h,
        })

        # ----- update state -----
        gd = r.home_score - r.away_score
        if gd > 0:
            res_h, res_a, winner = 1.0, 0.0, h
        elif gd < 0:
            res_h, res_a, winner = 0.0, 1.0, a
        else:
            res_h, res_a, winner = 0.5, 0.5, None

        k = k_factor(r.importance, gd)
        sh.elo += k * (res_h - expected_h)
        sa.elo += k * (res_a - (1 - expected_h))

        sh.recent_results.append(res_h)
        sa.recent_results.append(res_a)
        sh.recent_gd.append(gd)
        sa.recent_gd.append(-gd)
        sh.last_played = r.date
        sa.last_played = r.date
        h2h[h2h_key].append(winner)

    return pd.DataFrame(rows), dict(states), {k: list(v) for k, v in h2h.items()}


FEATURE_COLS = [
    "elo_diff", "rank_diff", "points_diff",
    "form_home", "form_away", "winrate_home", "winrate_away",
    "rest_home", "rest_away", "h2h_home_winrate",
    "neutral", "importance", "is_world_cup",
    "expected_home_score",
]


def main() -> None:
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    rankings = pd.read_parquet(PROCESSED / "rankings.parquet")
    print(f"Building features for {len(matches):,} matches...")
    feats, states, h2h = build_features(matches, rankings)
    feats.to_parquet(PROCESSED / "features.parquet", index=False)

    state_df = pd.DataFrame([
        {"team": t, "elo": s.elo,
         "last_played": s.last_played,
         "recent_gd": list(s.recent_gd),
         "recent_results": list(s.recent_results)}
        for t, s in states.items()
    ])
    state_df.to_parquet(PROCESSED / "team_state.parquet", index=False)

    h2h_df = pd.DataFrame([
        {"team_a": a, "team_b": b, "history": hist}
        for (a, b), hist in h2h.items()
    ])
    h2h_df.to_parquet(PROCESSED / "h2h.parquet", index=False)

    print(f"Features saved: {len(feats):,} rows, {len(FEATURE_COLS)} feature cols")
    print(f"Final team Elo (top 10):")
    top = state_df.sort_values("elo", ascending=False).head(10)
    for _, r in top.iterrows():
        print(f"  {r['team']:25s} {r['elo']:7.1f}")


if __name__ == "__main__":
    main()
