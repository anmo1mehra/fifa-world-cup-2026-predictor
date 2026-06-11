"""Build a per-team player-goal distribution from goalscorers.csv.

Methodology
-----------
- Keep goals from 2022-01-01 onwards (proxy for "currently active" players).
- Exclude own goals.
- For each of the 48 WC 2026 teams, every player who has scored at least once
  in that window gets a probability proportional to their goal count, so when
  the simulator decides a team scored N goals in a match it can sample N
  scorers from this distribution.

Run:
    python -m src.scorers

Outputs:
    data/processed/scorers.parquet      - per-player goals + team + scoring rate
    data/processed/team_scorer_dist.npz - team -> (players[], cumulative probs[])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .teams import WC_TEAMS, canonical

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Window we consider "active" players. Tweakable - shorter = sharper for
# current form, longer = fewer cold-start zeros.
ACTIVE_FROM = "2022-01-01"


def load_scorer_history(min_date: str = ACTIVE_FROM) -> pd.DataFrame:
    """Load goalscorers.csv, filter to 48 WC teams + recent window, exclude OGs."""
    df = pd.read_csv(RAW / "goalscorers.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "scorer", "team"])
    df = df[df["date"] >= pd.Timestamp(min_date)]
    df = df[~df["own_goal"].astype(bool)]

    df["team"] = df["team"].map(canonical)
    df = df[df["team"].isin(WC_TEAMS)]
    df["penalty"] = df["penalty"].astype(bool)
    return df.reset_index(drop=True)


def build_team_distributions(history: pd.DataFrame
                             ) -> dict[str, dict[str, np.ndarray]]:
    """Per team, return arrays of (players, cumulative probability) for fast
    sampling via np.searchsorted."""
    dists: dict[str, dict[str, np.ndarray]] = {}
    for team in WC_TEAMS:
        sub = history[history["team"] == team]
        if sub.empty:
            # Cold-start fallback: a single anonymous scorer.
            dists[team] = {
                "players": np.array([f"Unknown ({team})"], dtype=object),
                "cumprob": np.array([1.0]),
                "is_penalty": np.array([0.5]),
            }
            continue
        counts = sub.groupby("scorer").agg(
            goals=("scorer", "size"),
            penalties=("penalty", "sum"),
        ).sort_values("goals", ascending=False)
        probs = counts["goals"].to_numpy() / counts["goals"].sum()
        dists[team] = {
            "players": counts.index.to_numpy(dtype=object),
            "cumprob": np.cumsum(probs),
            "is_penalty": (counts["penalties"] / counts["goals"]).to_numpy(),
        }
    return dists


def sample_scorers(team_dist: dict[str, np.ndarray], n_goals: int,
                   rng: np.random.Generator) -> list[str]:
    """Sample n_goals scorers using the team's cumulative-probability vector."""
    if n_goals <= 0:
        return []
    u = rng.uniform(size=n_goals)
    idx = np.searchsorted(team_dist["cumprob"], u, side="right")
    idx = np.clip(idx, 0, len(team_dist["players"]) - 1)
    return list(team_dist["players"][idx])


def save_distributions(dists: dict[str, dict[str, np.ndarray]]) -> None:
    """Persist distributions to disk as one .npz file (compact + fast to load)."""
    PROCESSED.mkdir(exist_ok=True, parents=True)
    payload = {}
    for team, d in dists.items():
        safe = team.replace(" ", "_").replace("'", "")
        payload[f"{safe}__players"] = d["players"]
        payload[f"{safe}__cumprob"] = d["cumprob"]
    np.savez_compressed(PROCESSED / "team_scorer_dist.npz", **payload)


def load_distributions() -> dict[str, dict[str, np.ndarray]]:
    """Inverse of save_distributions()."""
    arr = np.load(PROCESSED / "team_scorer_dist.npz", allow_pickle=True)
    dists: dict[str, dict[str, np.ndarray]] = {}
    for team in WC_TEAMS:
        safe = team.replace(" ", "_").replace("'", "")
        if f"{safe}__players" not in arr:
            continue
        dists[team] = {
            "players": arr[f"{safe}__players"],
            "cumprob": arr[f"{safe}__cumprob"],
        }
    return dists


def main() -> None:
    print("Loading goalscorer history...")
    history = load_scorer_history()
    print(f"  {len(history):,} goals from {history['date'].min().date()} "
          f"to {history['date'].max().date()} across "
          f"{history['team'].nunique()} WC teams")

    print("Building per-team distributions...")
    dists = build_team_distributions(history)

    # Also save a flat per-player parquet for the dashboard.
    flat = history.groupby(["scorer", "team"]).agg(
        goals=("scorer", "size"),
        penalties=("penalty", "sum"),
        first_goal=("date", "min"),
        last_goal=("date", "max"),
    ).reset_index().sort_values("goals", ascending=False)
    flat.to_parquet(PROCESSED / "scorers.parquet", index=False)

    save_distributions(dists)

    print(f"\nTop 15 international scorers since {ACTIVE_FROM}:")
    for _, r in flat.head(15).iterrows():
        try:
            print(f"  {r['scorer']:30s} {r['team']:20s} {r['goals']} goals")
        except UnicodeEncodeError:
            safe = r["scorer"].encode("ascii", "replace").decode()
            print(f"  {safe:30s} {r['team']:20s} {r['goals']} goals")
    print(f"\nSaved: {PROCESSED / 'scorers.parquet'}, "
          f"{PROCESSED / 'team_scorer_dist.npz'}")


if __name__ == "__main__":
    main()
