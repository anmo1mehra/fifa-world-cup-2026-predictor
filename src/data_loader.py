"""Load, clean and normalize the three raw datasets into processed parquet files.

Run:
    python -m src.data_loader

Outputs (in data/processed/):
    matches.parquet      - international matches with canonical team names (2006+)
    rankings.parquet     - FIFA rankings with canonical team names
    schedule.parquet     - 2026 World Cup schedule with canonical names + match_id
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .teams import CODE_TO_CANONICAL, canonical

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

MIN_YEAR = 2006  # historical window used for training (matches the project brief)


def load_matches() -> pd.DataFrame:
    """Load historical matches, filter to >= MIN_YEAR, normalize team names."""
    df = pd.read_csv(RAW / "results.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df = df[df["date"].dt.year >= MIN_YEAR].copy()

    df["home_team"] = df["home_team"].map(canonical)
    df["away_team"] = df["away_team"].map(canonical)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Categorical outcome from the home team's perspective.
    df["outcome"] = "D"
    df.loc[df["home_score"] > df["away_score"], "outcome"] = "H"
    df.loc[df["home_score"] < df["away_score"], "outcome"] = "A"

    # Tournament importance weight - used for Elo K-factor and as a feature.
    def importance(t: str) -> float:
        t = (t or "").lower()
        if "fifa world cup" in t and "qualification" not in t:
            return 60.0
        if "uefa euro" in t or "copa américa" in t or "copa america" in t:
            return 50.0
        if "african cup" in t or "afc asian cup" in t or "gold cup" in t:
            return 40.0
        if "qualification" in t or "qualifier" in t:
            return 35.0
        if "uefa nations" in t or "concacaf nations" in t:
            return 30.0
        return 20.0  # friendlies and minor tournaments

    df["importance"] = df["tournament"].map(importance)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_rankings() -> pd.DataFrame:
    """Load FIFA rankings, normalize team names."""
    df = pd.read_csv(RAW / "fifa_ranking.csv")
    df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")
    df["team"] = df["country_full"].map(canonical)
    df = df[["team", "rank", "total_points", "confederation", "rank_date"]]
    df = df.sort_values("rank_date").reset_index(drop=True)
    return df


def _parse_schedule_year(date_str: str, time_str: str) -> pd.Timestamp:
    """Schedule rows look like 'Fri 12 Jun' / '03:00'. World Cup runs Jun-Jul 2026."""
    # Drop leading day-of-week if present.
    parts = str(date_str).split()
    if len(parts) == 3:
        date_str = " ".join(parts[1:])
    return pd.to_datetime(f"2026 {date_str} {time_str}", format="%Y %d %b %H:%M",
                          errors="coerce")


def load_schedule() -> pd.DataFrame:
    """Load the 2026 schedule, normalize team names, assign stable match_ids."""
    df = pd.read_excel(RAW / "schedule.xlsx", header=None)
    df.columns = ["date", "time", "round", "group", "fixture"]
    df = df.dropna(subset=["fixture"])
    df = df[df["fixture"] != "Fixture"].copy()

    df["fixture"] = df["fixture"].astype(str)
    parts = df["fixture"].str.split(" vs ", n=1, expand=True)
    df["team_a_code"] = parts[0].str.strip()
    df["team_b_code"] = parts[1].str.strip()

    def code_or_none(c: str) -> str | None:
        c = (c or "").strip().upper()
        return CODE_TO_CANONICAL.get(c)  # None for "TBC" knockout placeholders

    df["team_a"] = df["team_a_code"].map(code_or_none)
    df["team_b"] = df["team_b_code"].map(code_or_none)

    df["datetime"] = df.apply(
        lambda r: _parse_schedule_year(r["date"], r["time"]), axis=1
    )

    df = df.sort_values(["datetime", "round"]).reset_index(drop=True)
    df["match_id"] = range(1, len(df) + 1)
    df = df[["match_id", "datetime", "round", "group",
             "team_a_code", "team_b_code", "team_a", "team_b"]]
    return df


def build_processed() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print("Loading historical matches...")
    matches = load_matches()
    matches.to_parquet(PROCESSED / "matches.parquet", index=False)
    print(f"  -> {len(matches):,} matches saved "
          f"({matches['date'].min().date()} .. {matches['date'].max().date()})")

    print("Loading FIFA rankings...")
    rankings = load_rankings()
    rankings.to_parquet(PROCESSED / "rankings.parquet", index=False)
    print(f"  -> {len(rankings):,} ranking rows saved "
          f"({rankings['rank_date'].min().date()} .. {rankings['rank_date'].max().date()})")

    print("Loading 2026 schedule...")
    schedule = load_schedule()
    schedule.to_parquet(PROCESSED / "schedule.parquet", index=False)
    n_group = (schedule["round"] == "Group Stage").sum()
    n_ko = len(schedule) - n_group
    print(f"  -> {len(schedule)} fixtures ({n_group} group + {n_ko} knockout)")
    print("Done.")


if __name__ == "__main__":
    build_processed()
