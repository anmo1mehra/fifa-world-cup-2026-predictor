"""Mid-tournament re-run helper.

Usage:
    python -m src.update --results data/raw/results_so_far.csv [--n 5000]

`results_so_far.csv` schema (one row per played match):
    stage,team_a,team_b,score_a,score_b,winner
    Group,Mexico,South Africa,2,1,
    Round of 32,Brazil,France,,,Brazil

Notes:
    - For Group rows, score_a/score_b are required, winner is ignored.
    - For knockout rows, winner is required (in case the match went to pens).
    - Team names can be either the canonical name (e.g. "United States") or the
      FIFA 3-letter code (e.g. "USA"); both are recognized.

This thin wrapper just calls into src.simulate with the fixed_* dicts populated.
"""
from __future__ import annotations

import argparse

from .simulate import main as simulate_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="CSV of already-played matches to lock in")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Hand off to src.simulate's CLI by spoofing argv.
    import sys
    sys.argv = ["simulate", "--n", str(args.n),
                "--seed", str(args.seed), "--results", args.results]
    simulate_main()


if __name__ == "__main__":
    main()
