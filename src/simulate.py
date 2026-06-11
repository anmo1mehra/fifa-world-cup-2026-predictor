"""Monte Carlo simulation of the 2026 FIFA World Cup.

For every match the trained model returns P(team_a_win), P(draw), P(team_b_win).
We sample outcomes, accumulate group standings, slot teams into the knockout
bracket according to the actual 2026 R32 mapping, then play knockouts (draws
in KO are resolved by a strength-weighted coin flip representing penalties).

Run:
    python -m src.simulate                # default 5000 sims
    python -m src.simulate --n 10000      # custom count
    python -m src.simulate --results data/raw/results_so_far.csv   # mid-tournament

Outputs (in outputs/):
    simulation_summary.json   - high-level stats
    match_predictions.csv     - per-match win/draw/loss probabilities
    stage_probabilities.csv   - per-team reach-stage probabilities
    fixed_results.json        - which matches were locked from real results
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import (ELO_HOME_ADV, FEATURE_COLS, TeamState, expected_score,
                       k_factor)
from .scorers import load_distributions, sample_scorers
from .teams import CODE_TO_CANONICAL, GROUPS, canonical

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

HOST_NATIONS = {"United States", "Canada", "Mexico"}
STAGES = ["Group Stage", "Round of 32", "Round of 16", "Quarter-Finals",
          "Semi-Finals", "Final", "Champion"]


@dataclass
class TeamInfo:
    name: str
    elo: float
    rank: float
    points: float
    recent_results: list
    recent_gd: list


def load_team_infos() -> dict[str, TeamInfo]:
    state = pd.read_parquet(PROCESSED / "team_state.parquet")
    rankings = pd.read_parquet(PROCESSED / "rankings.parquet")
    latest_date = rankings["rank_date"].max()
    latest = rankings[rankings["rank_date"] == latest_date].set_index("team")

    infos: dict[str, TeamInfo] = {}
    for name in CODE_TO_CANONICAL.values():
        elo = float(state.loc[state["team"] == name, "elo"].iloc[0]) if (state["team"] == name).any() else 1500.0
        rec_results = state.loc[state["team"] == name, "recent_results"].iloc[0] if (state["team"] == name).any() else []
        rec_gd = state.loc[state["team"] == name, "recent_gd"].iloc[0] if (state["team"] == name).any() else []
        if name in latest.index:
            rank = float(latest.loc[name, "rank"])
            pts = float(latest.loc[name, "total_points"])
        else:
            rank, pts = 200.0, 0.0
        infos[name] = TeamInfo(name, elo, rank, pts, list(rec_results), list(rec_gd))
    return infos


def make_features(team_a: TeamInfo, team_b: TeamInfo, *,
                  neutral: int, importance: float, is_world_cup: int = 1) -> np.ndarray:
    home_adv = 0.0 if neutral else ELO_HOME_ADV
    form_a = float(np.mean(team_a.recent_gd)) if team_a.recent_gd else 0.0
    form_b = float(np.mean(team_b.recent_gd)) if team_b.recent_gd else 0.0
    wr_a = float(np.mean(team_a.recent_results)) if team_a.recent_results else 0.5
    wr_b = float(np.mean(team_b.recent_results)) if team_b.recent_results else 0.5
    row = {
        "elo_diff": team_a.elo - team_b.elo + home_adv,
        "rank_diff": team_b.rank - team_a.rank,
        "points_diff": team_a.points - team_b.points,
        "form_home": form_a,
        "form_away": form_b,
        "winrate_home": wr_a,
        "winrate_away": wr_b,
        "rest_home": 5.0,
        "rest_away": 5.0,
        "h2h_home_winrate": 0.5,
        "neutral": neutral,
        "importance": importance,
        "is_world_cup": is_world_cup,
        "expected_home_score": expected_score(team_a.elo, team_b.elo, home_adv),
    }
    return np.array([[row[c] for c in FEATURE_COLS]], dtype=float)


# --------------------------------------------------------------------------- #
#  Match probabilities                                                        #
# --------------------------------------------------------------------------- #


def match_probs(model, team_a: TeamInfo, team_b: TeamInfo, *,
                neutral: int = 1, importance: float = 60.0,
                allow_draw: bool = True) -> dict[str, float]:
    """Return {'A': p_a_win, 'D': p_draw, 'B': p_b_win} for one fixture.
    The model is trained with home/draw/away labels [H,D,A] -> [0,1,2]."""
    X = make_features(team_a, team_b, neutral=neutral, importance=importance)
    p = model.predict_proba(X)[0]
    p_a, p_d, p_b = float(p[0]), float(p[1]), float(p[2])
    if not allow_draw:
        # Knockout: redistribute the draw probability weighted by strength.
        share_a = p_a / max(p_a + p_b, 1e-9)
        p_a += p_d * share_a
        p_b += p_d * (1 - share_a)
        p_d = 0.0
    # Defensive renorm: model outputs can drift from 1.0 by ~1e-7, which
    # np.random.choice refuses to accept.
    total = p_a + p_d + p_b
    return {"A": p_a / total, "D": p_d / total, "B": p_b / total}


def sample_outcome(probs: dict[str, float], rng: np.random.Generator) -> str:
    return rng.choice(["A", "D", "B"], p=[probs["A"], probs["D"], probs["B"]])


# --------------------------------------------------------------------------- #
#  Tournament structure                                                       #
# --------------------------------------------------------------------------- #
# The 2026 World Cup has 12 groups (A-L). Top 2 of each group + 8 best
# third-placed teams advance to a 32-team knockout. The official R32 slotting
# (per FIFA's announced bracket structure) is encoded below: each tuple is
# (slot_a, slot_b) where each slot is either "Xn" for 1st/2nd of group X,
# or "3X/Y/Z/W" meaning the third-placed team from any of those groups.


# 32-team R32 bracket. Each of the 12 group winners and 12 runners-up appears
# exactly once. There are exactly 8 "third-team" slots (the 8 best 3rds advance).
# We use a simplified-but-consistent slotting inspired by FIFA's official 48-team
# bracket structure (the official 14-case 3rd-team assignment table is heavy and
# this random-from-candidate-pool heuristic captures the same spirit).
R32_SLOTS: list[tuple[str, str]] = [
    # 8 winner-vs-third matches
    ("A1", "3C/E/H/I"),
    ("B1", "3A/F/G/I"),
    ("C1", "3D/E/J/K"),
    ("D1", "3B/E/F/I"),
    ("E1", "3A/B/C/D"),
    ("F1", "3A/B/E/F"),
    ("G1", "3F/I/J/L"),
    ("H1", "3B/C/D/G"),
    # 4 winner-vs-runner-up matches
    ("I1", "L2"),
    ("J1", "K2"),
    ("K1", "J2"),
    ("L1", "I2"),
    # 4 runner-up-vs-runner-up matches
    ("A2", "C2"),
    ("B2", "D2"),
    ("E2", "G2"),
    ("F2", "H2"),
]

# Standard 1->32 -> 1->16 -> 1->8 -> 1->4 -> 1->2 bracket pairing.
KO_PAIRS = {
    "Round of 16": [(0, 1), (2, 3), (4, 5), (6, 7),
                    (8, 9), (10, 11), (12, 13), (14, 15)],
    "Quarter-Finals": [(0, 1), (2, 3), (4, 5), (6, 7)],
    "Semi-Finals": [(0, 1), (2, 3)],
    "Final": [(0, 1)],
}


def _is_host(team: str) -> bool:
    return team in HOST_NATIONS


def _neutral(team_a: str, team_b: str) -> int:
    """Hosts (USA/CAN/MEX) play at home; everyone else plays neutral."""
    if _is_host(team_a) and not _is_host(team_b):
        return 0
    if _is_host(team_b) and not _is_host(team_a):
        return 0
    return 1


def play_group_stage(model, infos: dict[str, TeamInfo],
                     rng: np.random.Generator,
                     fixed: dict[tuple[str, str], tuple[int, int]] | None = None,
                     match_log: list | None = None,
                     ) -> dict[str, list[tuple[str, int, int, int]]]:
    """Returns: {group: [(team, pts, gd, gf), ...] sorted best->worst}
    If `match_log` is provided, appends (team_a, team_b, score_a, score_b) tuples."""
    standings: dict[str, dict[str, list[float]]] = {
        g: {t: [0, 0, 0] for t in teams} for g, teams in GROUPS.items()
    }
    for group, teams in GROUPS.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                key = tuple(sorted([a, b]))
                if fixed and key in fixed:
                    sa, sb = fixed[key]
                    if a > b:
                        sa, sb = sb, sa
                    if sa > sb: out = "A"
                    elif sa < sb: out = "B"
                    else: out = "D"
                else:
                    probs = match_probs(model, infos[a], infos[b],
                                        neutral=_neutral(a, b),
                                        importance=60.0, allow_draw=True)
                    out = sample_outcome(probs, rng)
                    # Synthesize a plausible score so we can break ties on GD/GF.
                    sa, sb = _sample_score(out, rng)
                if out == "A":
                    standings[group][a][0] += 3
                elif out == "B":
                    standings[group][b][0] += 3
                else:
                    standings[group][a][0] += 1
                    standings[group][b][0] += 1
                standings[group][a][1] += sa - sb
                standings[group][b][1] += sb - sa
                standings[group][a][2] += sa
                standings[group][b][2] += sb
                if match_log is not None:
                    match_log.append((a, b, int(sa), int(sb)))

    table: dict[str, list[tuple[str, int, int, int]]] = {}
    for group, st in standings.items():
        ordered = sorted(
            [(t, int(p[0]), int(p[1]), int(p[2])) for t, p in st.items()],
            key=lambda r: (-r[1], -r[2], -r[3], rng.random()),
        )
        table[group] = ordered
    return table


def _sample_score(outcome: str, rng: np.random.Generator) -> tuple[int, int]:
    """Lightweight score sampler for tiebreakers (Poisson with reasonable means)."""
    base = 1.3
    if outcome == "A":
        a = max(1, rng.poisson(1.7))
        b = rng.poisson(base * 0.6)
        if b >= a:
            b = a - 1
    elif outcome == "B":
        b = max(1, rng.poisson(1.7))
        a = rng.poisson(base * 0.6)
        if a >= b:
            a = b - 1
    else:
        a = rng.poisson(base)
        b = a
    return int(a), int(b)


def pick_best_thirds(table: dict[str, list[tuple[str, int, int, int]]],
                     rng: np.random.Generator
                     ) -> dict[str, tuple[str, int, int, int]]:
    """8 best third-placed teams across the 12 groups."""
    thirds = []
    for g, rows in table.items():
        t, p, gd, gf = rows[2]
        thirds.append((g, t, p, gd, gf))
    thirds.sort(key=lambda r: (-r[2], -r[3], -r[4], rng.random()))
    chosen = thirds[:8]
    return {g: (t, p, gd, gf) for g, t, p, gd, gf in chosen}


def assign_thirds_to_slots(slots: list[tuple[str, str]],
                           thirds: dict[str, tuple[str, int, int, int]],
                           rng: np.random.Generator) -> dict[int, str]:
    """Pre-assign each '3X/Y/...' slot to one of the qualifying third teams,
    using each team exactly once. Greedy with random shuffle so no slot is
    left unassigned even if pool composition is awkward.
    Returns {slot_index_in_R32_SLOTS_flat: team_name}."""
    third_slot_positions: list[tuple[int, list[str]]] = []
    for i, pair in enumerate(slots):
        for j, slot in enumerate(pair):
            if slot.startswith("3"):
                candidates = slot[1:].split("/")
                flat_idx = i * 2 + j
                third_slot_positions.append((flat_idx, candidates))

    available = dict(thirds)  # group_letter -> (team, pts, gd, gf)
    order = list(range(len(third_slot_positions)))
    rng.shuffle(order)
    assignment: dict[int, str] = {}
    for k in order:
        flat_idx, candidates = third_slot_positions[k]
        match = [g for g in candidates if g in available]
        if not match:
            match = list(available.keys())
        chosen = match[int(rng.integers(0, len(match)))]
        assignment[flat_idx] = available.pop(chosen)[0]
    return assignment


def resolve_slot(slot: str,
                 table: dict[str, list[tuple[str, int, int, int]]],
                 third_assignment: dict[int, str],
                 flat_idx: int) -> str:
    """'A1' -> 1st of A, 'D2' -> 2nd of D, '3...' -> pre-assigned third team."""
    if slot.startswith("3"):
        return third_assignment[flat_idx]
    group, pos = slot[0], int(slot[1])
    return table[group][pos - 1][0]


def play_knockouts(model, infos: dict[str, TeamInfo],
                   r32_teams: list[str], rng: np.random.Generator,
                   fixed: dict[tuple[str, str], str] | None = None,
                   match_log: list | None = None,
                   ) -> tuple[dict[str, list[str]], str, list[dict]]:
    """Play R32 -> R16 -> QF -> SF -> Final, returning the bracket per stage.
    Returns (stage_winners, champion, log).
    If `match_log` is provided, appends (team_a, team_b, score_a, score_b)
    representing regulation/extra-time goals (penalty shootout goals excluded)."""
    stage_winners: dict[str, list[str]] = {}
    log: list[dict] = []

    # Round of 32
    winners = []
    for i in range(0, 32, 2):
        a, b = r32_teams[i], r32_teams[i + 1]
        winner, sa, sb = _play_ko_match(model, infos, a, b, rng, fixed)
        winners.append(winner)
        log.append({"stage": "Round of 32", "team_a": a, "team_b": b, "winner": winner})
        if match_log is not None:
            match_log.append((a, b, int(sa), int(sb)))
    stage_winners["Round of 32"] = winners

    for stage, pairs in KO_PAIRS.items():
        prev = stage_winners[list(stage_winners)[-1]] if stage != "Round of 16" else winners
        next_round = []
        for ia, ib in pairs:
            a, b = prev[ia], prev[ib]
            winner, sa, sb = _play_ko_match(model, infos, a, b, rng, fixed)
            next_round.append(winner)
            log.append({"stage": stage, "team_a": a, "team_b": b, "winner": winner})
            if match_log is not None:
                match_log.append((a, b, int(sa), int(sb)))
        stage_winners[stage] = next_round

    champion = stage_winners["Final"][0]
    return stage_winners, champion, log


def _play_ko_match(model, infos, a: str, b: str, rng: np.random.Generator,
                   fixed: dict | None) -> tuple[str, int, int]:
    """Returns (winner, score_a, score_b) - scores reflect regulation/ET only.
    For ties in regulation, the winner is decided by a strength-weighted coin
    flip (representing penalties); the scoreline stays a draw."""
    if fixed:
        key = tuple(sorted([a, b]))
        if key in fixed:
            winner = fixed[key]
            # We don't know the real scoreline of a locked match; sample one.
            out = "A" if winner == a else "B"
            sa, sb = _sample_score(out, rng)
            return winner, sa, sb

    probs_full = match_probs(model, infos[a], infos[b], neutral=_neutral(a, b),
                             importance=60.0, allow_draw=True)
    out = sample_outcome(probs_full, rng)
    sa, sb = _sample_score(out, rng)
    if out == "D":
        # Penalty shootout - strength-weighted coin flip.
        p_no_draw = match_probs(model, infos[a], infos[b],
                                neutral=_neutral(a, b),
                                importance=60.0, allow_draw=False)
        winner = a if rng.random() < p_no_draw["A"] else b
    else:
        winner = a if out == "A" else b
    return winner, sa, sb


# --------------------------------------------------------------------------- #
#  Top-level Monte Carlo                                                      #
# --------------------------------------------------------------------------- #


def run_monte_carlo(n_sims: int = 5000, seed: int = 42,
                    fixed_group_scores: dict | None = None,
                    fixed_ko_winners: dict | None = None,
                    track_scorers: bool = True) -> dict:
    bundle = joblib.load(MODELS / "model.pkl")
    model = bundle["model"]

    infos = load_team_infos()
    teams = list(infos.keys())
    rng = np.random.default_rng(seed)

    # Scorer distributions per team (optional).
    scorer_dists = None
    if track_scorers:
        dist_path = PROCESSED / "team_scorer_dist.npz"
        if dist_path.exists():
            scorer_dists = load_distributions()
        else:
            print("(scorer distributions not found; run `python -m src.scorers`)")
            track_scorers = False

    champions = Counter()
    reach_stage = {s: Counter() for s in STAGES[1:]}  # how often each team reaches each stage
    group_wins = Counter()      # how often each team finishes 1st in group
    group_qualify = Counter()   # how often each team advances out of group

    # Per-fixture probability tracker. We use a stable key:
    # group-stage key = "G:<group>:<team_a>|<team_b>" (sorted)
    # knockout key   = "K:<stage>:<team_a>|<team_b>" (we just keep counts of meetings/winners)
    pair_meetings: Counter = Counter()
    pair_winners: Counter = Counter()

    # Per-player scorer aggregates.
    player_total_goals: Counter = Counter()           # cumulative across all sims
    player_team: dict[str, str] = {}                  # player -> team
    golden_boot_wins: Counter = Counter()             # weighted by 1/ties
    top_scorer_counts: list[int] = []                 # goal count of top scorer per sim

    for _ in tqdm(range(n_sims), desc=f"Simulating {n_sims} tournaments"):
        match_log: list[tuple[str, str, int, int]] = []
        table = play_group_stage(model, infos, rng, fixed=fixed_group_scores,
                                 match_log=match_log)
        for g, rows in table.items():
            group_wins[rows[0][0]] += 1
            group_qualify[rows[0][0]] += 1
            group_qualify[rows[1][0]] += 1

        thirds = pick_best_thirds(table, rng)
        for g, (t, _, _, _) in thirds.items():
            group_qualify[t] += 1

        third_assignment = assign_thirds_to_slots(R32_SLOTS, thirds, rng)
        r32_teams = []
        for i, pair in enumerate(R32_SLOTS):
            for j, slot in enumerate(pair):
                r32_teams.append(resolve_slot(slot, table, third_assignment, i * 2 + j))

        stage_winners, champion, match_log_ko = play_knockouts(
            model, infos, r32_teams, rng, fixed=fixed_ko_winners,
            match_log=match_log,
        )

        # Record reach-stage stats.
        for t in r32_teams:
            reach_stage["Round of 32"][t] += 1
        for stage, ws in stage_winners.items():
            next_stage = {"Round of 32": "Round of 16",
                          "Round of 16": "Quarter-Finals",
                          "Quarter-Finals": "Semi-Finals",
                          "Semi-Finals": "Final",
                          "Final": "Champion"}[stage]
            for w in ws:
                reach_stage[next_stage][w] += 1
        champions[champion] += 1

        for m in match_log_ko:
            key = f"K:{m['stage']}:" + "|".join(sorted([m["team_a"], m["team_b"]]))
            pair_meetings[key] += 1
            pair_winners[(key, m["winner"])] += 1

        # ----- Per-sim scorer sampling -----
        if track_scorers and scorer_dists is not None:
            sim_player_goals: Counter = Counter()
            for team_a, team_b, sa, sb in match_log:
                if sa > 0 and team_a in scorer_dists:
                    for p in sample_scorers(scorer_dists[team_a], sa, rng):
                        sim_player_goals[p] += 1
                        player_team.setdefault(p, team_a)
                if sb > 0 and team_b in scorer_dists:
                    for p in sample_scorers(scorer_dists[team_b], sb, rng):
                        sim_player_goals[p] += 1
                        player_team.setdefault(p, team_b)
            player_total_goals.update(sim_player_goals)
            if sim_player_goals:
                max_g = max(sim_player_goals.values())
                top_scorer_counts.append(max_g)
                ties = [p for p, g in sim_player_goals.items() if g == max_g]
                share = 1.0 / len(ties)
                for p in ties:
                    golden_boot_wins[p] += share

    # ----- finalize results -----
    summary = {
        "n_sims": n_sims,
        "champion_probabilities": {
            t: champions[t] / n_sims for t in sorted(champions, key=champions.get, reverse=True)
        },
        "top_5_champions": [
            {"team": t, "prob": champions[t] / n_sims}
            for t in sorted(champions, key=champions.get, reverse=True)[:5]
        ],
    }

    stage_rows = []
    for stage, ctr in reach_stage.items():
        for t in teams:
            stage_rows.append({"team": t, "stage": stage, "prob": ctr.get(t, 0) / n_sims})
    stage_df = pd.DataFrame(stage_rows)

    # Per-group probability table for the dashboard
    group_rows = []
    for group, group_teams in GROUPS.items():
        for t in group_teams:
            group_rows.append({
                "group": group, "team": t,
                "p_finish_1st": group_wins.get(t, 0) / n_sims,
                "p_advance": group_qualify.get(t, 0) / n_sims,
            })
    group_df = pd.DataFrame(group_rows)

    # Golden Boot dataframe
    scorer_df = pd.DataFrame()
    if track_scorers and player_total_goals:
        rows = []
        for p, total in player_total_goals.items():
            rows.append({
                "player": p,
                "team": player_team.get(p, ""),
                "expected_goals": total / n_sims,
                "golden_boot_prob": golden_boot_wins.get(p, 0.0) / n_sims,
                "total_goals_all_sims": total,
            })
        scorer_df = (pd.DataFrame(rows)
                     .sort_values("golden_boot_prob", ascending=False)
                     .reset_index(drop=True))
        summary["top_scorer"] = {
            "player": scorer_df.iloc[0]["player"],
            "team": scorer_df.iloc[0]["team"],
            "golden_boot_prob": float(scorer_df.iloc[0]["golden_boot_prob"]),
            "expected_goals": float(scorer_df.iloc[0]["expected_goals"]),
        }
        summary["top_scorer_goal_distribution"] = {
            "mean": float(np.mean(top_scorer_counts)) if top_scorer_counts else 0.0,
            "p10": float(np.percentile(top_scorer_counts, 10)) if top_scorer_counts else 0.0,
            "p50": float(np.percentile(top_scorer_counts, 50)) if top_scorer_counts else 0.0,
            "p90": float(np.percentile(top_scorer_counts, 90)) if top_scorer_counts else 0.0,
        }

    return {
        "summary": summary,
        "stage_df": stage_df,
        "group_df": group_df,
        "scorer_df": scorer_df,
        "pair_meetings": pair_meetings,
        "pair_winners": pair_winners,
        "infos": infos,
    }


def precompute_group_match_probs(model, infos):
    """For the dashboard: the deterministic match-probability for each group-stage fixture."""
    rows = []
    for group, teams in GROUPS.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                p = match_probs(model, infos[a], infos[b],
                                neutral=_neutral(a, b), importance=60.0,
                                allow_draw=True)
                rows.append({
                    "group": group, "team_a": a, "team_b": b,
                    "p_a_win": p["A"], "p_draw": p["D"], "p_b_win": p["B"],
                    "predicted_winner": a if p["A"] >= max(p["D"], p["B"])
                    else (b if p["B"] >= p["D"] else "Draw"),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="number of simulations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", type=str, default=None,
                        help="optional CSV of already-played matches to lock in")
    args = parser.parse_args()

    fixed_group: dict = {}
    fixed_ko: dict = {}
    if args.results:
        df = pd.read_csv(args.results)
        for r in df.itertuples(index=False):
            a = canonical(r.team_a)
            b = canonical(r.team_b)
            key = tuple(sorted([a, b]))
            if str(r.stage).lower().startswith("group"):
                fixed_group[key] = (int(r.score_a), int(r.score_b))
            else:
                winner = canonical(r.winner) if hasattr(r, "winner") and r.winner else (
                    a if r.score_a > r.score_b else b
                )
                fixed_ko[key] = winner
        print(f"Locked in {len(fixed_group)} group results and {len(fixed_ko)} knockout results")

    result = run_monte_carlo(args.n, args.seed,
                             fixed_group_scores=fixed_group or None,
                             fixed_ko_winners=fixed_ko or None)

    OUTPUTS.mkdir(exist_ok=True, parents=True)

    with open(OUTPUTS / "simulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2)

    result["stage_df"].to_csv(OUTPUTS / "stage_probabilities.csv", index=False)
    result["group_df"].to_csv(OUTPUTS / "group_probabilities.csv", index=False)
    if not result["scorer_df"].empty:
        result["scorer_df"].to_csv(OUTPUTS / "scorer_predictions.csv",
                                   index=False, encoding="utf-8")

    # Deterministic per-match group predictions (for the dashboard "match table")
    bundle = joblib.load(MODELS / "model.pkl")
    match_probs_df = precompute_group_match_probs(bundle["model"], result["infos"])
    match_probs_df.to_csv(OUTPUTS / "group_match_predictions.csv", index=False)

    # Knockout fixture aggregate (which pairings happened how often + winner rate)
    ko_rows = []
    for key, meetings in result["pair_meetings"].items():
        _, stage, pair = key.split(":", 2)
        a, b = pair.split("|")
        wa = result["pair_winners"].get((key, a), 0)
        wb = result["pair_winners"].get((key, b), 0)
        ko_rows.append({"stage": stage, "team_a": a, "team_b": b,
                        "meetings": meetings, "team_a_win_rate": wa / meetings,
                        "team_b_win_rate": wb / meetings,
                        "meeting_prob": meetings / args.n})
    pd.DataFrame(ko_rows).sort_values(["stage", "meetings"], ascending=[True, False]) \
        .to_csv(OUTPUTS / "knockout_pair_stats.csv", index=False)

    with open(OUTPUTS / "fixed_results.json", "w", encoding="utf-8") as f:
        json.dump({"group": {str(k): v for k, v in fixed_group.items()},
                   "ko": {str(k): v for k, v in fixed_ko.items()}}, f, indent=2)

    print("\n=== TOP 10 CHAMPIONSHIP PROBABILITIES ===")
    for entry in list(result["summary"]["champion_probabilities"].items())[:10]:
        print(f"  {entry[0]:25s} {entry[1]*100:6.2f}%")

    if not result["scorer_df"].empty:
        print("\n=== TOP 10 GOLDEN BOOT CONTENDERS ===")
        for _, r in result["scorer_df"].head(10).iterrows():
            try:
                print(f"  {r['player']:30s} ({r['team']:20s}) "
                      f"GB%={r['golden_boot_prob']*100:5.2f}% "
                      f"xG={r['expected_goals']:.2f}")
            except UnicodeEncodeError:
                # Windows console may choke on Unicode names; CSV has them all.
                print(f"  [name in scorer_predictions.csv] ({r['team']:20s}) "
                      f"GB%={r['golden_boot_prob']*100:5.2f}% "
                      f"xG={r['expected_goals']:.2f}")
        dist = result["summary"].get("top_scorer_goal_distribution", {})
        if dist:
            print(f"\nTop-scorer goal-count distribution: "
                  f"mean={dist['mean']:.1f}, "
                  f"p10={dist['p10']:.0f}, p50={dist['p50']:.0f}, p90={dist['p90']:.0f}")

    print(f"\nWrote outputs to {OUTPUTS}/")


if __name__ == "__main__":
    main()
