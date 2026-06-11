# 🏆 FIFA World Cup 2026 — Match & Top-Scorer Predictor

A complete machine-learning pipeline that predicts the winner of every match
in the **2026 FIFA World Cup** — from group stage to the final — and the
**Golden Boot top scorer**, all via Monte Carlo simulation.

- **XGBoost** 3-class (home-win / draw / away-win) classifier
- Trained on ~20 years of international fixtures (2006 – present)
- Custom Elo ratings, FIFA rankings, form, head-to-head, rest days, host advantage
- **5,000 Monte Carlo** tournament simulations
- Per-player goal sampling for **Golden Boot** prediction
- Interactive **Streamlit dashboard** with country flags, heatmaps, group tabs & player rankings

## Sample predictions

| # | Team | P(Champion) |
|---|---|---:|
| 1 | 🇦🇷 Argentina | **31.7%** |
| 2 | 🇪🇸 Spain | **16.1%** |
| 3 | 🇧🇷 Brazil | **11.5%** |
| 4 | 🇲🇽 Mexico (host) | **7.9%** |
| 5 | 🇫🇷 France | **7.0%** |

| # | Player | Team | P(Golden Boot) | xG |
|---|---|---|---:|---:|
| 1 | Lionel Messi | 🇦🇷 Argentina | **11.6%** | 2.70 |
| 2 | Kylian Mbappé | 🇫🇷 France | **10.8%** | 2.59 |
| 3 | Enner Valencia | 🇪🇨 Ecuador | **9.4%** | 2.44 |
| 4 | Harry Kane | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | **8.6%** | 2.41 |
| 5 | Erling Haaland | 🇳🇴 Norway | **4.6%** | 1.82 |

## Project layout

```text
fifa/
├── data/
│   ├── raw/                     <- results.csv, fifa_ranking.csv, schedule.xlsx, goalscorers.csv
│   └── processed/               <- generated parquet files (gitignored)
├── src/
│   ├── teams.py                 <- canonical team-name mapping + flag helpers
│   ├── data_loader.py           <- cleans + normalizes raw data
│   ├── features.py              <- Elo, form, rest, h2h ... feature engineering
│   ├── scorers.py               <- builds per-team player goal distributions
│   ├── model.py                 <- trains the XGBoost model
│   ├── simulate.py              <- 5000-run Monte Carlo simulator + Golden Boot tracking
│   └── update.py                <- mid-tournament re-run helper
├── dashboard/
│   └── app.py                   <- Streamlit dashboard
├── models/                      <- trained model.pkl (gitignored, generated)
├── outputs/                     <- simulation results (gitignored, generated)
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/fifa-world-cup-2026-predictor.git
cd fifa-world-cup-2026-predictor
pip install -r requirements.txt

# Build everything (one-time, takes ~15 min total)
python -m src.data_loader      # 1. clean raw -> processed (~30s)
python -m src.features         # 2. Elo + features (~1m)
python -m src.scorers          # 3. player goal distributions (~5s)
python -m src.model            # 4. train XGBoost (~30s)
python -m src.simulate --n 5000   # 5. 5000 Monte Carlo sims (~12 min)

# Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`.

## Mid-tournament updates

When real-world results start coming in (group games, surprise upsets,
injuries), lock those into the simulation and re-roll the remaining matches.

1. Create a CSV at `data/raw/results_so_far.csv`:

   ```csv
   stage,team_a,team_b,score_a,score_b,winner
   Group,Mexico,South Africa,2,1,
   Group,South Korea,Czech Republic,1,1,
   Round of 32,Brazil,Norway,,,Brazil
   ```

2. Re-run:

   ```bash
   python -m src.update --results data/raw/results_so_far.csv --n 5000
   ```

3. Refresh the dashboard.

## Methodology

### Match-outcome model

Features per match, all computed as of kick-off (no leakage):

| Feature | What it captures |
|---|---|
| `elo_diff` | Custom Elo rating (FIFA K-factor, home advantage) |
| `rank_diff`, `points_diff` | Latest FIFA ranking + points |
| `form_home`, `form_away` | Goal difference average over last 5 matches |
| `winrate_home`, `winrate_away` | Win rate over last 10 matches |
| `rest_home`, `rest_away` | Days since each team's last match |
| `h2h_home_winrate` | Win rate vs this opponent (last 5 meetings) |
| `neutral`, `importance`, `is_world_cup` | Match context |
| `expected_home_score` | Elo expectation as a calibration anchor |

**Model:** `XGBClassifier(num_class=3, max_depth=5, n_estimators=400)`, multinomial log-loss
on a temporal split (train < 2022, test ≥ 2022).

**Results:** ~60% accuracy on the 3-class holdout (random baseline = 33%), log-loss ~0.90 — in line with published soccer-prediction papers.

### Tournament structure (2026 new 48-team format)

- 48 teams in 12 groups of 4 (each plays 3 group games)
- Top 2 of each group + 8 best 3rd-placed teams = 32 advance to **Round of 32**
- Single-elimination from there: R32 → R16 → QF → SF → Final
- Knockout draws resolved by a strength-weighted coin flip (representing penalties)
- **Host advantage**: USA, Canada and Mexico get the home-field Elo bump in matches they play

### Golden Boot model

- For each simulated match, the model gives `P(team_a_win), P(draw), P(team_b_win)` and a sampled scoreline.
- For each goal a team scores, a scorer is sampled from that team's player distribution.
- A player's probability in the distribution = `their goals since 2022 / team's total goals since 2022`.
- After 5,000 sims, each player has an **expected total goals** and a **P(Golden Boot)** = how often they led the tournament.

This naturally rewards both **scoring rate** and **expected number of matches played** — which is why Messi narrowly edges out Mbappé and Haaland in our predictions despite a lower per-90 rate (Argentina is most likely to play 7-8 games).

## Outputs (`outputs/`)

| File | What it contains |
|---|---|
| `simulation_summary.json` | Champion + Golden Boot favourites + top-scorer goal distribution |
| `stage_probabilities.csv` | P(team reaches stage X) for every team & stage |
| `group_probabilities.csv` | P(finish 1st) and P(advance) per team per group |
| `group_match_predictions.csv` | Per-fixture P(win/draw/loss) for the group stage |
| `knockout_pair_stats.csv` | Most likely knockout pairings + outcomes |
| `scorer_predictions.csv` | Per-player expected goals + P(Golden Boot) |
| `fixed_results.json` | Which real results were locked in for this run |

## Data sources

- **Historical results & goalscorers**: [Mart Jürisoo — International football results 1872-present (Kaggle, CC0)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- **FIFA Rankings**: [cashncarry — FIFA World Ranking 1992-2024 (Kaggle, public)](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)
- **2026 schedule**: official FIFA-published schedule
- **Country flags**: served via [flagcdn.com](https://flagcdn.com)

## Notes / caveats

- The included FIFA ranking snapshot is dated 2023-07-20. Drop a newer
  `fifa_ranking.csv` into `data/raw/` for sharper predictions; the loader
  always uses the latest snapshot it finds.
- The qualified team list is hard-coded in `src/teams.py` (`GROUPS`). Edit it
  if FIFA's final draw changes after the code was written.
- Knockout draws represent regulation + extra-time + penalties; we approximate
  the penalty shootout with a strength-weighted coin flip rather than modelling
  shootouts directly. (Shootout goals are intentionally **not** counted toward
  the Golden Boot, in line with FIFA's official rules.)
- The Golden Boot model uses each player's "share of team goals since 2022" as
  their per-goal probability. This is a strong heuristic but means a hot debutant
  with no prior international goals won't show up in our predictions until they
  start scoring.

## License

MIT — see `LICENSE`.

## Built with

`pandas` · `numpy` · `xgboost` · `scikit-learn` · `streamlit` · `plotly`
