"""Streamlit dashboard for the 2026 FIFA World Cup predictor.

Run from the project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make the `src/` package importable when running from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.teams import flag_emoji, flag_url, with_flag  # noqa: E402

OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"

STAGES_ORDER = ["Round of 32", "Round of 16", "Quarter-Finals",
                "Semi-Finals", "Final", "Champion"]

st.set_page_config(page_title="FIFA WC 2026 Predictor",
                   page_icon="🏆", layout="wide")

# Custom CSS tweaks for a nicer look.
st.markdown(
    """
    <style>
    .champion-card {
        background: linear-gradient(135deg, #6e0014 0%, #b8860b 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 24px;
    }
    .champion-name {
        font-size: 42px;
        font-weight: 700;
        margin: 12px 0 4px 0;
    }
    .champion-prob {
        font-size: 22px;
        opacity: 0.9;
    }
    .champion-flag {
        height: 88px;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .team-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .team-row img {
        width: 28px;
        height: 21px;
        border-radius: 2px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .vs-pill {
        background: #f0f2f6;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        color: #888;
        margin: 0 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏆 FIFA World Cup 2026 — Match Predictor")
st.caption(
    "XGBoost win/draw/loss model trained on ~20 years of international matches, "
    "validated via 5,000 Monte Carlo tournament simulations."
)

# ---------------- load outputs ----------------

if not (OUTPUTS / "simulation_summary.json").exists():
    st.error("No simulation results found. Run `python -m src.simulate` first.")
    st.stop()

summary = json.loads((OUTPUTS / "simulation_summary.json").read_text(encoding="utf-8"))
stage_df = pd.read_csv(OUTPUTS / "stage_probabilities.csv")
group_df = pd.read_csv(OUTPUTS / "group_probabilities.csv")
match_df = pd.read_csv(OUTPUTS / "group_match_predictions.csv")

try:
    ko_pairs = pd.read_csv(OUTPUTS / "knockout_pair_stats.csv")
except FileNotFoundError:
    ko_pairs = pd.DataFrame()

try:
    scorer_df = pd.read_csv(OUTPUTS / "scorer_predictions.csv", encoding="utf-8")
except FileNotFoundError:
    scorer_df = pd.DataFrame()

n_sims = summary["n_sims"]


# Helper: render a row with a flag image + team name in markdown
def team_html(team: str, size: str = "28x21") -> str:
    url = flag_url(team, size=size)
    if url:
        return (f"<span class='team-row'><img src='{url}' alt='{team} flag'/>"
                f"<span>{team}</span></span>")
    return f"<span class='team-row'><span>{team}</span></span>"


# ---------------- headline ----------------

champ_probs = summary["champion_probabilities"]
top_team = next(iter(champ_probs))
top_p = champ_probs[top_team]

flag_html = ""
if flag_url(top_team, "320x240"):
    flag_html = f"<img class='champion-flag' src='{flag_url(top_team, '320x240')}'/>"

st.markdown(
    f"""
    <div class='champion-card'>
        <div style='font-size:14px; letter-spacing:2px; opacity:0.85;'>PREDICTED CHAMPION</div>
        {flag_html}
        <div class='champion-name'>{top_team}</div>
        <div class='champion-prob'>{top_p*100:.1f}% probability of lifting the cup</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col2, col3, col4 = st.columns(3)
with col2:
    st.metric("Simulations", f"{n_sims:,}")
with col3:
    teams_with_chance = sum(1 for p in champ_probs.values() if p > 0)
    st.metric("Teams with non-zero chance", teams_with_chance)
with col4:
    runner_up = list(champ_probs.items())[1]
    st.metric("Next most likely", f"{runner_up[0]}", f"{runner_up[1]*100:.1f}%")

# ---------------- champion probabilities ----------------

st.subheader("Championship probability — top 15")
top_df = pd.DataFrame(
    [{"team": with_flag(t), "prob": p}
     for t, p in list(champ_probs.items())[:15]]
)
fig = px.bar(
    top_df, x="prob", y="team", orientation="h",
    text=top_df["prob"].apply(lambda p: f"{p*100:.1f}%"),
    color="prob", color_continuous_scale="Tealgrn",
)
fig.update_layout(
    yaxis=dict(autorange="reversed", title=""),
    xaxis=dict(tickformat=".0%", title="Probability"),
    height=520, margin=dict(l=10, r=10, t=10, b=10),
    coloraxis_showscale=False,
)
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# ---------------- reach-stage heatmap ----------------

st.subheader("Probability of reaching each stage")
heat = stage_df.pivot(index="team", columns="stage", values="prob")
heat = heat[[c for c in STAGES_ORDER if c in heat.columns]]
heat = heat.loc[heat["Champion"].sort_values(ascending=False).index]
# Re-label index with flag emoji prefixes.
heat.index = [with_flag(t) for t in heat.index]
fig2 = px.imshow(
    heat,
    color_continuous_scale="Blues",
    aspect="auto",
    labels=dict(color="probability"),
)
fig2.update_traces(
    text=heat.map(lambda v: f"{v*100:.0f}%").values,
    texttemplate="%{text}",
)
fig2.update_layout(height=900, margin=dict(l=10, r=10, t=10, b=10),
                   yaxis_title="", xaxis_title="")
st.plotly_chart(fig2, use_container_width=True)

# ---------------- group stage ----------------

st.subheader("Group stage — predicted advancement probabilities")
groups_sorted = sorted(group_df["group"].unique())
tabs = st.tabs([f"Group {g}" for g in groups_sorted])

for tab, g in zip(tabs, groups_sorted):
    with tab:
        gtable = (group_df[group_df["group"] == g]
                  .sort_values("p_advance", ascending=False)
                  .reset_index(drop=True))
        gtable["Flag"] = gtable["team"].apply(lambda t: flag_url(t, "60x45") or "")
        gtable["P(finish 1st)"] = (gtable["p_finish_1st"] * 100).round(1).astype(str) + "%"
        gtable["P(advance)"] = (gtable["p_advance"] * 100).round(1).astype(str) + "%"
        gtable.index = gtable.index + 1
        gtable = gtable.rename(columns={"team": "Team"})

        st.dataframe(
            gtable[["Flag", "Team", "P(finish 1st)", "P(advance)"]],
            column_config={
                "Flag": st.column_config.ImageColumn("Flag", width="small"),
                "Team": st.column_config.TextColumn("Team", width="medium"),
            },
            use_container_width=True,
        )

        st.markdown("**Predicted match outcomes:**")
        mtable = match_df[match_df["group"] == g].copy()
        mtable["Match"] = mtable.apply(
            lambda r: f"{flag_emoji(r['team_a'])} {r['team_a']}"
                      f"  vs  {flag_emoji(r['team_b'])} {r['team_b']}",
            axis=1,
        )
        mtable["P(A win)"] = (mtable["p_a_win"] * 100).round(1).astype(str) + "%"
        mtable["P(draw)"] = (mtable["p_draw"] * 100).round(1).astype(str) + "%"
        mtable["P(B win)"] = (mtable["p_b_win"] * 100).round(1).astype(str) + "%"
        mtable = mtable.rename(columns={"predicted_winner": "Predicted"})
        mtable["Predicted"] = mtable["Predicted"].apply(
            lambda t: f"{flag_emoji(t)} {t}" if t != "Draw" else "Draw"
        )
        st.dataframe(
            mtable[["Match", "P(A win)", "P(draw)", "P(B win)", "Predicted"]],
            use_container_width=True,
            hide_index=True,
        )

# ---------------- knockout pairings ----------------

if not ko_pairs.empty:
    st.subheader("Most likely knockout pairings")
    stage_sel = st.selectbox("Stage", STAGES_ORDER[:-1])
    sub = ko_pairs[ko_pairs["stage"] == stage_sel].copy()
    sub["Match"] = sub.apply(
        lambda r: f"{flag_emoji(r['team_a'])} {r['team_a']}"
                  f"  vs  {flag_emoji(r['team_b'])} {r['team_b']}",
        axis=1,
    )
    sub["P(this pairing)"] = (sub["meeting_prob"] * 100).round(1).astype(str) + "%"
    sub["Team A win %"] = (sub["team_a_win_rate"] * 100).round(1).astype(str) + "%"
    sub["Team B win %"] = (sub["team_b_win_rate"] * 100).round(1).astype(str) + "%"
    sub = sub.sort_values("meetings", ascending=False).head(20)
    st.dataframe(
        sub[["Match", "P(this pairing)", "Team A win %", "Team B win %"]],
        use_container_width=True,
        hide_index=True,
    )

# ---------------- Golden Boot ----------------

if not scorer_df.empty:
    st.markdown("---")
    st.subheader("⚽ Golden Boot — top scorer race")

    top_scorer = summary.get("top_scorer", {})
    dist = summary.get("top_scorer_goal_distribution", {})

    if top_scorer:
        flag_html_gb = ""
        if flag_url(top_scorer["team"], "200x150"):
            flag_html_gb = (f"<img class='champion-flag' "
                            f"src='{flag_url(top_scorer['team'], '200x150')}' "
                            f"style='height:64px;'/>")
        st.markdown(
            f"""
            <div class='champion-card' style='background: linear-gradient(135deg, #1a3d2e 0%, #c9a227 100%);'>
                <div style='font-size:14px; letter-spacing:2px; opacity:0.85;'>FAVOURITE FOR THE GOLDEN BOOT</div>
                {flag_html_gb}
                <div class='champion-name' style='font-size:34px;'>{top_scorer['player']}</div>
                <div class='champion-prob'>{top_scorer['team']} &nbsp;|&nbsp;
                    {top_scorer['golden_boot_prob']*100:.1f}% chance &nbsp;|&nbsp;
                    {top_scorer['expected_goals']:.2f} expected goals
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if dist:
        cgb1, cgb2, cgb3 = st.columns(3)
        cgb1.metric("Top scorer goal count (median)", f"{dist['p50']:.0f}")
        cgb2.metric("Average across sims", f"{dist['mean']:.1f}")
        cgb3.metric("10th – 90th percentile",
                    f"{dist['p10']:.0f} – {dist['p90']:.0f} goals")

    top_n = st.slider("Show top N contenders", min_value=10, max_value=40,
                      value=20, step=5, key="gb_slider")

    sc = scorer_df.head(top_n).copy()
    sc["Flag"] = sc["team"].apply(lambda t: flag_url(t, "60x45") or "")
    sc["Player"] = sc["player"]
    sc["Team"] = sc["team"]
    sc["P(Golden Boot)"] = (sc["golden_boot_prob"] * 100).round(2).astype(str) + "%"
    sc["Expected goals"] = sc["expected_goals"].round(2)
    sc.index = sc.index + 1

    st.dataframe(
        sc[["Flag", "Player", "Team", "P(Golden Boot)", "Expected goals"]],
        column_config={
            "Flag": st.column_config.ImageColumn("Flag", width="small"),
            "Player": st.column_config.TextColumn("Player", width="large"),
            "Team": st.column_config.TextColumn("Team", width="medium"),
            "Expected goals": st.column_config.NumberColumn(
                "xG (tournament total)", format="%.2f"
            ),
        },
        use_container_width=True,
        height=min(600, 40 + 36 * top_n),
    )

    # Horizontal bar chart of expected goals
    chart_df = scorer_df.head(top_n).copy()
    chart_df["label"] = chart_df.apply(
        lambda r: f"{flag_emoji(r['team'])} {r['player']}", axis=1
    )
    fig_gb = px.bar(
        chart_df, x="expected_goals", y="label", orientation="h",
        text=chart_df["expected_goals"].apply(lambda v: f"{v:.2f}"),
        color="golden_boot_prob", color_continuous_scale="YlOrRd",
        labels={"expected_goals": "Expected goals", "label": "",
                "golden_boot_prob": "P(Golden Boot)"},
    )
    fig_gb.update_layout(
        yaxis=dict(autorange="reversed"),
        height=max(400, 22 * top_n + 80),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_gb.update_traces(textposition="outside")
    st.plotly_chart(fig_gb, use_container_width=True)

# ---------------- footer ----------------

st.markdown("---")
st.caption(
    f"Model: XGBoost 3-class | Trained on matches 2006 - present | "
    f"{n_sims:,} Monte Carlo simulations | Hosts (USA/CAN/MEX) get home advantage. "
    f"Flags from flagcdn.com. "
    f"To re-run with mid-tournament results: "
    f"`python -m src.update --results data/raw/results_so_far.csv`."
)
