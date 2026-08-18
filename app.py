"""
A Needle in a Data Haystack — NBA Draft Value Explorer
=======================================================
Course: A Needle in a Data Haystack (67978)
Team: Ido Bargal, Asaf Vitenshtein
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================================
# PAGE CONFIG (must be the first Streamlit call)
# =========================================================================
st.set_page_config(
    page_title="A Needle in a Data Haystack — NBA Draft Value",
    page_icon="🪡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# DESIGN TOKENS
# =========================================================================
INK = "#101826"
PAPER = "#F6F7F9"
GOLD = "#D8A73D"
GOLD_BRIGHT = "#E8B923"
TEAL = "#2F6F72"
BRICK = "#B5432F"
MUTED = "#6B7280"

TIER_COLORS = {
    "Bust": "#C9C4B6",
    "Role Player": "#B79A66",
    "Contributor": "#C98F3A",
    "Star": GOLD_BRIGHT,
}


def inject_css():
    # CRITICAL FIX: No blank empty lines and no indentation inside the <style> block.
    # This prevents Streamlit's Markdown parser from breaking the CSS into plain text.
    st.markdown("""<style>
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; color: #101826; }
h1 { font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { font-weight: 600 !important; border-bottom: 2px solid #D8A73D; padding-bottom: 6px; margin-top: 1.4rem !important; }
h3 { font-weight: 600 !important; }
[data-testid="stMetricValue"] { font-family: 'Fraunces', Georgia, serif; color: #101826; }
[data-testid="stMetricLabel"] { color: #6B7280; }
.badge { display: inline-block; background: #FDF3DC; color: #7A5B12; border: 1px solid #E8B923; padding: 3px 11px; border-radius: 999px; font-size: 0.80rem; margin-bottom: 0.7rem; }
.badge-live { background: #E7F1EF; color: #1F4D45; border: 1px solid #2F6F72; }
.pill { display: inline-block; background: rgba(232,185,35,0.14); border: 1px solid #D8A73D; color: #101826; padding: 3px 12px; border-radius: 999px; font-size: 0.78rem; margin: 2px 6px 2px 0; }
.question-eyebrow { color: #E8B923; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; font-size: 0.78rem; margin-bottom: -0.6rem; }
</style>""", unsafe_allow_html=True)


def sample_badge(text="Illustrative sample data — wire up DataManager for live results"):
    st.markdown(f'<div class="badge">🔧 {text}</div>', unsafe_allow_html=True)


def confirmed_badge(text="Figure confirmed in the written report"):
    st.markdown(f'<div class="badge badge-live">✓ {text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=420, show_legend=True):
    fig.update_layout(
        height=height,
        font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="IBM Plex Sans, sans-serif"),
        title=dict(font=dict(family="Fraunces, Georgia, serif", size=17, color=INK)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(16,24,38,0.25)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(16,24,38,0.08)", zeroline=False)
    return fig


# =========================================================================
# DATA MANAGER
# =========================================================================
class DataManager:
    @staticmethod
    @st.cache_data
    def get_value_curve():
        picks = np.arange(1, 61)
        anchor_picks = np.array([1, 5, 10, 15, 20, 25, 30, 40, 50, 60])
        anchor_values = np.array([18.0, 13.0, 8.0, 7.0, 5.5, 4.2, 3.1, 2.2, 1.6, 1.2])
        expected_ws = np.interp(picks, anchor_picks, anchor_values)
        band_low = np.clip(expected_ws * 0.15, 0, None)
        band_high = expected_ws * 2.3 + 2.0
        return pd.DataFrame(
            {"Pick": picks, "Expected_WS": expected_ws, "Band_Low": band_low, "Band_High": band_high}
        )

    @staticmethod
    def get_needle_players():
        curve = DataManager.get_value_curve().set_index("Pick")["Expected_WS"]
        rows = [
            {"name": "Rudy Gobert", "pick": 27, "note": "3x Defensive Player of the Year", "lift": 2.7},
            {"name": "Isaiah Thomas", "pick": 60, "note": "All-NBA guard, the last pick in the draft", "lift": 3.4},
            {"name": "Carl Landry", "pick": 31, "note": "A decade of efficient rotation minutes", "lift": 2.2},
        ]
        for r in rows:
            r["sample_ws"] = round(float(curve.loc[r["pick"]]) * r["lift"], 1)
        return pd.DataFrame(rows)

    @staticmethod
    @st.cache_data
    def get_outcome_rates():
        data = {
            "Range": ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-60 (Rd 2)"],
            "Bust": [6, 10, 15, 20, 23, 25, 64],
            "Role Player": [30, 35, 38, 42, 42, 42, 27],
            "Contributor": [34, 33, 31, 29, 27, 26, 7.1],
            "Star": [30, 22, 16, 9, 8, 7, 1.9],
        }
        return pd.DataFrame(data)

    @staticmethod
    @st.cache_data
    def get_model_comparison():
        return pd.DataFrame(
            {"Model": ["Draft Pick Alone", "Pick + Physical Measurables"], "R_squared": [0.11, 0.26]}
        )

    @staticmethod
    @st.cache_data
    def get_bias_breakdown():
        return pd.DataFrame(
            [
                {"Group": "Teenagers (19 & under)", "WS_Delta": -1.4, "Confirmed": True},
                {"Group": "22+ prospects", "WS_Delta": 0.5, "Confirmed": False},
                {"Group": "Shooting Guard", "WS_Delta": -0.9, "Confirmed": False},
                {"Group": "Point Guard", "WS_Delta": 0.1, "Confirmed": False},
                {"Group": "Small Forward", "WS_Delta": -0.2, "Confirmed": False},
                {"Group": "Power Forward", "WS_Delta": 0.3, "Confirmed": False},
                {"Group": "Center", "WS_Delta": 1.1, "Confirmed": False},
            ]
        )

    @staticmethod
    @st.cache_data
    def get_prospect_board():
        rng = np.random.default_rng(7)
        n = 40
        positions = rng.choice(["PG", "SG", "SF", "PF", "C"], size=n, p=[0.22, 0.22, 0.20, 0.18, 0.18])
        ages = rng.integers(18, 23, size=n)
        projected_pick = np.sort(rng.choice(np.arange(1, 61), size=n, replace=False))
        combine_available = rng.random(n) > 0.22
        wingspan_diff = np.round(rng.normal(0, 2.0, size=n), 1)

        df = pd.DataFrame(
            {
                "Name": [f"Prospect {i + 1:02d}" for i in range(n)],
                "Position": positions,
                "Age": ages,
                "Projected Pick": projected_pick,
                "Combine Data": np.where(combine_available, "Available", "Imputed (median)"),
                "Wingspan vs Height (in)": np.where(combine_available, wingspan_diff, 0.0),
            }
        )
        tilt = np.zeros(n)
        tilt += np.where(df["Age"] <= 19, -1.4, 0.0)
        tilt += np.where(df["Position"] == "SG", -0.9, 0.0)
        tilt += np.where(df["Position"] == "C", 1.1, 0.0)
        tilt += df["Wingspan vs Height (in)"] * 0.15
        df["Value Tilt (WS)"] = np.round(tilt, 2)
        return df

    @staticmethod
    @st.cache_data
    def get_hype_preview():
        rng = np.random.default_rng(3)
        n = 34
        pick = np.sort(rng.choice(np.arange(1, 61), size=n, replace=False))
        hype = np.clip(100 - pick * 1.3 + rng.normal(0, 12, n), 1, 100)
        outcome_ws = np.clip(18 * np.exp(-pick / 18) + rng.normal(0, 3, n), 0, None)
        return pd.DataFrame(
            {"Projected Pick": pick, "Pre-Draft Hype Score": np.round(hype, 1), "Rookie-Window WS": np.round(outcome_ws, 1)}
        )

    @staticmethod
    @st.cache_data
    def get_draft_class_sample():
        rng = np.random.default_rng(11)
        anchor_picks = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60]
        anchor_values = [18.0, 13.0, 8.0, 7.0, 5.5, 4.2, 3.1, 2.2, 1.6, 1.2]
        sample_picks = [1, 3, 7, 14, 22, 27, 31, 45, 55, 60]
        rows = []
        for p in sample_picks:
            base = float(np.interp(p, anchor_picks, anchor_values))
            noise_mult = rng.uniform(0.4, 2.0)
            seasons_ws = np.clip(base * noise_mult / 4 + rng.normal(0, 0.5, 4), 0, None)
            for szn_idx, ws in enumerate(seasons_ws, start=1):
                rows.append({"Pick": p, "Label": f"Pick #{p}", "Season": f"Year {szn_idx}", "WS": round(float(ws), 2)})
        return pd.DataFrame(rows)


# =========================================================================
# CHART BUILDERS
# =========================================================================
def render_value_curve(df_curve, df_needles=None, compact=False):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([df_curve["Pick"], df_curve["Pick"][::-1]]),
            y=np.concatenate([df_curve["Band_High"], df_curve["Band_Low"][::-1]]),
            fill="toself",
            fillcolor="rgba(216,167,61,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Outcome range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_curve["Pick"],
            y=df_curve["Expected_WS"],
            mode="lines",
            line=dict(color=INK, width=3),
            name="Expected value",
        )
    )
    if df_needles is not None:
        fig.add_trace(
            go.Scatter(
                x=df_needles["pick"],
                y=df_needles["sample_ws"],
                mode="markers+text",
                text=df_needles["name"],
                textposition="top center",
                textfont=dict(size=11, color=INK),
                marker=dict(color=GOLD_BRIGHT, size=13, line=dict(color=INK, width=1.5), symbol="star"),
                name="Needles in the haystack",
            )
        )
    fig.update_layout(title=None if compact else "Draft Value Curve — Expected Rookie-Window Win Shares")
    fig.update_xaxes(title=None if compact else "Draft Pick")
    fig.update_yaxes(title=None if compact else "Win Shares (first 4 seasons)")
    return style_fig(fig, height=280 if compact else 460, show_legend=not compact)


def render_outcome_rates(df):
    fig = go.Figure()
    for cat in ["Bust", "Role Player", "Contributor", "Star"]:
        fig.add_trace(go.Bar(x=df["Range"], y=df[cat], name=cat, marker_color=TIER_COLORS[cat]))
    fig.update_layout(barmode="stack", title="Outcome Odds by Draft Range")
    fig.update_xaxes(title="Draft Pick Range")
    fig.update_yaxes(title="Share of Picks (%)")
    return style_fig(fig, height=440)


def render_bias_chart(df):
    df = df.sort_values("WS_Delta")
    colors = [TEAL if v >= 0 else BRICK for v in df["WS_Delta"]]
    fig = go.Figure(
        go.Bar(
            x=df["WS_Delta"],
            y=df["Group"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f} WS" for v in df["WS_Delta"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_layout(title="Who Gets Mispriced? WS Above/Below Slot Expectation")
    fig.update_xaxes(title="Win Shares vs. expectation for that slot")
    fig.update_yaxes(title=None)
    return style_fig(fig, height=380, show_legend=False)


def render_model_comparison(df):
    fig = go.Figure(
        go.Bar(
            x=df["Model"],
            y=df["R_squared"],
            marker_color=[MUTED, GOLD_BRIGHT],
            text=[f"{v:.2f}" for v in df["R_squared"]],
            textposition="outside",
            width=0.5,
        )
    )
    fig.update_layout(title="Explained Variance in Rookie-Window Value")
    fig.update_yaxes(title="R²", range=[0, 0.34])
    fig.update_xaxes(title=None)
    return style_fig(fig, height=360, show_legend=False)


def render_trajectory(df_player, expected_val):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_player["Season"], y=df_player["WS"], marker_color=INK, name="Actual"))
    fig.add_hline(
        y=expected_val / 4,
        line_dash="dash",
        line_color=BRICK,
        annotation_text="Expected pace for this pick",
        annotation_position="top left",
    )
    fig.update_layout(title="Rookie-Window Trajectory vs. Expectation")
    fig.update_yaxes(title="Win Shares (per season)")
    fig.update_xaxes(title=None)
    return style_fig(fig, height=380, show_legend=False)


# =========================================================================
# PAGES
# =========================================================================
def page_overview(dm: DataManager):
    
    # CRITICAL FIX: No empty lines in this HTML block. 
    # Everything is densely packed so the Markdown parser doesn't break it into code blocks.
    st.markdown("""<div style="position:relative; overflow:hidden; padding:2.5rem 2.2rem; border-radius:18px; margin-bottom:1.3rem; background:linear-gradient(135deg, #101826 0%, #1D2C46 100%); color:#F6F7F9; box-shadow:0 4px 20px rgba(0,0,0,0.1);">
<img src="https://images.unsplash.com/photo-1546519638-68e109498ffc?auto=format&fit=crop&w=1200&q=80" style="position:absolute; top:0; left:0; width:100%; height:100%; object-fit:cover; opacity:0.15; z-index:1;">
<div style="position:relative; z-index:2;">
<div class="question-eyebrow" style="color:#E8B923; margin-bottom:0.4rem;">A Needle in a Data Haystack · 67978</div>
<h1 style="color:#F6F7F9 !important; margin-top:0; margin-bottom:0.4rem;">Where is the hidden value in the NBA draft?</h1>
<p style="color:#C7CEDA; font-size:1.05rem; max-width:62ch; margin-bottom:0;">Rookie contracts are cheap relative to what veterans cost under the salary cap, so a draft pick is the cheapest path to surplus value a team has. This project measures what each pick is actually worth, hunts for the picks the market consistently gets wrong, and asks whether pre-draft hype is doing more to predict draft slot than it does to predict who can actually play.</p>
</div>
</div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="pill">Q1 · What is a pick worth?</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="pill">Q2 · Where does the market misprice?</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="pill">Q3 · Does hype help or hurt?</div>', unsafe_allow_html=True)

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pick-order correlation (Spearman ρ)", "0.50")
    m2.metric("R² — pick alone", "0.11")
    m3.metric("R² — pick + physicals", "0.26")
    m4.metric("Teenager penalty", "-1.4 WS")
    confirmed_badge("The four figures above are confirmed in the written report")

    # --- NBA LEGENDS SECTION ---
    st.divider()
    st.subheader("Championship Pedigree: The Pursuit of Legends")
    st.markdown("Every front office searches the draft board for the next generation of greatness. Whether it’s finding a franchise cornerstone or a missing championship piece, **finding hidden value in the data is how you find the next legends.**")
    
    legend1, legend2, legend3 = st.columns(3)
    
    with legend1:
        with st.container(border=True):
            st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/893.png", use_container_width=True)
            st.markdown("<h4 style='text-align:center; margin-bottom:0;'>Michael Jordan</h4><p style='text-align:center; color:#6B7280; font-size:0.9rem;'>Pick #3 (1984)</p>", unsafe_allow_html=True)
        
    with legend2:
        with st.container(border=True):
            st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/1449.png", use_container_width=True)
            st.markdown("<h4 style='text-align:center; margin-bottom:0;'>Larry Bird</h4><p style='text-align:center; color:#6B7280; font-size:0.9rem;'>Pick #6 (1978)</p>", unsafe_allow_html=True)
        
    with legend3:
        with st.container(border=True):
            st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/77142.png", use_container_width=True)
            st.markdown("<h4 style='text-align:center; margin-bottom:0;'>Magic Johnson</h4><p style='text-align:center; color:#6B7280; font-size:0.9rem;'>Pick #1 (1979)</p>", unsafe_allow_html=True)

    # --- SECOND ROUND GEMS SECTION ---
    st.divider()
    st.subheader("The Ultimate Needles: Second-Round Gems")
    st.markdown("While the top of the draft yields the most superstars, the second round is where true competitive advantage is won. **If you analyze the data correctly, you can uncover hidden greatness and future Hall of Famers late in the draft**—long after the rest of the league has passed on them.")
    
    gem1, gem2, gem3 = st.columns(3)
    
    with gem1:
        with st.container(border=True):
            # 203999 is Nikola Jokic's official NBA player ID
            st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/203999.png", use_container_width=True)
            st.markdown("<h4 style='text-align:center; margin-bottom:0;'>Nikola Jokić</h4><p style='text-align:center; color:#6B7280; font-size:0.9rem;'>Pick #41 (2014)</p>", unsafe_allow_html=True)
            
    with gem2:
        with st.container(border=True):
            # 1938 is Manu Ginobili's official NBA player ID
            st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/1938.png", use_container_width=True)
            st.markdown("<h4 style='text-align:center; margin-bottom:0;'>Manu Ginóbili</h4><p style='text-align:center; color:#6B7280; font-size:0.9rem;'>Pick #57 (1999)</p>", unsafe_allow_html=True)
            
    with gem3:
        with st.container(border=True):
            # 203110 is Draymond Green's official NBA player ID
            st.image("https://cdn.nba.com/headshots/nba/latest/1040x760/203110.png", use_container_width=True)
            st.markdown("<h4 style='text-align:center; margin-bottom:0;'>Draymond Green</h4><p style='text-align:center; color:#6B7280; font-size:0.9rem;'>Pick #35 (2012)</p>", unsafe_allow_html=True)

    st.divider()
    st.subheader("The whole question, in one chart")
    st.caption(
        "Draft position explains most of the predictable signal — the curve falls fast, then flattens. "
        "But at every pick, the outcome band is wide, and a handful of picks land far above the line. "
        "Those are the needles."
    )
    sample_badge()
    curve = dm.get_value_curve()
    needles = dm.get_needle_players()
    st.plotly_chart(render_value_curve(curve, needles), use_container_width=True)

    with st.expander("How to read this app"):
        st.markdown(
            "- **Q1 — What's a Pick Worth?** rebuilds the value curve and the outcome-rate breakdown "
            "(Figures 1 & 2 in the report).\n"
            "- **Q2 — Finding the Hidden Value** shows the systematic biases (age, position) and lets you "
            "filter a prospect board by the same logic.\n"
            "- **Q3 — Hype vs. Outcome** is a scaffold for the media-hype question, ready for real data.\n"
            "- **Draft Class Explorer** compares a sample pick's season-by-season trajectory to expectation.\n"
            "- **Data & Methods** covers sourcing, coverage gaps, and the impediments the team hit."
        )


def page_q1(dm: DataManager):
    st.markdown('<div class="question-eyebrow" style="color:#B5432F;">Question 1</div>', unsafe_allow_html=True)
    st.title("What is each draft pick actually worth?")
    st.markdown(
        "Win Shares over a player's first four seasons stand in for value here — roughly, 20 WS across "
        "four years is about five wins a season, a very good player. Two views of the same underlying data:"
    )

    tab1, tab2 = st.tabs(["📉 The Value Curve", "🎯 Outcome Odds by Range"])

    with tab1:
        sample_badge()
        curve = dm.get_value_curve()
        needles = dm.get_needle_players()
        st.plotly_chart(render_value_curve(curve, needles), use_container_width=True)
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(
                "**Reading the curve.** Value falls fast at the top — pick 1 projects far above pick 10 — "
                "then flattens: picks 15 through 30 barely separate. Draft position matters most at the very "
                "top of the board, and the shaded band never narrows to zero. Even at pick 5, outcomes run "
                "from bust to superstar."
            )
        with col2:
            st.markdown("**The needles**")
            for _, row in needles.iterrows():
                st.markdown(f"- **{row['name']}** — pick #{int(row['pick'])}. {row['note']}.")
            st.caption("Positions on the chart are illustrative until final win-share totals are computed.")

    with tab2:
        sample_badge("Illustrative distribution calibrated to the report's stated anchors")
        rates = dm.get_outcome_rates()
        st.plotly_chart(render_outcome_rates(rates), use_container_width=True)
        st.markdown(
            "A top-5 pick carries roughly a **30% shot at a star** and only about a **6% chance of a total "
            "bust**. By picks 15–30 the star rate has dropped near **7–9%** and roughly **1 in 4 busts**. "
            "In the second round, most picks — roughly **64%** — produce close to nothing, yet stars still "
            "turn up occasionally. The draft order works on average; its risk never disappears."
        )

    with st.expander("A data quirk worth knowing"):
        st.markdown(
            "2001–2002 shows only 29 first-round picks in the historical record instead of 30. That isn't a "
            "data error — Minnesota forfeited a first-round pick as a penalty in the Joe Smith cap-circumvention "
            "case, so the anomaly matches real history."
        )


def page_q2(dm: DataManager):
    st.markdown('<div class="question-eyebrow" style="color:#B5432F;">Question 2</div>', unsafe_allow_html=True)
    st.title("Can pre-draft data beat the pick order — and where does it miss?")

    m1, m2 = st.columns(2)
    with m1:
        confirmed_badge()
        st.plotly_chart(render_model_comparison(dm.get_model_comparison()), use_container_width=True)
        st.caption(
            "The draft pick alone explains about 11% of the variance in rookie-window value. Adding just "
            "physical-measurement data more than doubles that to 26%. The market is broadly efficient — "
            "pick number carries most of the predictable signal — but it isn't perfectly efficient, and its "
            "misses are systematic."
        )
    with m2:
        st.plotly_chart(render_bias_chart(dm.get_bias_breakdown()), use_container_width=True)
        st.caption(
            "Teenagers underperform their slot (confirmed: **-1.4 WS**). Shooting guards read as over-drafted "
            "and centers as under-drafted in the report's findings; the other bars here are illustrative "
            "pending the full regression output."
        )

    st.divider()
    st.subheader("💎 Hidden Gems: filter the prospect board")
    st.caption(
        "Not a black-box prediction — every prospect's 'Value Tilt' is built directly from the bias chart "
        "above, so the reasoning behind every recommendation is visible."
    )
    sample_badge("Mock 40-prospect board — swap in the real board via get_prospect_board()")

    board = dm.get_prospect_board()
    f1, f2, f3 = st.columns(3)
    with f1:
        pos_filter = st.multiselect("Position", ["PG", "SG", "SF", "PF", "C"], default=["PG", "SG", "SF", "PF", "C"])
    with f2:
        pick_max = st.slider("Projected pick at or later than", 1, 60, 15)
    with f3:
        min_tilt = st.slider("Minimum Value Tilt (WS)", -2.0, 2.0, 0.0, step=0.1)

    filtered = board[
        board["Position"].isin(pos_filter)
        & (board["Projected Pick"] >= pick_max)
        & (board["Value Tilt (WS)"] >= min_tilt)
    ].sort_values("Value Tilt (WS)", ascending=False)

    if filtered.empty:
        st.warning("No prospects match these filters — widen the range above.")
    else:
        st.success(f"{len(filtered)} prospect(s) the market has historically undervalued at this range.")
        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Value Tilt (WS)": st.column_config.NumberColumn(format="%.2f"),
                "Wingspan vs Height (in)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

    with st.expander("Coverage note"):
        st.markdown(
            "Not every prospect attends the combine — internationals especially are under-represented. "
            "Missing physical measurements are median-imputed rather than dropped, so the coverage gap "
            "doesn't silently bias the board toward players who happened to show up."
        )


def page_q3(dm: DataManager):
    st.markdown('<div class="question-eyebrow" style="color:#B5432F;">Question 3</div>', unsafe_allow_html=True)
    st.title("Does pre-draft hype help or hurt?")
    st.info(
        "🚧 **Preview.** The media-coverage analysis hasn't run yet, so this page is a working scaffold: "
        "the two charts below are wired up and ready, just waiting on the real hype-volume dataset.",
        icon="🚧",
    )
    sample_badge("Fully synthetic placeholder — no relationship implied")

    hype = dm.get_hype_preview()
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(hype, x="Projected Pick", y="Pre-Draft Hype Score", color_discrete_sequence=[INK])
        fig1.update_layout(title="Hype vs. Draft Position")
        st.plotly_chart(style_fig(fig1, height=380, show_legend=False), use_container_width=True)
    with c2:
        fig2 = px.scatter(hype, x="Pre-Draft Hype Score", y="Rookie-Window WS", color_discrete_sequence=[GOLD_BRIGHT])
        fig2.update_layout(title="Hype vs. On-Court Outcome")
        st.plotly_chart(style_fig(fig2, height=380, show_legend=False), use_container_width=True)

    st.markdown(
        "**What goes here once the crawl finishes:** compare how strongly pre-draft media volume predicts "
        "*draft position* against how strongly it predicts *rookie-window Win Shares*. If hype tracks draft "
        "slot much more tightly than it tracks outcome, that's evidence hype is a market-sentiment signal "
        "more than a talent signal — a second, independent route to the same 'hidden value' question as Q1/Q2."
    )


def page_explorer(dm: DataManager):
    st.title("🔎 Draft Class Explorer")
    st.markdown("Pick a sample draft slot and compare its rookie-window trajectory to the expected pace for that pick.")
    sample_badge("Synthetic sample picks — replace with real per-player, per-season data")

    trajectories = dm.get_draft_class_sample()
    curve = dm.get_value_curve().set_index("Pick")["Expected_WS"]

    labels = trajectories[["Pick", "Label"]].drop_duplicates().sort_values("Pick")
    choice = st.selectbox("Sample pick", labels["Label"])
    chosen_pick = int(labels.loc[labels["Label"] == choice, "Pick"].iloc[0])

    player_df = trajectories[trajectories["Pick"] == chosen_pick]
    total_ws = player_df["WS"].sum()
    expected_total = float(curve.loc[chosen_pick])
    delta = total_ws - expected_total

    c1, c2, c3 = st.columns(3)
    c1.metric("Total rookie-window WS", f"{total_ws:.1f}")
    c2.metric("Expected for this pick", f"{expected_total:.1f}")
    c3.metric("Above / below expectation", f"{delta:+.1f} WS")

    st.plotly_chart(render_trajectory(player_df, expected_total), use_container_width=True)

    if delta > 1.5:
        st.success("This slot outperformed expectation by a wide margin — a needle.")
    elif delta < -1.5:
        st.warning("This slot underperformed expectation by a wide margin.")
    else:
        st.info("This slot landed roughly where the curve predicted.")


def page_methods():
    st.title("📋 Data & Methods")

    st.subheader("Domain")
    st.markdown(
        "The project sits in NBA talent evaluation, centered on the annual draft. Because rookie-scale "
        "contracts are cheap relative to veteran deals under the salary cap, draft picks are a team's "
        "cheapest source of surplus value — which is why drafting well is one of the strongest predictors "
        "of long-run team success."
    )

    st.subheader("Data sources")
    st.markdown(
        "- **Pre-draft:** college statistics, physical measurements, recruiting rankings\n"
        "- **Post-draft:** rookie-window (first 4 seasons) career outcomes, primarily Win Shares\n"
        "- **Context:** media coverage volume surrounding each prospect (for Q3)"
    )
    d1, d2, d3 = st.columns(3)
    d1.metric("Records", "— TBD —")
    d2.metric("Seasons covered", "— TBD —")
    d3.metric("Data size", "— TBD —")
    st.caption("Fill these in once the final dataset is locked — placeholders left intentionally blank.")

    st.subheader("Known data issue")
    st.markdown(
        "2001–2002 shows only 29 first-round picks rather than 30. This matches real history: Minnesota "
        "forfeited a first-round pick as a penalty in the Joe Smith cap-circumvention case, so the anomaly "
        "is a correct reflection of the record, not a bug."
    )
    st.markdown(
        "Combine attendance is also incomplete — international prospects especially are under-represented. "
        "Missing physical measurements are median-imputed rather than dropped."
    )

    st.subheader("Impediments")
    with st.container(border=True):
        st.markdown(
            "**Rate limiting on the initial crawl.** The first pass at pulling player data through the "
            "Wikimedia API got rate-limited partway through. The fix: randomized pacing between requests, "
            "exponential backoff that honors the API's `Retry-After` header, and incremental checkpointing "
            "so an interrupted run resumes instead of restarting from zero."
        )

    st.subheader("Team")
    st.markdown("A Needle in a Data Haystack · Course 67978")
    st.markdown("Ido Bargal · Asaf Vitenshtein · Dor Snapiri · Israel Fahima")


# =========================================================================
# NAVIGATION / MAIN
# =========================================================================
PAGES = {
    "Overview": "🏠",
    "Q1 · What's a Pick Worth?": "📉",
    "Q2 · Finding the Hidden Value": "💎",
    "Q3 · Hype vs. Outcome": "📣",
    "Draft Class Explorer": "🔎",
    "Data & Methods": "📋",
}


def render_sidebar():
    st.sidebar.markdown(
        "<div style='font-family: Fraunces, serif; font-size: 1.4rem; font-weight:700; color:#F6F7F9;'>"
        "🪡 A Needle in a Haystack</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("NBA Draft Value Explorer")
    st.sidebar.divider()

    if "nav" not in st.session_state:
        st.session_state.nav = "Overview"

    selection = st.sidebar.radio(
        "Navigate",
        list(PAGES.keys()),
        format_func=lambda p: f"{PAGES[p]}  {p}",
        key="nav",
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption("Confirmed findings so far")
    st.sidebar.markdown(
        "- Pick–outcome correlation: **ρ = 0.50**\n"
        "- R² lift with physicals: **0.11 → 0.26**\n"
        "- Teenager penalty: **-1.4 WS**"
    )
    st.sidebar.divider()
    st.sidebar.caption("Course 67978 · Final project")
    return selection


def main():
    inject_css()
    dm = DataManager()
    selection = render_sidebar()

    if selection == "Overview":
        page_overview(dm)
    elif selection == "Q1 · What's a Pick Worth?":
        page_q1(dm)
    elif selection == "Q2 · Finding the Hidden Value":
        page_q2(dm)
    elif selection == "Q3 · Hype vs. Outcome":
        page_q3(dm)
    elif selection == "Draft Class Explorer":
        page_explorer(dm)
    elif selection == "Data & Methods":
        page_methods()


if __name__ == "__main__":
    main()
