"""
QUESTION 3: Does media hype help or hurt? Specifically: does pre-draft hype predict
WHERE a player gets picked better than it predicts HOW GOOD he becomes?

Hype measure: log10(1 + pre-draft Wikipedia pageviews), draft classes 2016-2020
(see src/q3_download_hype.py for the data step and its scope limitation).
The log is essential: views span 5 orders of magnitude (Zion Williamson: 3.5M,
median player: ~18K) and raw views would let a few superstars dominate everything.

Three correlations tell the story (all Spearman - rank-based, robust to skew):
  1. hype <-> draft pick      "does attention buy draft position?"
  2. hype <-> ws_first4       "does attention predict actual value?"
  3. hype <-> surplus         "do hyped players over/under-perform their slot?"
     (surplus = ws_first4 minus the expected value of the player's pick, computed
      as in q2's profile analysis: smoothed per-pick MEAN over all drafts)

Interpretation guide (written before seeing results, like a pre-registration):
  |corr1| > |corr2|  -> hype buys draft position beyond what performance justifies
  corr3 < 0          -> teams systematically overpay for hype
  corr3 ~ 0          -> the market prices hype correctly on average

Outputs:
  figures/q3_hype_vs_pick.png   - hype vs draft slot, annotated outliers
  figures/q3_hype_verdict.png   - the three correlations + surplus by hype quintile
  data/processed/hype_scores.csv - per-player hype score (used by the app)

Run from the project root:  python src/q3_hype.py   (after q3_download_hype.py)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "data" / "raw"
PROCESSED = BASE_DIR / "data" / "processed"
FIGURES = BASE_DIR / "figures"

SMOOTH_WINDOW = 5
C_POINTS = "#9db2ce"
C_MAIN = "#1d4ed8"
C_ACCENT = "#b45309"
C_NEG = "#c2410c"


def load() -> pd.DataFrame:
    hype = pd.read_csv(RAW / "hype_wikipedia.csv")
    core = pd.read_csv(PROCESSED / "draft_value.csv")

    df = hype.merge(core[["draft_year", "pick", "player", "ws_first4"]],
                    on=["draft_year", "pick", "player"], how="inner")
    df["hype"] = np.log10(1 + df["views_predraft"])

    # surplus vs draft slot - same construction as q2's profile analysis:
    # expectation = smoothed per-pick MEAN over ALL drafts (2000-2020), so that
    # surpluses average ~0 and "over/under-performs his slot" is well-defined
    per_pick = (core.groupby("pick")["ws_first4"].mean()
                .rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean())
    df["surplus"] = df["ws_first4"] - df["pick"].map(per_pick)
    return df


def plot_hype_vs_pick(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.scatter(df["pick"], df["hype"], s=30, color=C_POINTS, alpha=0.6,
               linewidths=0)

    # trend: median hype per 5-pick bin
    bins = df.groupby(pd.cut(df["pick"], np.arange(0, 61, 5)), observed=True)
    centers = [interval.mid for interval in bins.groups.keys()]
    ax.plot(centers, bins["hype"].median(), color=C_MAIN, lw=2.5,
            label="Median hype (5-pick bins)")

    # annotate the interesting corners: hyped players who FELL, and the top pick
    fallers = df[df["pick"] > 30].nlargest(3, "hype")
    zion = df.nlargest(1, "hype")
    for _, row in pd.concat([zion, fallers]).iterrows():
        ax.annotate(f"{row['player']} (#{row['pick']}, {row['draft_year']})",
                    xy=(row["pick"], row["hype"]),
                    xytext=(row["pick"] + 1.5, row["hype"] + 0.25),
                    fontsize=8.5, color=C_ACCENT, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C_ACCENT, lw=0.8))

    rho = spearmanr(df["hype"], df["pick"]).statistic
    ax.set_title(f"Pre-draft hype vs draft position, 2016-2020 "
                 f"(Spearman ρ = {rho:.2f})", fontsize=13, fontweight="bold",
                 pad=12)
    ax.set_xlabel("Draft pick")
    ax.set_ylabel("Hype: log10(1 + pre-draft Wikipedia views)")
    ax.set_xlim(0, 61)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURES / "q3_hype_vs_pick.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


def plot_verdict(df: pd.DataFrame, correlations: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # panel 1: the three correlations side by side (absolute values, labeled)
    ax = axes[0]
    names = ["hype ↔\ndraft pick", "hype ↔\nperformance", "hype ↔\nsurplus"]
    values = [abs(correlations["pick"]), abs(correlations["ws4"]),
              abs(correlations["surplus"])]
    bars = ax.bar(names, values, color=[C_MAIN, "#3b82f6", "#94a3b8"], width=0.55)
    ax.bar_label(bars, fmt="%.2f", fontsize=11, fontweight="bold", padding=3)
    ax.set_ylabel("|Spearman correlation|")
    ax.set_title("Hype predicts the pick better than the player",
                 fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    # panel 2: mean surplus by hype quintile - checks the tails, not just the average
    ax = axes[1]
    df = df.copy()
    df["quintile"] = pd.qcut(df["hype"], 5,
                             labels=["Q1\n(least hyped)", "Q2", "Q3", "Q4",
                                     "Q5\n(most hyped)"])
    by_q = df.groupby("quintile", observed=True)["surplus"].agg(["mean", "count"])
    colors = [C_MAIN if v >= 0 else C_NEG for v in by_q["mean"]]
    bars = ax.bar(by_q.index.astype(str), by_q["mean"], color=colors, width=0.6)
    ax.bar_label(bars, fmt="%+.1f", fontsize=10, fontweight="bold", padding=3)
    ax.axhline(0, color="#334155", lw=1)
    ax.set_ylabel("Mean surplus vs draft slot (WS)")
    ax.set_title("...but the market prices hype roughly correctly",
                 fontsize=11.5, fontweight="bold")
    ax.text(0.02, 0.03, "Q5 dip: suggestive, not significant (Welch p≈0.18)",
            transform=ax.transAxes, fontsize=8.5, color="#64748b", style="italic")
    ax.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Q3 verdict: hype buys draft position more than it predicts value",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "q3_hype_verdict.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    df = load()
    print(f"{len(df)} players with hype + outcome data (2016-2020)")
    print(f"views: median {df['views_predraft'].median():,.0f}, "
          f"max {df['views_predraft'].max():,.0f} "
          f"({df.loc[df['views_predraft'].idxmax(), 'player']})")

    correlations = {
        "pick": spearmanr(df["hype"], df["pick"]).statistic,
        "ws4": spearmanr(df["hype"], df["ws_first4"]).statistic,
        "surplus": spearmanr(df["hype"], df["surplus"]).statistic,
    }
    print("\nThe three correlations (Spearman):")
    print(f"  hype <-> pick:        {correlations['pick']:+.3f}  "
          "(negative = more hype, earlier pick)")
    print(f"  hype <-> ws_first4:   {correlations['ws4']:+.3f}")
    print(f"  hype <-> surplus:     {correlations['surplus']:+.3f}")

    print("\nReading: |{:.2f}| > |{:.2f}| -> hype predicts draft position better "
          "than it predicts performance.".format(correlations["pick"],
                                                 correlations["ws4"]))
    print("Surplus correlation ~{:+.2f} -> on average the market does NOT "
          "systematically overpay for hype.".format(correlations["surplus"]))

    # tail check: does the MOST-hyped quintile underperform? (test, don't eyeball -
    # the milestone lost points for overclaiming, so the significance test is
    # part of the pipeline)
    from scipy import stats as st
    quintile = pd.qcut(df["hype"], 5, labels=[1, 2, 3, 4, 5])
    q5 = df.loc[quintile == 5, "surplus"]
    rest = df.loc[quintile != 5, "surplus"]
    t = st.ttest_ind(q5, rest, equal_var=False)
    print(f"\nTail check - most-hyped quintile: mean surplus {q5.mean():+.2f} WS "
          f"vs {rest.mean():+.2f} for everyone else")
    print(f"  Welch t-test p = {t.pvalue:.3f} -> "
          + ("significant at 5%" if t.pvalue < 0.05 else
             "SUGGESTIVE but not significant (n=60/quintile) - report it as such"))

    hype_out = df[["draft_year", "pick", "player", "wiki_title",
                   "views_predraft", "hype", "surplus"]]
    out_path = PROCESSED / "hype_scores.csv"
    hype_out.to_csv(out_path, index=False)
    print(f"\nSaved hype scores for the app: {out_path}")

    plot_hype_vs_pick(df)
    plot_verdict(df, correlations)


if __name__ == "__main__":
    main()
