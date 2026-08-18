"""
QUESTION 1: What is each draft pick actually worth?

Uses the fair metric ws_first4 (Win Shares in the player's first 4 seasons - see
src/data_prep.py) to build the pick-value curve for all picks 1-60, drafts 2000-2020.

This replaces the milestone's Figure 2, which the course feedback criticized on two
grounds, both addressed here:
  * career WS is biased by career length     -> fixed by using ws_first4
  * the per-pick median line zig-zags        -> each pick has only ~21 players, so the
    raw median is noisy; we show the raw medians as points and overlay a rolling
    median (window of 5 picks) as the trend, plus the interquartile range as a band.

Outcome tiers (thresholds are round numbers; the script prints the percentile each
one corresponds to, which is the justification used in the writeup):
  bust        ws_first4 <  1   (essentially zero contribution in 4 seasons)
  contributor ws_first4 >= 10  (~2.5 wins added per season - rotation player or better)
  star        ws_first4 >= 20  (~5 wins added per season - top ~8% of all picks)

Outputs:
  figures/q1_value_curve.png     - the pick-value curve (scatter + trend + IQR band)
  figures/q1_outcome_rates.png   - bust/contributor/star rates by pick range
  data/processed/pick_value.csv  - expected value per pick (reused by q2 as the
                                   draft-order baseline, and by the app)

Run from the project root:  python src/q1_value_curve.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
FIGURES = BASE_DIR / "figures"

BUST_MAX = 1.0
CONTRIBUTOR_MIN = 10.0
STAR_MIN = 20.0
SMOOTH_WINDOW = 5  # picks; the trend line is a rolling median over this window

# colors: one blue for the data itself, amber only as the annotation accent
C_POINTS = "#9db2ce"   # individual players (recessive)
C_MEDIAN = "#1d4ed8"   # per-pick median + trend line
C_BAND = "#1d4ed8"     # IQR band (drawn at low alpha)
C_ACCENT = "#b45309"   # annotated steals
C_TIERS = {"bust": "#c2410c", "role player": "#9db2ce", "contributor": "#3b82f6",
           "star": "#1e3a8a"}


def load() -> pd.DataFrame:
    path = PROCESSED / "draft_value.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path} - run:  python src/data_prep.py")
    return pd.read_csv(path)


def pick_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pick summary + smoothed expected value. Saved for reuse by q2 and the app."""
    per_pick = df.groupby("pick")["ws_first4"].agg(
        median="median", q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75), n="count",
    ).reset_index()
    roll = lambda s: s.rolling(SMOOTH_WINDOW, center=True, min_periods=1).median()
    per_pick["expected_ws4"] = roll(per_pick["median"])
    per_pick["band_low"] = roll(per_pick["q25"])
    per_pick["band_high"] = roll(per_pick["q75"])
    return per_pick


def plot_value_curve(df: pd.DataFrame, per_pick: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(df["pick"], df["ws_first4"], s=22, color=C_POINTS, alpha=0.45,
               linewidths=0, label="Individual player (2000-2020)")
    ax.scatter(per_pick["pick"], per_pick["median"], s=26, color=C_MEDIAN,
               alpha=0.9, linewidths=0, zorder=4, label="Per-pick median")
    ax.plot(per_pick["pick"], per_pick["expected_ws4"], color=C_MEDIAN, lw=2.5,
            zorder=5, label=f"Trend (rolling median, {SMOOTH_WINDOW} picks)")
    ax.fill_between(per_pick["pick"], per_pick["band_low"], per_pick["band_high"],
                    color=C_BAND, alpha=0.12, linewidth=0,
                    label="Middle 50% of outcomes (IQR)")

    # annotate the best value picks found outside the lottery - the "hidden gems".
    # labels are staggered vertically (one slot per label) so they never collide.
    steals = df[df["pick"] > 20].nlargest(5, "ws_first4").sort_values("pick")
    dy_slots = [4.0, 8.5, 13.0, 6.0, 10.5]
    for slot, (_, row) in zip(dy_slots, steals.iterrows()):
        left_side = row["pick"] > 48  # near the right edge -> label to the left
        ax.annotate(f"{row['player']} ({row['draft_year']}, #{row['pick']})",
                    xy=(row["pick"], row["ws_first4"]),
                    xytext=(row["pick"] + (-1.5 if left_side else 1.5),
                            row["ws_first4"] + slot),
                    ha="right" if left_side else "left",
                    fontsize=8.5, color=C_ACCENT, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C_ACCENT, lw=0.8))

    ax.set_title("What is a draft pick worth? Win Shares in first 4 seasons, "
                 "by pick (2000-2020)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Draft pick")
    ax.set_ylabel("Win Shares, first 4 seasons")
    ax.set_xlim(0, 61)
    ax.set_xticks([1] + list(range(5, 61, 5)))
    ax.axvline(30.5, color="#94a3b8", lw=1, ls=":")
    ax.text(30.8, ax.get_ylim()[1] * 0.97, "2nd round", fontsize=8.5,
            color="#64748b", va="top")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURES / "q1_value_curve.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


def outcome_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Share of busts / role players / contributors / stars per pick range."""
    ranges = [(1, 5), (6, 10), (11, 14), (15, 30), (31, 60)]
    labels = ["1-5", "6-10", "11-14\n(late lottery)", "15-30", "31-60\n(2nd round)"]
    rows = []
    for (lo, hi), label in zip(ranges, labels):
        g = df[(df["pick"] >= lo) & (df["pick"] <= hi)]["ws_first4"]
        rows.append({
            "range": label, "n": len(g),
            "bust": (g < BUST_MAX).mean(),
            "role player": ((g >= BUST_MAX) & (g < CONTRIBUTOR_MIN)).mean(),
            "contributor": ((g >= CONTRIBUTOR_MIN) & (g < STAR_MIN)).mean(),
            "star": (g >= STAR_MIN).mean(),
        })
    return pd.DataFrame(rows)


def plot_outcome_rates(rates: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = pd.Series(0.0, index=rates.index)
    for tier in ["bust", "role player", "contributor", "star"]:
        ax.bar(rates["range"], rates[tier], bottom=bottom, width=0.62,
               color=C_TIERS[tier], label=tier, edgecolor="white", linewidth=2)
        # label segments that are large enough to hold text
        for i, v in rates[tier].items():
            if v >= 0.07:
                ax.text(i, bottom[i] + v / 2, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += rates[tier]

    ax.set_title("Draft outcomes by pick range: even top-5 picks bust, "
                 "and stars appear in round 2", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Share of players")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    ax.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURES / "q1_outcome_rates.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    df = load()

    # justify the tier thresholds by their empirical percentiles (for the writeup)
    print("Outcome-tier thresholds and their empirical shares (all picks):")
    print(f"  bust        (< {BUST_MAX:.0f} WS):  "
          f"{(df['ws_first4'] < BUST_MAX).mean():.1%} of picks")
    print(f"  contributor (>= {CONTRIBUTOR_MIN:.0f} WS): "
          f"{(df['ws_first4'] >= CONTRIBUTOR_MIN).mean():.1%} of picks")
    print(f"  star        (>= {STAR_MIN:.0f} WS): "
          f"{(df['ws_first4'] >= STAR_MIN).mean():.1%} of picks")

    per_pick = pick_value_table(df)
    out = PROCESSED / "pick_value.csv"
    per_pick.to_csv(out, index=False)
    print(f"\nSaved expected value per pick: {out}")

    print("\nExpected value (smoothed median ws_first4) at key picks:")
    for p in [1, 3, 5, 10, 15, 20, 30, 40, 50, 60]:
        row = per_pick[per_pick["pick"] == p]
        if not row.empty:
            print(f"  pick {p:>2}: {row['expected_ws4'].iloc[0]:5.1f} WS")

    plot_value_curve(df, per_pick)
    rates = outcome_rates(df)
    print("\nOutcome rates by pick range:")
    print(rates.set_index("range").to_string(
        float_format=lambda v: f"{v:.1%}" if v <= 1 else str(v)))
    plot_outcome_rates(rates)


if __name__ == "__main__":
    main()
