"""
Data representativeness audit - the proper answer to the milestone feedback:
"the chart alone does not prove that the dataset is fully representative".

Instead of inferring representativeness from the shape of a distribution, this script
checks it directly:
    1. Coverage: how many picks does the dataset contain per draft year, and does it
       match the official draft size (~58-60 picks per year, 30 in round 1)?
    2. Season-data match rate: what share of drafted players were matched to
       season-level stats, and who are the unmatched ones?
    3. Missingness: which fields have missing values, and how many?

Outputs:
    figures/audit_coverage.png   - picks per draft year vs expected size
    printed report               - paste-ready numbers for the writeup's Data section

Run from the project root:  python src/audit.py   (after data_prep.py)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
FIGURES = BASE_DIR / "figures"

FIRST_ROUND_SIZE = 30  # picks 1-30 every year in 2000-2020


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Picks per draft year, total and first round."""
    per_year = df.groupby("draft_year").agg(
        total_picks=("pick", "count"),
        first_round=("pick", lambda p: (p <= FIRST_ROUND_SIZE).sum()),
        max_pick=("pick", "max"),
    )
    return per_year


def plot_coverage(per_year: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(per_year.index, per_year["total_picks"], color="#1e3a8a", label="Picks in dataset")
    ax.axhline(FIRST_ROUND_SIZE, color="red", linestyle="--", linewidth=1.5,
               label=f"First round size ({FIRST_ROUND_SIZE})")
    ax.set_title("Dataset coverage: draft picks per year (2000-2020)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Draft year")
    ax.set_ylabel("Number of picks in dataset")
    ax.set_xticks(list(per_year.index))
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    out = FIGURES / "audit_coverage.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


def main() -> None:
    core_path = PROCESSED / "draft_value.csv"
    if not core_path.exists():
        raise SystemExit(f"Missing {core_path} - run:  python src/data_prep.py")
    df = pd.read_csv(core_path)

    print("=" * 60)
    print("1) COVERAGE PER DRAFT YEAR")
    print("=" * 60)
    per_year = coverage_report(df)
    print(per_year.to_string())
    incomplete = per_year[per_year["first_round"] < FIRST_ROUND_SIZE]
    if incomplete.empty:
        print(f"\nFirst round complete ({FIRST_ROUND_SIZE} picks) in every year. ✓")
    else:
        print("\nWARNING - years with an incomplete first round:")
        print(incomplete.to_string())
    plot_coverage(per_year)

    print()
    print("=" * 60)
    print("2) SEASON-DATA MATCH RATE")
    print("=" * 60)
    matched = df["matched"].sum()
    print(f"Matched: {matched}/{len(df)} ({df['matched'].mean():.1%})")
    unmatched = df[~df["matched"]]
    if not unmatched.empty:
        by_round = unmatched["pick"].le(FIRST_ROUND_SIZE).map(
            {True: "first round", False: "second round"}).value_counts()
        print(f"Unmatched by round:\n{by_round.to_string()}")
        print("\nUnmatched FIRST-ROUND picks (these deserve a manual look - a "
              "first-rounder who never played is unusual):")
        cols = [c for c in ["draft_year", "pick", "player"] if c in unmatched.columns]
        print(unmatched[unmatched["pick"] <= FIRST_ROUND_SIZE][cols].to_string(index=False))

    print()
    print("=" * 60)
    print("3) MISSING VALUES PER COLUMN")
    print("=" * 60)
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print(missing.to_string() if not missing.empty else "No missing values. ✓")

    print("\nDone. Use these numbers in the writeup's Data section "
          "(data issues / biases).")


if __name__ == "__main__":
    main()
