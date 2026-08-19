"""
QUESTION 2: Can we predict a prospect's NBA value from pre-draft data better than
the draft order itself does - and which profiles get over/under-drafted?

FEATURE SETS - the script is organized so each set maps to a report subsection:
  SIMPLE   (always available, no extra downloads):
             age_at_draft, position, has_college
  ADVANCED (optional; needs data/raw/combine_stats.csv from src/download_combine.py):
             SIMPLE + height, weight, wingspan
The script always runs SIMPLE; it runs ADVANCED too when the combine file exists,
and saves every figure twice (suffix _simple / _advanced) so the report can show
"model v1" vs "model v2 (+measurements)" side by side.

MODELS COMPARED (identical train/test split for all three):
  A. market baseline - prediction = the pick's expected value, i.e. the smoothed
     median ws_first4 of that pick computed on TRAINING years only. This is "just
     trust the draft order", the baseline required by the course instructions.
  B. features only   - random forest on pre-draft features, WITHOUT the pick.
     Measures how much of the outcome is knowable while ignoring the market.
  C. pick + features - random forest on the pick number plus the features.
     Measures whether pre-draft data adds information ON TOP of the market.

EVALUATION - temporal split, as in real forecasting: train on drafts 2000-2014,
test on 2015-2020 (all test players have completed their 4-season window).
Metrics: MAE (average error in WS), Spearman rank correlation, R².

LEAKAGE NOTES (also for the writeup):
  * age/position are missing exactly for players who never played an NBA game -
    a "missing" indicator would leak the outcome, so we impute (median age, modal
    position) WITHOUT indicators.
  * the baseline's pick values are computed from training years only, so the
    comparison is fair to all models.

Outputs:
  figures/q2_model_comparison_<set>.png  - MAE + Spearman for models A/B/C
  figures/q2_profiles_<set>.png          - who gets over/under-drafted (surplus
                                           by age at draft, and by position)
  data/processed/q2_predictions_<set>.csv - test-set predictions (used by the app)

Run from the project root:  python src/q2_model.py   (after q1_value_curve.py)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "data" / "raw"
PROCESSED = BASE_DIR / "data" / "processed"
FIGURES = BASE_DIR / "figures"

TRAIN_YEARS = range(2000, 2015)   # 2000-2014
TEST_YEARS = range(2015, 2021)    # 2015-2020
SMOOTH_WINDOW = 5                 # same smoothing as q1
RF_PARAMS = dict(n_estimators=400, min_samples_leaf=5, random_state=42, n_jobs=-1)
# min_samples_leaf=5 keeps the forest from memorizing single players (~1,000 rows)

C_BARS = ["#94a3b8", "#3b82f6", "#1e3a8a"]  # baseline / features / pick+features
C_POS = "#1d4ed8"
C_NEG = "#c2410c"


# --------------------------------------------------------------------------- data

def load_core() -> pd.DataFrame:
    path = PROCESSED / "draft_value.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path} - run:  python src/data_prep.py")
    df = pd.read_csv(path)
    df["has_college"] = df["College"].notna().astype(int)
    return df


def add_simple_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """SIMPLE set: age_at_draft, position (one-hot), has_college."""
    out = df.copy()
    out["pos"] = out["pos"].str.split("-").str[0]
    # RAW copies (pre-imputation) - used by the PROFILE analysis, where imputing
    # never-played players into the modal position would fabricate a bias
    out["age_raw"] = out["age_at_draft"]
    out["pos_raw"] = out["pos"]
    # impute WITHOUT missing-indicators (see leakage note in the module docstring)
    out["age_at_draft"] = out["age_at_draft"].fillna(out["age_at_draft"].median())
    out["pos"] = out["pos"].fillna(out["pos"].mode().iloc[0])  # 'SF-SG' -> 'SF'
    pos_dummies = pd.get_dummies(out["pos"], prefix="pos")
    out = pd.concat([out, pos_dummies], axis=1)
    features = ["age_at_draft", "has_college"] + list(pos_dummies.columns)
    return out, features


def add_college_features(df: pd.DataFrame, features: list) -> tuple[pd.DataFrame, list] | None:
    """COLLEGE set: +measurements features PLUS final-college-season production
    (per-game pts/trb/ast/mp, shooting percentages) from our BBRef crawl.
    Joined by (draft_year, pick) - an exact key, no name matching needed.
    77% of picks have college stats; the rest (internationals, preps-to-pros)
    are median-imputed, with has_college already in the feature set as the
    legitimate, pre-draft-knowable indicator."""
    path = RAW / "college_stats.csv"
    if not path.exists():
        return None

    college = pd.read_csv(path)
    stat_cols = ["pts", "trb", "ast", "mp", "g", "fg_pct", "fg3_pct", "ft_pct"]
    keep = [c for c in ["draft_year", "pick"] + stat_cols if c in college.columns]
    out = df.merge(college[keep], on=["draft_year", "pick"], how="left")

    present = [c for c in stat_cols if c in out.columns]
    coverage = out[present[0]].notna().mean()
    for col in present:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median())
    print(f"COLLEGE: college stats merged (coverage before imputation: {coverage:.0%})")
    return out, features + present


def add_advanced_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list] | None:
    """ADVANCED set: SIMPLE + combine measurements. Returns None if unavailable."""
    path = RAW / "combine_stats.csv"
    if not path.exists():
        return None

    combine = pd.read_csv(path)
    name_col = next((c for c in ["PLAYER_NAME", "player_name"] if c in combine), None)
    measure_cols = {
        "height": next((c for c in ["HEIGHT_WO_SHOES", "height_wo_shoes"] if c in combine), None),
        "weight": next((c for c in ["WEIGHT", "weight"] if c in combine), None),
        "wingspan": next((c for c in ["WINGSPAN", "wingspan"] if c in combine), None),
    }
    if name_col is None or any(v is None for v in measure_cols.values()):
        print(f"combine_stats.csv found but columns not recognized "
              f"({list(combine.columns)[:8]}...) - skipping ADVANCED set")
        return None

    from data_prep import normalize_name  # same normalization as the core pipeline
    combine["name_key"] = combine[name_col].map(normalize_name)
    keep = combine.groupby("name_key").first().reset_index()  # one row per player

    out = df.copy()
    out["name_key"] = out["player"].map(normalize_name)
    out = out.merge(keep[["name_key"] + list(measure_cols.values())],
                    on="name_key", how="left")
    rename = {v: k for k, v in measure_cols.items()}
    out = out.rename(columns=rename)

    out, simple_feats = add_simple_features(out)
    for col in ["height", "weight", "wingspan"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["has_combine"] = out["wingspan"].notna()  # real measurement vs imputed
    coverage = out["has_combine"].mean()  # BEFORE imputation, or it's always 100%
    for col in ["height", "weight", "wingspan"]:
        out[col] = out[col].fillna(out[col].median())
    print(f"ADVANCED: combine data merged (coverage before imputation: {coverage:.0%})")
    return out, simple_feats + ["height", "weight", "wingspan"]


# ------------------------------------------------------------------------- models

def market_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Model A: expected ws_first4 of each pick, learned from TRAINING years only."""
    per_pick = train.groupby("pick")["ws_first4"].median().sort_index()
    smoothed = per_pick.rolling(SMOOTH_WINDOW, center=True, min_periods=1).median()
    fallback = smoothed.iloc[-5:].mean()  # picks unseen in training (rare)
    return test["pick"].map(smoothed).fillna(fallback).to_numpy()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "Spearman": spearmanr(y_true, y_pred).statistic,
        "R2": r2_score(y_true, y_pred),
    }


def run_feature_set(df: pd.DataFrame, features: list, set_name: str) -> None:
    print(f"\n{'=' * 62}\n{set_name.upper()} FEATURE SET  ({len(features)} features)\n{'=' * 62}")
    print("features:", ", ".join(features))

    train = df[df["draft_year"].isin(TRAIN_YEARS)]
    test = df[df["draft_year"].isin(TEST_YEARS)]
    y_train, y_test = train["ws_first4"], test["ws_first4"]
    print(f"train: {len(train)} picks ({TRAIN_YEARS.start}-{TRAIN_YEARS.stop - 1}), "
          f"test: {len(test)} picks ({TEST_YEARS.start}-{TEST_YEARS.stop - 1})")

    results, predictions = {}, {}

    predictions["A: draft order"] = market_baseline(train, test)

    rf_b = RandomForestRegressor(**RF_PARAMS).fit(train[features], y_train)
    predictions["B: features only"] = rf_b.predict(test[features])

    feats_c = ["pick"] + features
    rf_c = RandomForestRegressor(**RF_PARAMS).fit(train[feats_c], y_train)
    predictions["C: pick + features"] = rf_c.predict(test[feats_c])

    for name, pred in predictions.items():
        results[name] = evaluate(y_test.to_numpy(), pred)

    res = pd.DataFrame(results).T
    print("\ntest-set results:")
    print(res.round(3).to_string())

    fi = pd.Series(rf_c.feature_importances_, index=feats_c).sort_values(ascending=False)
    print("\nfeature importance (model C):")
    print(fi.round(3).to_string())

    out = test[["draft_year", "pick", "player", "ws_first4"]].copy()
    for name, pred in predictions.items():
        out[name.split(": ")[1].replace(" ", "_")] = pred.round(2)
    pred_path = PROCESSED / f"q2_predictions_{set_name}.csv"
    out.to_csv(pred_path, index=False)
    print(f"\nSaved test predictions: {pred_path}")

    plot_comparison(res, set_name)
    plot_profiles(df, train, set_name)


# ------------------------------------------------------------------------ figures

def plot_comparison(res: pd.DataFrame, set_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, better in [(axes[0], "MAE", "lower"), (axes[1], "Spearman", "higher")]:
        vals = res[metric]
        bars = ax.bar(range(len(vals)), vals, color=C_BARS, width=0.6)
        ax.bar_label(bars, fmt="%.2f", fontsize=10, fontweight="bold", padding=2)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([n.replace(": ", ":\n") for n in vals.index], fontsize=9)
        ax.set_title(f"{metric} ({better} is better)", fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#e2e8f0", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Predicting first-4-season Win Shares - test drafts 2015-2020 "
                 f"({set_name} features)", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    out = FIGURES / f"q2_model_comparison_{set_name}.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


def plot_profiles(df: pd.DataFrame, train: pd.DataFrame, set_name: str) -> None:
    """Surplus vs the pick's expected value, by age at draft and by position.
    Positive = that profile outperforms its draft slot (under-drafted).
    NOTE: the expectation here is the smoothed per-pick MEAN (not median) so that
    surpluses average ~0 overall - otherwise the right-skew of outcomes makes every
    group look positive and the comparison unreadable."""
    per_pick = train.groupby("pick")["ws_first4"].mean()
    smoothed = per_pick.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
    data = df.copy()
    data["surplus"] = data["ws_first4"] - data["pick"].map(smoothed).fillna(0)

    # IMPORTANT: group by the RAW (pre-imputation) age/position. Imputation assigns
    # every never-played player (ws_first4=0) the modal position, which fabricates
    # a large negative "bias" for that position. Players whose age/position is
    # unrecorded (never played in the NBA) are excluded from the group means.
    data["age_group"] = pd.cut(data["age_raw"], [0, 19, 20, 21, 22, 99],
                               labels=["≤19", "20", "21", "22", "23+"])
    by_age = data.groupby("age_group", observed=True)["surplus"].agg(["mean", "count"])
    by_pos = data.groupby("pos_raw")["surplus"].agg(["mean", "count"]).reindex(
        ["PG", "SG", "SF", "PF", "C"]).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, table, title, xlabel in [
        (axes[0], by_age, "by age at draft", "age on draft night"),
        (axes[1], by_pos, "by position", "position (first NBA season)"),
    ]:
        colors = [C_POS if v >= 0 else C_NEG for v in table["mean"]]
        labels = [f"{g}\n(n={n})" for g, n in zip(table.index.astype(str), table["count"])]
        bars = ax.bar(labels, table["mean"], color=colors, width=0.6)
        ax.bar_label(bars, fmt="%+.1f", fontsize=10, fontweight="bold", padding=2)
        ax.axhline(0, color="#334155", lw=1)
        ax.set_title(f"Surplus vs draft slot, {title}", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Win Shares above/below pick expectation")
        ax.grid(axis="y", color="#e2e8f0", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Which profiles get over/under-drafted? (all drafts 2000-2020)",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.005, "Players who never appeared in the NBA (age/position unrecorded) "
             "are excluded from group means.", ha="center", fontsize=8.5,
             color="#64748b", style="italic")
    fig.tight_layout()
    out = FIGURES / f"q2_profiles_{set_name}.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


# --------------------------------------------------------------------------- main

def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    core = load_core()

    df_simple, feats_simple = add_simple_features(core)
    run_feature_set(df_simple, feats_simple, "simple")

    advanced = add_advanced_features(core)
    if advanced is None:
        print("\nADVANCED set skipped: data/raw/combine_stats.csv not found "
              "(run src/download_combine.py to enable it)")
    else:
        df_adv, feats_adv = advanced
        run_feature_set(df_adv, feats_adv, "advanced")

        college = add_college_features(df_adv, feats_adv)
        if college is None:
            print("\nCOLLEGE set skipped: data/raw/college_stats.csv not found "
                  "(run src/download_college.py to enable it)")
        else:
            df_col, feats_col = college
            run_feature_set(df_col, feats_col, "college")


if __name__ == "__main__":
    main()
