"""
QUESTION 2 (part 2): player ARCHETYPES - do certain physical profiles get
systematically over- or under-drafted?

Method: KMeans clustering on the pre-draft PHYSICAL profile only - age at draft,
height, weight, wingspan (standardized). Position is deliberately NOT an input:
if the archetypes recover position groups by themselves, that validates the
clustering; and mixing one-hot categoricals into KMeans distances distorts them.

k is chosen by silhouette score over k=3..8 (printed, so the choice is justified
in the writeup and not arbitrary).

For the app: each player also gets 2D coordinates (PCA of the same standardized
features) - this is the "Draft Map" scatter the website shows.

Outputs:
  data/processed/clusters.csv     - player, cluster id+label, pca_x/pca_y,
                                    features, ws_first4, surplus (app input)
  figures/q2_cluster_map.png      - the 2D map, colored by archetype
  figures/q2_cluster_outcomes.png - avg pick, star rate, surplus per archetype
  printed cluster profile table   - for naming the archetypes in the writeup

Run from the project root:  python src/q2_clusters.py   (after q2_model.py, since
it reuses the combine merge; combine_stats.csv must exist)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from q2_model import add_advanced_features, load_core

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
FIGURES = BASE_DIR / "figures"

# pure physique: age is deliberately excluded - it adds no cluster separation
# (all clusters average ~21.1-21.5 years) but its discrete values dominate the
# PCA's second axis and draw distracting horizontal bands on the map
CLUSTER_FEATURES = ["height", "weight", "wingspan"]
K_RANGE = range(3, 9)
SMOOTH_WINDOW = 5
STAR_MIN = 20.0  # same tier definition as q1

# fixed categorical palette (assigned by cluster size order, never cycled)
PALETTE = ["#1d4ed8", "#0891b2", "#b45309", "#7c3aed", "#be185d", "#4d7c0f",
           "#475569", "#0d9488"]


def prepare() -> tuple[pd.DataFrame, np.ndarray]:
    advanced = add_advanced_features(load_core())
    if advanced is None:
        raise SystemExit("combine_stats.csv missing - run src/download_combine.py")
    df, _ = advanced

    per_pick = (df.groupby("pick")["ws_first4"].mean()
                .rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean())
    df["surplus"] = df["ws_first4"] - df["pick"].map(per_pick)

    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    return df, X


def choose_k(X: np.ndarray) -> int:
    print("silhouette scores (higher = cleaner separation):")
    scores = {}
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
        print(f"  k={k}: {scores[k]:.3f}")
    best = max(scores, key=scores.get)
    print(f"chosen k = {best}")
    return best


def describe_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Profile table used to NAME the archetypes (and for the writeup).
    OUTCOME columns are computed on MEASURED players only: the 36% without combine
    data are median-imputed to the center of the space, land in the middle cluster,
    and are dominated by never-played busts - including them would fabricate that
    cluster's underperformance."""
    measured = df[df["has_combine"]] if "has_combine" in df.columns else df
    profile = measured.groupby("cluster").agg(
        n_measured=("player", "count"),
        age=("age_at_draft", "mean"),
        height_cm=("height", lambda s: s.mean() * 2.54),  # BBRef inches -> cm
        weight_kg=("weight", lambda s: s.mean() * 0.4536),
        wingspan_cm=("wingspan", lambda s: s.mean() * 2.54),
        main_pos=("pos", lambda s: s.mode().iloc[0] if not s.mode().empty else "?"),
        avg_pick=("pick", "mean"),
        mean_ws4=("ws_first4", "mean"),
        star_rate=("ws_first4", lambda s: (s >= STAR_MIN).mean()),
        mean_surplus=("surplus", "mean"),
    ).round(2)
    profile.insert(0, "n_total", df.groupby("cluster")["player"].count())
    return profile.sort_values("n_total", ascending=False)


def auto_label(profile_row: pd.Series, all_profiles: pd.DataFrame) -> str:
    """Short readable label like 'young long bigs' from the profile extremes."""
    parts = []
    # only mention age if the clusters actually differ meaningfully on it
    if all_profiles["age"].max() - all_profiles["age"].min() >= 0.8:
        if profile_row["age"] <= all_profiles["age"].min() + 0.3:
            parts.append("young")
        elif profile_row["age"] >= all_profiles["age"].max() - 0.3:
            parts.append("older")
    size_rank = all_profiles["height_cm"].rank()
    if size_rank[profile_row.name] == size_rank.max():
        parts.append("bigs")
    elif size_rank[profile_row.name] == size_rank.min():
        parts.append("small guards")
    else:
        parts.append("wings/mid-size")
    return " ".join(parts) or f"cluster {profile_row.name}"


def plot_map(df: pd.DataFrame, labels: dict) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    order = df["cluster"].value_counts().index  # size order = color order
    for color, cl in zip(PALETTE, order):
        sub = df[df["cluster"] == cl]
        ax.scatter(sub["pca_x"], sub["pca_y"], s=26, color=color, alpha=0.65,
                   linewidths=0, label=f"{labels[cl]} (n={len(sub)})")
    ax.set_title("The Draft Map: 2000-2020 draftees by physical profile "
                 "(PCA of height, weight, wingspan)",
                 fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(color="#e2e8f0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURES / "q2_cluster_map.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


def plot_outcomes(profile: pd.DataFrame, labels: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5))
    names = [labels[i] for i in profile.index]
    colors = PALETTE[: len(profile)]
    panels = [("avg_pick", "Average draft pick", "%.0f"),
              ("star_rate", "Star rate (ws4 ≥ 20)", "%.0%%"),
              ("mean_surplus", "Mean surplus vs slot (WS)", "%+.1f")]
    for ax, (col, title, fmt) in zip(axes, panels):
        vals = profile[col]
        bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.6)
        if col == "star_rate":
            ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals[::-1]],
                         fontsize=9.5, fontweight="bold", padding=3)
        else:
            ax.bar_label(bars, fmt=fmt, fontsize=9.5, fontweight="bold", padding=3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="x", color="#e2e8f0", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if col == "mean_surplus":
            ax.axvline(0, color="#334155", lw=1)
    fig.suptitle("Archetype outcomes: where does the market misprice a body type?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "q2_cluster_outcomes.png"
    fig.savefig(out, dpi=300)
    print(f"Saved: {out}")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    df, X = prepare()

    k = choose_k(X)
    df["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["pca_x"], df["pca_y"] = coords[:, 0], coords[:, 1]
    print(f"PCA explains {pca.explained_variance_ratio_.sum():.0%} of variance")

    profile = describe_clusters(df)
    labels = {cl: auto_label(profile.loc[cl], profile) for cl in profile.index}
    profile["label"] = [labels[cl] for cl in profile.index]
    print("\nCluster profiles (name them properly in the writeup):")
    print(profile.to_string())

    df["cluster_label"] = df["cluster"].map(labels)
    keep = ["draft_year", "pick", "player", "cluster", "cluster_label",
            "pca_x", "pca_y", "age_at_draft", "height", "weight", "wingspan",
            "has_combine", "pos", "ws_first4", "surplus"]
    # has_combine=False means physique was median-imputed - these players sit
    # artificially near the map's center; the app should mark them (limitation
    # also noted in the writeup)
    out_path = PROCESSED / "clusters.csv"
    df[keep].to_csv(out_path, index=False)
    print(f"\nSaved app input: {out_path}")

    plot_map(df, labels)
    plot_outcomes(profile, labels)


if __name__ == "__main__":
    main()
