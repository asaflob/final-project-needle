"""
Builds the core table of the project: one row per drafted player (2000-2020),
with the project's central metric  ws_first4  = Win Shares accumulated in the
player's FIRST 4 SEASONS in the NBA (the rookie-contract window).

Why first 4 seasons and not career Win Shares?
    Career WS is unfair: a player drafted in 2003 had ~20 seasons to accumulate WS,
    a player drafted in 2019 had ~5. Using a fixed window makes every player in the
    2000-2020 draft classes comparable by construction (all have completed 4 seasons).
    Players who washed out of the league in fewer than 4 seasons keep the WS they
    earned - washing out early IS a draft outcome, so this is the correct treatment.

Inputs  (see src/download_data.py):
    data/raw/draft.csv     - draft history, one row per pick (career totals included)
    data/raw/Advanced.csv  - season-level advanced stats, one row per player-season

Outputs:
    data/processed/draft_value.csv       - the core table used by every analysis
    data/processed/unmatched_players.csv - drafted players we could not match to
                                           season data (input for src/audit.py)

Run from the project root:  python src/data_prep.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DRAFT_YEARS = range(2000, 2021)  # 2000-2020 inclusive
ROOKIE_WINDOW = 4  # first N seasons that define the metric

# candidate column names in draft.csv (the file may use different capitalizations)
YEAR_CANDIDATES = ["Year", "Draft_Yr", "year", "draft_year", "DraftYear"]
PLAYER_CANDIDATES = ["Player", "player", "name", "Name"]
PICK_CANDIDATES = ["Pk", "pick", "Pick", "overall_pick"]

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Normalize a player name so the same player matches across datasets.

    Handles accents (Dončić -> doncic), punctuation (P.J. -> pj), asterisks that
    Basketball-Reference adds to Hall-of-Famers, and generational suffixes (Jr./III).
    """
    if not isinstance(name, str):
        return ""
    # strip accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.lower().replace("*", "")
    # remove anything that is not a letter or space (drops dots, apostrophes, hyphens)
    name = re.sub(r"[^a-z ]", "", name)
    tokens = [t for t in name.split() if t not in NAME_SUFFIXES]
    return " ".join(tokens)


def find_column(df: pd.DataFrame, candidates: list, what: str) -> str:
    """Return the first candidate column that exists in df, or exit with a clear error."""
    for col in candidates:
        if col in df.columns:
            return col
    raise SystemExit(
        f"Could not find a '{what}' column in the file.\n"
        f"Tried: {candidates}\nActual columns: {list(df.columns)}"
    )


def load_draft() -> pd.DataFrame:
    """Load the draft table, keep 2000-2020, return a clean frame."""
    path = RAW_DIR / "draft_full.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path} - run:  python src/download_data.py")

    df = pd.read_csv(path, encoding="utf-8")
    year_col = find_column(df, YEAR_CANDIDATES, "draft year")
    player_col = find_column(df, PLAYER_CANDIDATES, "player name")
    pick_col = find_column(df, PICK_CANDIDATES, "pick number")

    df = df.rename(columns={year_col: "draft_year", player_col: "player", pick_col: "pick"})
    df = df[df["draft_year"].isin(DRAFT_YEARS)].copy()
    df = df.dropna(subset=["player", "pick"])
    df["pick"] = df["pick"].astype(int)
    df["name_key"] = df["player"].map(normalize_name)

    # keep career WS too (if present) - useful for showing WHY the old metric was biased
    if "WS" in df.columns:
        df = df.rename(columns={"WS": "ws_career"})

    keep = ["draft_year", "pick", "player", "name_key"]
    for optional in ["Rd", "Tm", "College", "Age", "ws_career"]:
        if optional in df.columns:
            keep.append(optional)
    return df[keep]


def load_seasons() -> pd.DataFrame:
    """Load season-level advanced stats; one row per player-season with WS."""
    path = RAW_DIR / "Advanced.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}\n"
            "Download 'Advanced.csv' from Kaggle (instructions in src/download_data.py)."
        )

    df = pd.read_csv(path)
    required = {"season", "player", "ws"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Advanced.csv is missing expected columns {missing}. "
            f"Actual columns: {list(df.columns)}"
        )

    # our draft window starts in 2000, so no need for seasons before 2001
    # (in this dataset 'season' is the ENDING year: 2001 means the 2000-01 season)
    df = df[df["season"] >= 2001].copy()

    # players traded mid-season appear once per team PLUS a combined row whose team
    # is '2TM'/'3TM'... - keep only the combined row, otherwise the season's WS is
    # counted twice. (verified: without this, 4,925 player-season rows are duplicated)
    dedup_id = "player_id" if "player_id" in df.columns else "player"
    df["is_combined"] = df["team"].astype(str).str.endswith("TM")
    df = df.sort_values("is_combined", ascending=False)
    df = df.drop_duplicates(subset=[dedup_id, "season"], keep="first")

    df["name_key"] = df["player"].map(normalize_name)
    # age and pos are kept for q2 (model features: age at draft, position)
    cols = [c for c in ["name_key", "season", "ws", "age", "pos"] if c in df.columns]
    return df[cols]


def build_ws_first4(draft: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """For every drafted player, sum WS over his first ROOKIE_WINDOW seasons."""
    merged = draft.merge(seasons, on="name_key", how="left")

    # a player's rookie-window seasons: the first 4 seasons AFTER his draft year.
    # the season window also protects against name collisions (two different players
    # with the same normalized name active in different eras).
    in_window = (merged["season"] > merged["draft_year"]) & (
        merged["season"] <= merged["draft_year"] + ROOKIE_WINDOW
    )
    window = merged[in_window].sort_values("season")

    agg_spec = dict(
        ws_first4=("ws", "sum"),
        n_seasons_first4=("season", "nunique"),
        first_season=("season", "min"),
    )
    # age/pos of the first season played - used by q2 to derive pre-draft features
    if "age" in window.columns:
        agg_spec["first_season_age"] = ("age", "first")
    if "pos" in window.columns:
        agg_spec["pos"] = ("pos", "first")

    agg = window.groupby(["draft_year", "pick"]).agg(**agg_spec).reset_index()

    # approximate age on draft night: age in the first season played, minus the
    # years between the draft and that season (a player can debut late)
    if "first_season_age" in agg.columns:
        agg["age_at_draft"] = agg["first_season_age"] - (
            agg["first_season"] - 1 - agg["draft_year"]
        )
        agg = agg.drop(columns=["first_season_age"])

    # per-season rows inside the window - exported for the app's Explorer page
    export = window.copy()
    export["season_index"] = export["season"] - export["draft_year"]
    export[["draft_year", "pick", "player", "season", "season_index", "ws"]].to_csv(
        PROCESSED_DIR / "player_seasons.csv", index=False)

    result = draft.merge(agg, on=["draft_year", "pick"], how="left")
    # players with no matched seasons: either never played in the NBA (WS = 0 is the
    # honest value for the value-curve) or a name-matching failure (audit.py checks).
    result["matched"] = result["n_seasons_first4"].notna()
    result["ws_first4"] = result["ws_first4"].fillna(0.0)
    result["n_seasons_first4"] = result["n_seasons_first4"].fillna(0).astype(int)
    return result


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    draft = load_draft()
    print(f"Draft table: {len(draft)} picks, {draft['draft_year'].nunique()} draft years")

    seasons = load_seasons()
    print(f"Season table: {len(seasons)} player-seasons (2001 onward)")

    result = build_ws_first4(draft, seasons)

    out = PROCESSED_DIR / "draft_value.csv"
    result.drop(columns=["name_key"]).to_csv(out, index=False)
    print(f"\nSaved core table: {out}  ({len(result)} rows)")

    unmatched = result[~result["matched"]]
    unmatched_out = PROCESSED_DIR / "unmatched_players.csv"
    unmatched.drop(columns=["name_key"]).to_csv(unmatched_out, index=False)

    match_rate = result["matched"].mean()
    print(f"Matched to season data: {result['matched'].sum()}/{len(result)} "
          f"({match_rate:.1%})")
    print(f"Unmatched players saved to: {unmatched_out}")
    print("NOTE: some unmatched players are legitimate (drafted but never played in "
          "the NBA, e.g. international stashes). src/audit.py digs into this.")


if __name__ == "__main__":
    main()
