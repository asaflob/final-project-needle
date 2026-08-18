"""
Downloads NBA Draft Combine measurements (height, weight, wingspan...) via nba_api
and SAVES them to data/raw/combine_stats.csv. Nothing else uses this file yet -
it is the input for the ADVANCED feature set of q2_model.py.

Run ONCE from the project root:  python src/download_combine.py
(takes ~1 minute; needs internet access to stats.nba.com)
"""

import time
from pathlib import Path

import pandas as pd

try:
    from nba_api.stats.endpoints import draftcombinestats
except ImportError:
    raise SystemExit("nba_api is not installed - run:  pip install nba_api")

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TARGET = RAW_DIR / "combine_stats.csv"

# combine seasons are named by the season AFTER the combine: the 2000 draft class
# measured at the "2000-01" combine, ... the 2020 class at "2020-21"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2000, 2021)]


def main() -> None:
    if TARGET.exists():
        print(f"Already exists, skipping: {TARGET}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in SEASONS:
        try:
            result = draftcombinestats.DraftCombineStats(season_all_time=season,
                                                         timeout=60)
            df = result.get_data_frames()[0]
            df["COMBINE_SEASON"] = season
            frames.append(df)
            print(f"{season}: {len(df)} players")
        except Exception as exc:  # one bad year should not kill the whole download
            print(f"{season}: FAILED ({exc}) - continuing")
        time.sleep(1.0)  # polite pacing for the NBA stats API

    if not frames:
        raise SystemExit("Nothing downloaded - is stats.nba.com reachable?")

    combine = pd.concat(frames, ignore_index=True)
    combine.to_csv(TARGET, index=False, encoding="utf-8")
    print(f"\nSaved {len(combine)} rows to {TARGET}")
    print("Columns:", list(combine.columns))


if __name__ == "__main__":
    main()
