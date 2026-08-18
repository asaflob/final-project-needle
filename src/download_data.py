"""
Downloads the raw data files into data/raw/.

Run this ONCE from the project root:  python src/download_data.py
(takes ~2 minutes: 21 pages with a polite delay between requests)

Files produced:
1. draft_full.csv - all NBA draft picks 2000-2020 (both rounds), scraped from the
                    official draft pages on Basketball-Reference. Replaces the
                    milestone's draft.csv, which covered only 2000-2009 round 1 and
                    had broken encoding for accented names (Miličić, Türkoğlu...).
2. Advanced.csv   - season-level advanced stats from the Kaggle dataset
                    "NBA Stats (1947-present)" by Sumitro Datta. Kaggle requires a
                    login, so it is downloaded manually (instructions printed below).
"""

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DRAFT_YEARS = range(2000, 2021)  # 2000-2020 inclusive

# Basketball-Reference allows polite scraping; stay well under their rate limit
SECONDS_BETWEEN_REQUESTS = 3.5
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

KAGGLE_INSTRUCTIONS = """
MANUAL STEP - season-level Win Shares (skip if data/raw/Advanced.csv already exists)
------------------------------------------------------------------------------------
1. Open: https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats
2. Download the dataset (free Kaggle account needed).
3. Copy the file  Advanced.csv  into:  data/raw/
4. Then run:  python src/data_prep.py
"""


def fetch_draft_year(year: int) -> pd.DataFrame:
    """Scrape one draft year's table from Basketball-Reference."""
    url = f"https://www.basketball-reference.com/draft/NBA_{year}.html"
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    # the page is UTF-8, but the server's headers don't always say so - without this
    # line requests guesses latin-1 and accented names (Dončić, Türkoğlu) get mangled
    response.encoding = "utf-8"

    # the page has one big table; header spans two rows -> take the second level
    tables = pd.read_html(StringIO(response.text), header=1)
    df = tables[0]

    # the table contains separator rows (repeated headers / "Round 2" banner).
    # real pick rows have a numeric 'Pk'; separators do not - and they also mark
    # the boundary between round 1 and round 2.
    df["is_pick"] = pd.to_numeric(df["Pk"], errors="coerce").notna()
    df["Rd"] = (~df["is_pick"]).cumsum() + 1  # separators seen so far -> round number
    df = df[df["is_pick"]].copy()

    df["Pk"] = df["Pk"].astype(int)
    df["Year"] = year
    keep = ["Year", "Rd", "Pk", "Tm", "Player", "College", "Yrs", "WS", "WS/48"]
    return df[[c for c in keep if c in df.columns]]


def download_draft_tables() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "draft_full.csv"
    if target.exists():
        print(f"Already exists, skipping: {target}")
        return

    frames = []
    for year in DRAFT_YEARS:
        print(f"Fetching draft {year} ...", flush=True)
        frames.append(fetch_draft_year(year))
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    draft = pd.concat(frames, ignore_index=True)
    draft.to_csv(target, index=False, encoding="utf-8")
    print(f"\nSaved {len(draft)} picks ({DRAFT_YEARS.start}-{DRAFT_YEARS.stop - 1}) "
          f"to {target}")


if __name__ == "__main__":
    download_draft_tables()
    print(KAGGLE_INSTRUCTIONS)
