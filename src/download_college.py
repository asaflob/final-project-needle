"""
Downloads each drafted player's COLLEGE stats from his Basketball-Reference page.
This is the input for q2's third feature set (college production - how a prospect
actually played, not just how he measures).

DESIGN - fetch and parse are separate on purpose:
  1. FETCH: every player page is downloaded once and cached (gzipped) under
     data/raw/bbref_cache/. Interruptions are free: a re-run skips cached pages.
  2. PARSE: college stats are extracted from the cached pages. If the parser turns
     out to have a bug, fixing it requires NO re-crawling - just re-run the script.

CRAWLING ETIQUETTE: Basketball-Reference allows ~20 requests/minute and blocks
violators for a full hour. We stay at ~14/min (3.5-4.5s randomized sleep), honor
Retry-After on 429, and cache everything. Full run: ~85 minutes for ~1,270 pages.
Safe to Ctrl+C and re-run anytime - it resumes.

A quirk worth knowing (documented for the writeup's Impediments section): BBRef
ships secondary tables inside HTML comments (an old lazy-loading trick), so the
college table is invisible to normal parsers. We strip the comment markers before
parsing.

Output: data/raw/college_stats.csv - one row per drafted player 2000-2020:
  draft_year, pick, player, bbref_id, has_page, has_college_stats,
  college_season, school, g, mp, pts, trb, ast, fg_pct, fg3_pct, ft_pct

Run from the project root:  python src/download_college.py
"""

import gzip
import random
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "data" / "raw"
CACHE = RAW / "bbref_cache"
TARGET = RAW / "college_stats.csv"

DRAFT_YEARS = range(2000, 2021)
BASE_URL = "https://www.basketball-reference.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
MAX_RETRIES = 4

session = requests.Session()
session.headers.update(HEADERS)


def polite_sleep() -> None:
    time.sleep(random.uniform(3.5, 4.5))  # ~14 req/min, under BBRef's 20/min limit


def fetch_cached(path: str) -> str | None:
    """Fetch a BBRef path with disk caching. Returns HTML, or None on failure."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / (path.strip("/").replace("/", "_") + ".html.gz")
    if cache_file.exists():
        return gzip.decompress(cache_file.read_bytes()).decode("utf-8")

    for attempt in range(1, MAX_RETRIES + 1):
        response = session.get(BASE_URL + path, timeout=60)
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 0)) or 60 * attempt
            print(f"    rate-limited - waiting {wait}s (attempt {attempt})")
            time.sleep(wait)
            continue
        if response.status_code == 404:
            return None
        response.raise_for_status()
        response.encoding = "utf-8"
        cache_file.write_bytes(gzip.compress(response.text.encode("utf-8")))
        polite_sleep()
        return response.text
    return None


def player_links_for_year(year: int) -> list[dict]:
    """From a draft page, get every pick's player-page link (if he has one)."""
    html = fetch_cached(f"/draft/NBA_{year}.html")
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("table tbody tr"):
        pick_cell = tr.find(attrs={"data-stat": "pick_overall"})
        player_cell = tr.find(attrs={"data-stat": "player"})
        if pick_cell is None or player_cell is None:
            continue
        pick_text = pick_cell.get_text(strip=True)
        if not pick_text.isdigit():
            continue  # separator rows between rounds
        link = player_cell.find("a")
        rows.append({
            "draft_year": year,
            "pick": int(pick_text),
            "player": player_cell.get_text(strip=True),
            "href": link["href"] if link else None,
        })
    return rows


def parse_college_stats(html: str) -> dict | None:
    """Extract the FINAL college season's per-game stats from a player page.

    The college table on a BBRef player page (verified against cached pages):
      * hidden inside an HTML comment (uncommented below)
      * identity column is 'College' (not 'School' - that is the college site)
      * two header rows -> pandas MultiIndex columns grouped as
        (blank: Season/Age/College/G/MP), Totals, Shooting, Per Game
    """
    html = html.replace("<!--", "").replace("-->", "")
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return None

    college = None
    for t in tables:
        if isinstance(t.columns, pd.MultiIndex):
            lower = [str(c[-1]) for c in t.columns]
            if "College" in lower and "Season" in lower:
                college = t
                break
    if college is None:
        return None

    # flatten the two-level header: Per Game -> _pg suffix, Totals -> _tot
    flat = []
    for top, low in college.columns:
        top, low = str(top), str(low)
        if top.startswith("Per Game"):
            flat.append(f"{low}_pg")
        elif top.startswith("Totals"):
            flat.append(f"{low}_tot")
        else:
            flat.append(low)
    college.columns = flat

    # keep real season rows ('2016-17'), drop the 'Career' summary row
    seasons = college[college["Season"].astype(str).str.match(r"^\d{4}-\d{2}$",
                                                              na=False)]
    if seasons.empty:
        return None
    final = seasons.iloc[-1]

    out = {"college_season": final["Season"], "school": final.get("College")}
    mapping = {"G": "g", "MP_pg": "mp", "PTS_pg": "pts", "TRB_pg": "trb",
               "AST_pg": "ast", "FG%": "fg_pct", "3P%": "fg3_pct",
               "FT%": "ft_pct"}
    for src, dst in mapping.items():
        out[dst] = pd.to_numeric(final.get(src), errors="coerce")
    return out


def main() -> None:
    # RESUME at the output level too: skip players already in the CSV
    done: set = set()
    rows: list = []
    if TARGET.exists():
        previous = pd.read_csv(TARGET)
        rows = previous.to_dict("records")
        done = set(zip(previous["draft_year"], previous["pick"]))
        print(f"Resuming: {len(rows)} players already parsed")

    print("Collecting player links from draft pages ...")
    all_players = []
    for year in DRAFT_YEARS:
        all_players.extend(player_links_for_year(year))
    todo = [p for p in all_players if (p["draft_year"], p["pick"]) not in done]
    print(f"{len(all_players)} picks total, {len(todo)} to fetch/parse "
          f"(~{len(todo) * 4 / 60:.0f} min on a cold cache)")

    for i, p in enumerate(todo, 1):
        record = {**{k: p[k] for k in ["draft_year", "pick", "player"]},
                  "bbref_id": None, "has_page": False, "has_college_stats": False}
        try:
            if p["href"]:
                record["bbref_id"] = re.sub(r"\.html$", "",
                                            p["href"].split("/")[-1])
                html = fetch_cached(p["href"])
                if html:
                    record["has_page"] = True
                    stats = parse_college_stats(html)
                    if stats:
                        record["has_college_stats"] = True
                        record.update(stats)
        except Exception as exc:
            print(f"  {p['player']}: FAILED ({exc}) - continuing")
        rows.append(record)
        if i % 20 == 0:
            pd.DataFrame(rows).to_csv(TARGET, index=False, encoding="utf-8")
            print(f"  {i}/{len(todo)} done - progress saved")

    out = pd.DataFrame(rows)
    out.to_csv(TARGET, index=False, encoding="utf-8")

    print(f"\nSaved {len(out)} rows to {TARGET}")
    print(f"Has a BBRef page:    {out['has_page'].mean():.0%}")
    print(f"Has college stats:   {out['has_college_stats'].mean():.0%} "
          f"(internationals and preps-to-pros players legitimately have none)")
    parsed = out[out["has_college_stats"] == True]  # noqa: E712
    if not parsed.empty:
        print("\nSample (eyeball check):")
        print(parsed[["draft_year", "pick", "player", "school", "pts", "trb",
                      "ast"]].tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
