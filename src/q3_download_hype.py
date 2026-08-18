"""
QUESTION 3 (data step): downloads the pre-draft "hype" measure for each player.

Hype proxy: Wikipedia pageviews of the player's article in the months BEFORE his
draft. Public attention on a prospect (news, highlights, mock drafts) shows up
directly in how many people open his Wikipedia page.

Scope: draft classes 2016-2020 only. The Wikimedia pageviews API begins in
July 2015, so earlier classes have no pre-draft window - this constraint is
stated in the writeup.

CRAWLING ETIQUETTE (learned the hard way - the first version got rate-limited):
  * 1.2-2.5s randomized sleep between requests (~20 minutes total for 300 players)
  * on 429/503 the script BACKS OFF and retries (respecting the Retry-After header)
  * progress is saved every 20 players and a re-run RESUMES where it stopped,
    so an interruption never loses work

Output: data/raw/hype_wikipedia.csv with one row per player:
  draft_year, pick, player, wiki_title, resolved (bool), views_predraft,
  months_with_data
The wiki_title column is saved so mismatches can be audited by hand.

Run from the project root:  python src/q3_download_hype.py
(safe to re-run after an interruption - it continues, not restarts)
"""

import random
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "data" / "raw"
PROCESSED = BASE_DIR / "data" / "processed"
TARGET = RAW / "hype_wikipedia.csv"

HYPE_YEARS = range(2016, 2021)

SEARCH_API = "https://en.wikipedia.org/w/api.php"
PAGEVIEWS_API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                 "en.wikipedia/all-access/user/{title}/monthly/{start}/{end}")
# Wikimedia's policy asks clients to identify themselves with a contact address
HEADERS = {"User-Agent": "HUJI-course-67978-student-project/1.1 "
                         "(asafvit2@gmail.com) python-requests"}

MAX_RETRIES = 5
SAVE_EVERY = 20  # players

session = requests.Session()
session.headers.update(HEADERS)


def polite_sleep() -> None:
    time.sleep(random.uniform(1.2, 2.5))


def get_with_backoff(url: str, params: dict | None = None) -> requests.Response:
    """GET with exponential backoff on rate-limit (429) and server errors (5xx)."""
    for attempt in range(1, MAX_RETRIES + 1):
        response = session.get(url, params=params, timeout=30)
        if response.status_code in (429, 503):
            wait = int(response.headers.get("Retry-After", 0)) or 20 * attempt
            print(f"    rate-limited (HTTP {response.status_code}) - "
                  f"waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        return response
    return response  # last response; caller will raise_for_status if needed


def resolve_title(player: str) -> str | None:
    """Find the player's Wikipedia article title (or None if nothing matches)."""
    params = {
        "action": "query", "list": "search", "format": "json",
        "srsearch": f"{player} basketball", "srlimit": 3,
    }
    r = get_with_backoff(SEARCH_API, params)
    r.raise_for_status()
    hits = r.json().get("query", {}).get("search", [])
    if not hits:
        return None
    for hit in hits:  # prefer a title starting with the player's first name
        if hit["title"].lower().startswith(player.split()[0].lower()):
            return hit["title"]
    return hits[0]["title"]


def predraft_views(title: str, draft_year: int) -> tuple[float, int]:
    """Sum of monthly pageviews from July (draft_year-1) to May (draft_year)."""
    url = PAGEVIEWS_API.format(title=quote(title.replace(" ", "_"), safe=""),
                               start=f"{draft_year - 1}0701",
                               end=f"{draft_year}0531")
    r = get_with_backoff(url)
    if r.status_code == 404:  # article did not exist yet -> nobody could view it
        return 0.0, 0
    r.raise_for_status()
    items = r.json().get("items", [])
    return float(sum(i["views"] for i in items)), len(items)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    core = pd.read_csv(PROCESSED / "draft_value.csv")
    players = core[core["draft_year"].isin(HYPE_YEARS)]

    # RESUME: load whatever a previous (interrupted) run already saved
    done_keys: set = set()
    rows: list = []
    if TARGET.exists():
        previous = pd.read_csv(TARGET)
        rows = previous.to_dict("records")
        done_keys = set(zip(previous["draft_year"], previous["pick"]))
        print(f"Resuming: {len(rows)} players already downloaded, "
              f"{len(players) - len(rows)} to go")

    todo = [r for r in players.itertuples()
            if (r.draft_year, r.pick) not in done_keys]
    if not todo:
        print(f"Nothing to do - {TARGET} is complete ({len(rows)} players).")
        return

    print(f"Downloading pre-draft Wikipedia pageviews for {len(todo)} players "
          f"(~{len(todo) * 3.7 / 60:.0f} minutes at polite speed) ...")

    for i, row in enumerate(todo, 1):
        title, views, months, resolved = None, 0.0, 0, False
        try:
            title = resolve_title(row.player)
            polite_sleep()
            if title is not None:
                resolved = True
                views, months = predraft_views(title, row.draft_year)
        except Exception as exc:
            print(f"  {row.player}: FAILED ({exc}) - continuing")
        rows.append({
            "draft_year": row.draft_year, "pick": row.pick, "player": row.player,
            "wiki_title": title, "resolved": resolved,
            "views_predraft": views, "months_with_data": months,
        })
        if i % SAVE_EVERY == 0:
            pd.DataFrame(rows).to_csv(TARGET, index=False, encoding="utf-8")
            print(f"  {i}/{len(todo)} done - progress saved")
        polite_sleep()

    out = pd.DataFrame(rows)
    out.to_csv(TARGET, index=False, encoding="utf-8")

    print(f"\nSaved {len(out)} rows to {TARGET}")
    print(f"Resolved to a Wikipedia article: {out['resolved'].mean():.0%}")
    print("\nTop 10 by pre-draft views (eyeball check - these should be famous "
          "prospects):")
    top = out.nlargest(10, "views_predraft")[["draft_year", "pick", "player",
                                              "wiki_title", "views_predraft"]]
    print(top.to_string(index=False))
    print("\nIf a wiki_title looks like the wrong person, fix it by hand in the "
          "CSV and re-run q3_hype.py (the analysis reads this file).")


if __name__ == "__main__":
    main()
