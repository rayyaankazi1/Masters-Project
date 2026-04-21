"""
signal/scraping/scraper.py
──────────────────────────
Stage 1 of the signal pipeline (README §Pipeline 1 — Stage 1).

Scrapes presidential speeches from the Casa Rosada archive and writes a
single flat CSV to data/raw/speeches_raw.csv.  This file is the only
output of this stage; downstream stages (preprocessing, LDA, scoring)
read from it and add nothing back here.

Columns produced
────────────────
speech_id       int     row index (0-based), stable within a run
date            date    publication date parsed from the page
url             str     canonical source URL
president       str     name label (Kirchner | CFK | Macri | AF | Milei)
president_id    int     numeric id (1–5) matching PRESIDENTS config
year            int
month           int
year_month      str     e.g. "2024-03"
n_words         int     word count of raw text (post basic normalisation)
text_raw        str     full speech text as extracted from the page
"""

import os
import re
import time
import unicodedata
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Output path ───────────────────────────────────────────────────────────────
# Resolve relative to the project root (two levels up from this file)
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUTPUT_DIR   = os.path.join(_PROJECT_ROOT, "data", "raw")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "speeches_raw.csv")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL      = "https://www.casarosada.gob.ar/informacion/discursos"
STOP_DATE_STR = "2015-12-10"   # scrape nothing older than this
PAGE_SIZE     = 40
SPEECH_DELAY  = 0.4            # seconds between individual speech requests
PAGE_DELAY    = 1.0            # seconds between index pages
TIMEOUT       = 20
MIN_WORDS     = 50             # discard very short items (greetings, captions)

# ── Presidents ────────────────────────────────────────────────────────────────
PRESIDENTS = [
    ("2003-05-25", "2007-12-10", "Kirchner", 1),
    ("2007-12-10", "2015-12-10", "CFK",      2),
    ("2015-12-10", "2019-12-10", "Macri",    3),
    ("2019-12-10", "2023-12-10", "AF",       4),
    ("2023-12-10", "2099-12-31", "Milei",    5),
]
_PRES_RANGES = [
    (pd.Timestamp(s), pd.Timestamp(e), name, pid)
    for s, e, name, pid in PRESIDENTS
]

# ── Internals ─────────────────────────────────────────────────────────────────
_DATE_RE  = re.compile(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})")
_CLEAN_RE = re.compile(r"[^a-z\s]")
_MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


def _normalise(text: str) -> str:
    """Lowercase + strip accents + remove non-alpha characters.
    Used only for word-count filtering here; full spaCy preprocessing
    happens in signal/preprocessing/.
    """
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return _CLEAN_RE.sub("", text)


def _parse_date(raw: str) -> Optional[datetime]:
    m = _DATE_RE.search(raw.lower())
    if not m:
        return None
    month = _MONTHS_ES.get(m.group(2))
    if not month:
        return None
    return datetime(int(m.group(3)), month, int(m.group(1)))


def _get_president(date: Optional[datetime]) -> tuple[str, int]:
    if date is None:
        return ("Unknown", 0)
    ts = pd.Timestamp(date)
    for start, end, name, pid in _PRES_RANGES:
        if start <= ts <= end:
            return (name, pid)
    return ("Unknown", 0)


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    return s


# ── Core scraper ──────────────────────────────────────────────────────────────

def scrape(stop_date: str = STOP_DATE_STR) -> pd.DataFrame:
    """
    Paginate through the Casa Rosada discursos index, fetch each speech,
    and return a DataFrame.  Stops when it encounters a speech dated
    before `stop_date`.

    Parameters
    ----------
    stop_date : str
        ISO date string (YYYY-MM-DD).  Speeches older than this are skipped.

    Returns
    -------
    pd.DataFrame
        One row per speech, columns as documented in the module docstring.
    """
    limit   = pd.to_datetime(stop_date)
    session = _make_session()
    records = []
    offset  = 0

    print(f"Scraping Casa Rosada — stopping before {stop_date}\n")

    while True:
        # ── Fetch index page ──────────────────────────────────────────────
        try:
            resp = session.get(f"{BASE_URL}?start={offset}", timeout=TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Index page failed at offset {offset}: {e}")
            break

        soup  = BeautifulSoup(resp.text, "html.parser")
        links = list(dict.fromkeys(
            "https://www.casarosada.gob.ar" + a["href"]
            for a in soup.find_all("a", href=True)
            if "/informacion/discursos/" in a["href"]
            and "start=" not in a["href"]
        ))

        if not links:
            print("No more links found — scrape complete.")
            break

        print(f"Page offset={offset}  ({len(links)} links found)")
        stop_flag = False

        # ── Fetch each speech ─────────────────────────────────────────────
        for url in links:
            try:
                r = session.get(url, timeout=TIMEOUT)
                r.raise_for_status()
                page = BeautifulSoup(r.text, "html.parser")
            except requests.RequestException as e:
                print(f"  Failed: {url[-40:]} — {e}")
                time.sleep(SPEECH_DELAY)
                continue

            # Date
            time_tag = page.find("time")
            date     = _parse_date(
                time_tag.get_text() if time_tag else page.get_text()[:500]
            )

            # Body
            article = (
                page.find("div", {"itemprop": "articleBody"})
                or page.find("div", class_="item-page")
            )
            if not article:
                time.sleep(SPEECH_DELAY)
                continue

            text_raw   = article.get_text(" ")
            text_clean = _normalise(text_raw)
            n_words    = len(text_clean.split())

            if n_words < MIN_WORDS:
                time.sleep(SPEECH_DELAY)
                continue

            # Stop condition
            if date and pd.Timestamp(date) < limit:
                print(f"  Reached {date.date()} — older than cut-off, stopping.")
                stop_flag = True
                break

            records.append({
                "date":     date,
                "url":      url,
                "n_words":  n_words,
                "text_raw": text_raw,
            })
            print(f"  Saved  {date.date() if date else '??-??-??'}  ({n_words} words)  {url[-50:]}")
            time.sleep(SPEECH_DELAY)

        if stop_flag:
            break

        offset += PAGE_SIZE
        time.sleep(PAGE_DELAY)

    if not records:
        print("No speeches collected.")
        return pd.DataFrame()

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", ascending=False, inplace=True, ignore_index=True)

    # Stable speech ID (0-based after sort)
    df.insert(0, "speech_id", range(len(df)))

    # President labels
    pres_info        = df["date"].apply(_get_president)
    df["president"]  = pres_info.apply(lambda x: x[0])
    df["president_id"] = pres_info.apply(lambda x: x[1])

    # Time helpers
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # Reorder columns for readability
    df = df[[
        "speech_id", "date", "url",
        "president", "president_id",
        "year", "month", "year_month",
        "n_words", "text_raw",
    ]]

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = scrape(STOP_DATE_STR)

    if df.empty:
        print("No data collected — check connectivity and Casa Rosada URL.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n{len(df)} speeches saved to {OUTPUT_CSV}")
        print("\nBreakdown by president:")
        print(df["president"].value_counts().to_string())
        print("\nDate range:")
        print(f"  Earliest: {df['date'].min().date()}")
        print(f"  Latest:   {df['date'].max().date()}")
