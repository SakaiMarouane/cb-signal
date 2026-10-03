"""Ingest ECB speeches.

ECB used to publish a single pipe-delimited CSV containing every speech
(`/press/key/date/html/all_speeches.en.csv`); that feed is now dead — the ECB
site redesign replaced the browsable speech listing with a client-side search
widget (AddSearch) and dropped the bulk export entirely.

Individual speech pages are still served as plain server-rendered HTML at
stable permalinks (e.g. `/press/key/date/2023/html/ecb.sp230516~....en.html`),
so we discover them through AddSearch's public search API directly — the same
API the widget calls from the browser, keyed by ECB's site key that ships in
their bundled JS — and then fetch + parse each page.

This discovery is full-text search rather than an exhaustive index, so recall
is not guaranteed to be 100%: it should find the large majority of speeches
(anything whose page contains the word "speech", which in practice is nearly
all of them, since the subtitle reads "Speech by ...").
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

from cb_signal.ingest._http import polite_get, project_cache_dir
from cb_signal.ingest.schema import CANONICAL_COLUMNS

ECB_SEARCH_API = "https://api.addsearch.com/v1/search/61893af990d2673c4a92b492dd7f6631"
ECB_SEARCH_TERM = "speech"
ECB_SEARCH_PAGE_SIZE = 100

_SPEECH_URL_RE = re.compile(r"/press/key/date/", re.IGNORECASE)
_SPEAKER_RE = re.compile(r"\bby\s+([A-ZÀ-Ýa-zà-ÿ.'\- ]+?),")


def _stable_doc_id(source: str, date: str, title: str) -> str:
    key = f"{source}|{date}|{title}".encode()
    return hashlib.sha1(key).hexdigest()[:16]


def _extract_speaker(subtitle: str) -> str:
    """Best-effort speaker name from a subtitle like 'Speech by X, Y, at Z'."""
    m = _SPEAKER_RE.search(subtitle)
    return m.group(1).strip() if m else ""


def _discover_via_search(root: Path) -> list[str]:
    """Page through AddSearch results for ECB_SEARCH_TERM, keep speech URLs."""
    cache = project_cache_dir(root)
    urls: set[str] = set()
    page = 1
    while True:
        qs = urlencode(
            {"term": ECB_SEARCH_TERM, "page": page, "lang": "en", "limit": ECB_SEARCH_PAGE_SIZE}
        )
        try:
            raw = polite_get(f"{ECB_SEARCH_API}?{qs}", cache_dir=cache)
        except Exception as e:
            logger.warning(f"[ECB] search page {page} failed: {e}")
            break
        data = json.loads(raw)
        hits = data.get("hits", [])
        if not hits:
            break
        for hit in hits:
            url = hit.get("url", "").replace(".eu//press", ".eu/press")
            if _SPEECH_URL_RE.search(url):
                urls.add(url)
        total_hits = data.get("total_hits", 0)
        if page * ECB_SEARCH_PAGE_SIZE >= total_hits:
            break
        page += 1
    return sorted(urls)


def _parse_speech_page(html: bytes, url: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    main = soup.find("main") or soup

    title_node = main.select_one("div.title h1")
    title = title_node.get_text(strip=True) if title_node else ""

    subtitle_node = main.select_one("h2.ecb-pressContentSubtitle")
    subtitle = subtitle_node.get_text(strip=True) if subtitle_node else ""

    meta_date = soup.select_one('meta[property="article:published_time"]')
    date = meta_date["content"][:10] if meta_date and meta_date.get("content") else None
    if not date:
        return None

    title_div = main.select_one("div.title")
    content_div = title_div.find_next_sibling("div") if title_div else None
    if content_div is None:
        return None
    paragraphs = content_div.find_all("p", class_=lambda c: c != "ecb-publicationDate")
    text = "\n\n".join(p.get_text(strip=True) for p in paragraphs).strip()

    speaker = _extract_speaker(subtitle) or _extract_speaker(title)
    n_tokens = len(text.split())
    if n_tokens < 50:
        return None

    return {
        "doc_id": _stable_doc_id("ECB", date, title),
        "source": "ECB",
        "event_type": "speech",
        "date": date,
        "speaker": speaker,
        "title": title,
        "subtitle": subtitle or None,
        "url": url,
        "text": text,
        "n_tokens": n_tokens,
    }


def fetch_ecb_speeches(root: Path) -> pd.DataFrame:
    """Discover, fetch, parse and persist the ECB speeches corpus."""
    output_path = root / "data" / "raw" / "ecb" / "speeches.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = project_cache_dir(root)

    urls = _discover_via_search(root)
    logger.info(f"[ECB] Discovered {len(urls)} speech pages")

    rows: list[dict] = []
    for url in urls:
        try:
            html = polite_get(url, cache_dir=cache)
        except Exception as e:
            logger.warning(f"[ECB] {url} failed: {e}")
            continue
        row = _parse_speech_page(html, url)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop_duplicates(subset=["doc_id"]).sort_values("date").reset_index(drop=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"[ECB] Wrote {len(df)} rows to {output_path}")
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    fetch_ecb_speeches(root)
