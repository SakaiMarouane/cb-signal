"""Ingest Bank of England communications: MPC minutes / summaries and speeches.

BoE's publication listing pages (`/monetary-policy-summary-and-minutes`,
`/news/speeches`) are now rendered client-side by a Cludo search widget, so
plain HTTP GETs no longer see any document links in the response HTML. We
discover documents instead via BoE's own sitemap feed, which is server
rendered and lists every URL on the site:

    https://www.bankofengland.co.uk/_api/sitemap/getsitemap

Individual document pages still carry the publication date and speaker in
structured metadata; body text is inside `<div class="page-content">` /
similar wrappers, and that part of the site has not changed.

We filter the sitemap on URL patterns:

* `/monetary-policy-summary-and-minutes/{yyyy}/...` -> MPC minutes/summary
* `/speech/{yyyy}/...` -> speech

BoE HTML changes periodically; the parser extracts text conservatively.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

from cb_signal.ingest._http import polite_get, project_cache_dir
from cb_signal.ingest.schema import CANONICAL_COLUMNS

BOE_BASE = "https://www.bankofengland.co.uk"
BOE_SITEMAP_URL = f"{BOE_BASE}/_api/sitemap/getsitemap"

_MPC_HREF_RE = re.compile(r"/monetary-policy-summary-and-minutes/\d{4}/", re.IGNORECASE)
_SPEECH_HREF_RE = re.compile(r"/speech/\d{4}/", re.IGNORECASE)
_SITEMAP_LOC_RE = re.compile(r"<loc>([^<]+)</loc>")


def _stable_doc_id(source: str, date_str: str, url: str) -> str:
    return hashlib.sha1(f"{source}|{date_str}|{url}".encode()).hexdigest()[:16]


def _soup(raw: bytes) -> BeautifulSoup:
    return BeautifulSoup(raw, "lxml")


def _clean_text(node: BeautifulSoup | None) -> str:
    if node is None:
        return ""
    for tag in node.select("script, style, nav, footer, header, aside, .related, .footnotes"):
        tag.decompose()
    text = node.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def _discover_via_sitemap(root: Path, href_re: re.Pattern[str]) -> list[str]:
    """Fetch BoE's sitemap and return URLs matching `href_re`."""
    cache = project_cache_dir(root)
    raw = polite_get(BOE_SITEMAP_URL, cache_dir=cache)
    urls = _SITEMAP_LOC_RE.findall(raw.decode("utf-8", errors="ignore"))
    matched = sorted({u for u in urls if href_re.search(u)})
    return matched


_BOE_DATE_META_RE = re.compile(r"\d{1,2}\s+\w+\s+\d{4}")
_BOE_SPEAKER_TITLE_RE = re.compile(r"speech\s+by\s+(.+)$", re.IGNORECASE)


def _parse_page(html: bytes, url: str) -> dict | None:
    soup = _soup(html)

    # date: the page's own "Published on <date>" label is authoritative. The
    # `<time>` tags elsewhere on the page belong to "related content" teaser
    # cards, not the article itself, so they are only a last-resort fallback.
    date: dt.date | None = None
    pub_node = soup.find(attrs={"class": re.compile(r"published-date", re.I)})
    if pub_node:
        m = _BOE_DATE_META_RE.search(pub_node.get_text(" ", strip=True))
        if m:
            for fmt in ("%d %B %Y", "%d %b %Y"):
                try:
                    date = dt.datetime.strptime(m.group(0), fmt).date()
                    break
                except ValueError:
                    continue
    if date is None:
        meta_date = soup.find("meta", attrs={"name": "published-date"}) or soup.find(
            "meta", attrs={"property": "article:published_time"}
        )
        if meta_date and meta_date.get("content"):
            with contextlib.suppress(ValueError):
                date = dt.datetime.fromisoformat(meta_date["content"][:10]).date()
    if date is None:
        m = _BOE_DATE_META_RE.search(soup.get_text(" ")[:2000])
        if m:
            for fmt in ("%d %B %Y", "%d %b %Y"):
                try:
                    date = dt.datetime.strptime(m.group(0), fmt).date()
                    break
                except ValueError:
                    continue
    if date is None:
        time_node = soup.find("time")
        if time_node:
            try:
                iso = time_node.get("datetime", "")[:10]
                date = dt.datetime.fromisoformat(iso).date() if iso else None
            except ValueError:
                date = None
    if date is None:
        return None

    title_node = soup.find("h1")
    title = title_node.get_text(strip=True) if title_node else ""

    # BoE speech titles read "<title> - speech by <name>"; that is far more
    # reliable than any single page element, whose class name may collide
    # with unrelated chart captions elsewhere on the page.
    speaker = ""
    m = _BOE_SPEAKER_TITLE_RE.search(title)
    if m:
        speaker = m.group(1).strip()

    body_node = (
        soup.find("div", class_="page-content")
        or soup.find("article")
        or soup.find("main")
    )
    text = _clean_text(body_node)
    if len(text.split()) < 100:
        return None

    if "/monetary-policy-summary-and-minutes/" in url:
        event_type = "minutes"
        default_speaker = "MPC"
    else:
        event_type = "speech"
        default_speaker = "BoE speaker"

    return {
        "doc_id": _stable_doc_id("BOE", str(date), url),
        "source": "BOE",
        "event_type": event_type,
        "date": date,
        "speaker": speaker or default_speaker,
        "title": title,
        "subtitle": None,
        "url": url,
        "text": text,
        "n_tokens": len(text.split()),
    }


def fetch_boe_corpus(root: Path, *, include_speeches: bool = True) -> pd.DataFrame:
    output_dir = root / "data" / "raw" / "boe"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = project_cache_dir(root)

    urls: list[str] = _discover_via_sitemap(root, _MPC_HREF_RE)
    logger.info(f"[BOE] Discovered {len(urls)} MPC pages")
    if include_speeches:
        speech_urls = _discover_via_sitemap(root, _SPEECH_HREF_RE)
        logger.info(f"[BOE] Discovered {len(speech_urls)} speech pages")
        urls.extend(speech_urls)

    rows: list[dict] = []
    for url in urls:
        try:
            html = polite_get(url, cache_dir=cache)
        except Exception as e:
            logger.warning(f"[BOE] {url} failed: {e}")
            continue
        row = _parse_page(html, url)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    df = df.drop_duplicates(subset=["doc_id"]).sort_values("date").reset_index(drop=True)

    out_path = output_dir / "corpus.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"[BOE] Wrote {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    fetch_boe_corpus(root)
