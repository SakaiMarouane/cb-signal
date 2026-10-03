"""Ingest Federal Reserve communications: FOMC statements, FOMC minutes, speeches.

Sources and URL patterns
------------------------
* Current calendar page listing recent meetings and future ones:
    https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
* Historical calendar page per year (5+ years old):
    https://www.federalreserve.gov/monetarypolicy/fomchistorical{YYYY}.htm
* FOMC statement per meeting date:
    https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm
* FOMC minutes per meeting date (released ~3 weeks later):
    https://www.federalreserve.gov/monetarypolicy/fomcminutes{YYYYMMDD}.htm
* Speech archive for a given year (older format):
    https://www.federalreserve.gov/newsevents/speech/{YYYY}-speeches.htm
  and (newer format):
    https://www.federalreserve.gov/newsevents/speech/{YYYY}speech.htm

The Fed website occasionally reorganises URLs. Every fetch is wrapped in
`polite_get` which retries and caches, and a missing page is logged and
skipped rather than aborting the whole run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

from cb_signal.ingest._http import polite_get, project_cache_dir
from cb_signal.ingest.schema import CANONICAL_COLUMNS

FED_BASE = "https://www.federalreserve.gov"
FED_CALENDARS_URL = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"
FED_HISTORICAL_URL_TMPL = f"{FED_BASE}/monetarypolicy/fomchistorical{{year}}.htm"
FED_STATEMENT_URL_TMPL = f"{FED_BASE}/newsevents/pressreleases/monetary{{ymd}}a.htm"
FED_MINUTES_URL_TMPL = f"{FED_BASE}/monetarypolicy/fomcminutes{{ymd}}.htm"
FED_SPEECH_ARCHIVE_URL_TMPLS = [
    f"{FED_BASE}/newsevents/speech/{{year}}-speeches.htm",
    f"{FED_BASE}/newsevents/speech/{{year}}speech.htm",
]


@dataclass(frozen=True)
class FomcMeeting:
    date: dt.date
    is_scheduled: bool  # True for regular meetings, False for unscheduled/conference calls


def _stable_doc_id(source: str, date_str: str, key: str) -> str:
    h = hashlib.sha1(f"{source}|{date_str}|{key}".encode()).hexdigest()
    return h[:16]


def _soup(raw: bytes) -> BeautifulSoup:
    return BeautifulSoup(raw, "lxml")


def _clean_text(node: BeautifulSoup | None) -> str:
    if node is None:
        return ""
    for tag in node.select("script, style, nav, footer, header, aside, .footnotes"):
        tag.decompose()
    text = node.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------------------
# Meeting date discovery
#
# Both the historical per-year pages and the current combined calendar page
# list *meetings* as "Month D-D" (no comma before the year) — e.g.
# "January 29-30 Meeting - 2019" on the historical pages, or "January 27-28"
# grouped under a "2026 FOMC Meetings" heading on the current page. Both
# pages *also* print each meeting's minutes-release date in a "(Released
# Month D, YYYY)" aside, which uses the "Month D, YYYY" comma format. An
# earlier version of this regex matched on the comma format and so picked up
# the minutes-release dates instead of the actual meeting dates throughout —
# every "statement" and "minutes" fetch ended up using a date ~3 weeks after
# the real meeting, silently mislabelling the minutes-release-day press
# release as the statement. We strip the "(Released ...)" asides before
# matching, specifically to avoid that trap.
# --------------------------------------------------------------------------------------

_RELEASED_ASIDE_RE = re.compile(r"\(Released[^)]*\)", re.IGNORECASE)
_MONTH_ABBREVIATIONS = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December",
}
# Cross-month meetings ("Jan-Feb", "Oct-Nov") are spelled out in full on the
# historical per-year pages ("April/May") but abbreviated to 3 letters on the
# current combined calendar page ("Apr/May"); accept both.
_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|" + "|".join(_MONTH_ABBREVIATIONS)
)


def _full_month_name(token: str) -> str:
    return _MONTH_ABBREVIATIONS.get(token, token)
# Historical per-year pages: "January 29-30 Meeting - 2019" / "October 4
# (unscheduled) - 2019" / "April/May 30-1 Meeting - 2019".
_HISTORICAL_MEETING_RE = re.compile(
    rf"(?P<m1>{_MONTH_NAMES})(?:/(?P<m2>{_MONTH_NAMES}))?"
    r"\s+(?P<d1>\d{1,2})(?:-(?P<d2>\d{1,2}))?"
    r"\s*(?:\(unscheduled\)\s*)?(?:Meeting\s*)?-\s*(?P<year>\d{4})",
    re.IGNORECASE,
)
# Current calendar page: a "YYYY FOMC Meetings" heading followed by entries
# like "January 27-28" or "April/May 30-1" with no year of their own.
_YEAR_HEADING_RE = re.compile(r"(\d{4})\s+FOMC Meetings", re.IGNORECASE)
_CALENDAR_MEETING_RE = re.compile(
    rf"(?P<m1>{_MONTH_NAMES})(?:/(?P<m2>{_MONTH_NAMES}))?"
    r"\s+(?P<d1>\d{1,2})(?:-(?P<d2>\d{1,2}))?\*?",
    re.IGNORECASE,
)


def _meeting_end_date(m1: str, m2: str | None, d1: str, d2: str | None, year: int) -> dt.date | None:
    """Resolve a meeting's last day, honouring a month change (e.g. April/May)."""
    end_month = _full_month_name(m2 or m1)
    end_day = int(d2) if d2 else int(d1)
    try:
        return dt.datetime.strptime(f"{end_month} {end_day} {year}", "%B %d %Y").date()
    except ValueError:
        return None


def _parse_meeting_dates_from_page(html: bytes) -> list[dt.date]:
    """Extract real meeting dates (not minutes-release dates) from a FOMC
    calendar or historical page, taking the last day of each meeting."""
    soup = _soup(html)
    text = _RELEASED_ASIDE_RE.sub("", soup.get_text(" "))
    dates: set[dt.date] = set()

    for m in _HISTORICAL_MEETING_RE.finditer(text):
        d = _meeting_end_date(m.group("m1"), m.group("m2"), m.group("d1"), m.group("d2"), int(m.group("year")))
        if d is not None:
            dates.add(d)

    headings = list(_YEAR_HEADING_RE.finditer(text))
    for i, hm in enumerate(headings):
        year = int(hm.group(1))
        segment_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        segment = text[hm.end() : segment_end]
        for m in _CALENDAR_MEETING_RE.finditer(segment):
            d = _meeting_end_date(m.group("m1"), m.group("m2"), m.group("d1"), m.group("d2"), year)
            if d is not None:
                dates.add(d)

    return sorted(dates)


def discover_fomc_meetings(root: Path, start_year: int = 1994) -> list[FomcMeeting]:
    """List FOMC meeting dates by aggregating calendar and historical pages."""
    current_year = dt.date.today().year
    cache = project_cache_dir(root)
    all_dates: set[dt.date] = set()

    # Historical years
    for year in range(start_year, current_year - 4):
        url = FED_HISTORICAL_URL_TMPL.format(year=year)
        try:
            html = polite_get(url, cache_dir=cache)
        except Exception as e:
            logger.warning(f"[FED] historical calendar {year} unavailable: {e}")
            continue
        found = _parse_meeting_dates_from_page(html)
        # historical pages sometimes list years around the target year; filter
        all_dates.update(d for d in found if d.year == year)

    # Recent + current + upcoming
    try:
        html = polite_get(FED_CALENDARS_URL, cache_dir=cache, force=True)
        all_dates.update(_parse_meeting_dates_from_page(html))
    except Exception as e:
        logger.warning(f"[FED] recent calendar unavailable: {e}")

    # Only keep past meetings for scraping (statements exist only after the meeting)
    today = dt.date.today()
    meetings = [FomcMeeting(d, is_scheduled=True) for d in sorted(all_dates) if d <= today]
    logger.info(f"[FED] Discovered {len(meetings)} FOMC meeting dates")
    return meetings


# --------------------------------------------------------------------------------------
# Statement parsing
# --------------------------------------------------------------------------------------


def _extract_press_release_body(html: bytes) -> tuple[str, str]:
    """Return (title, body) from a Federal Reserve press release page."""
    soup = _soup(html)
    title_node = soup.find("h3", class_="title") or soup.find("h1")
    title = title_node.get_text(strip=True) if title_node else ""

    # Modern press releases wrap the body in <div id="article">.
    body_node = soup.find("div", id="article") or soup.find("div", class_="col-xs-12 col-md-8")
    if body_node is None:
        body_node = soup.find("div", role="main") or soup
    text = _clean_text(body_node)
    return title, text


def fetch_statement(root: Path, meeting_date: dt.date) -> dict | None:
    """Fetch and parse the FOMC statement for a given meeting date.

    Returns a dict in canonical schema, or None if the page is missing.
    """
    ymd = meeting_date.strftime("%Y%m%d")
    url = FED_STATEMENT_URL_TMPL.format(ymd=ymd)
    try:
        html = polite_get(url, cache_dir=project_cache_dir(root))
    except Exception as e:
        logger.debug(f"[FED] statement {ymd} missing: {e}")
        return None

    title, text = _extract_press_release_body(html)
    if len(text.split()) < 50:
        return None
    return {
        "doc_id": _stable_doc_id("FED", str(meeting_date), f"statement|{ymd}"),
        "source": "FED",
        "event_type": "statement",
        "date": meeting_date,
        "speaker": "FOMC",
        "title": title or f"FOMC statement {meeting_date.isoformat()}",
        "subtitle": None,
        "url": url,
        "text": text,
        "n_tokens": len(text.split()),
    }


# --------------------------------------------------------------------------------------
# Minutes parsing
# --------------------------------------------------------------------------------------


def fetch_minutes(root: Path, meeting_date: dt.date) -> dict | None:
    ymd = meeting_date.strftime("%Y%m%d")
    url = FED_MINUTES_URL_TMPL.format(ymd=ymd)
    try:
        html = polite_get(url, cache_dir=project_cache_dir(root))
    except Exception as e:
        logger.debug(f"[FED] minutes {ymd} missing: {e}")
        return None

    title, text = _extract_press_release_body(html)
    if len(text.split()) < 200:  # minutes are always long
        return None
    return {
        "doc_id": _stable_doc_id("FED", str(meeting_date), f"minutes|{ymd}"),
        "source": "FED",
        "event_type": "minutes",
        "date": meeting_date,
        "speaker": "FOMC",
        "title": title or f"FOMC minutes {meeting_date.isoformat()}",
        "subtitle": None,
        "url": url,
        "text": text,
        "n_tokens": len(text.split()),
    }


# --------------------------------------------------------------------------------------
# Speech discovery + parsing
# --------------------------------------------------------------------------------------

_SPEECH_LINK_RE = re.compile(r"/newsevents/speech/[a-z\-]+\d{8}[a-z]?\.htm", re.IGNORECASE)


def _discover_speech_urls_for_year(root: Path, year: int) -> list[str]:
    cache = project_cache_dir(root)
    urls: set[str] = set()
    for tmpl in FED_SPEECH_ARCHIVE_URL_TMPLS:
        try:
            html = polite_get(tmpl.format(year=year), cache_dir=cache)
        except Exception:
            continue
        soup = _soup(html)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if _SPEECH_LINK_RE.search(href):
                urls.add(FED_BASE + href if href.startswith("/") else href)
        if urls:
            break  # first working template wins
    return sorted(urls)


_SPEECH_DATE_RE = re.compile(r"(\d{8})[a-z]?\.htm$", re.IGNORECASE)


def _parse_speech_page(html: bytes, url: str) -> dict | None:
    soup = _soup(html)

    # date is encoded in the URL; also present in the page header as fallback
    m = _SPEECH_DATE_RE.search(url)
    if not m:
        return None
    try:
        date = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None

    title_node = soup.find("h3", class_="title") or soup.find("h1")
    title = title_node.get_text(strip=True) if title_node else ""

    speaker_node = soup.find("p", class_="speaker") or soup.find("p", class_="article__subtitle")
    speaker = speaker_node.get_text(strip=True) if speaker_node else ""

    body_node = soup.find("div", id="article") or soup.find("div", class_="col-xs-12 col-md-8")
    text = _clean_text(body_node)
    if len(text.split()) < 100:
        return None

    return {
        "doc_id": _stable_doc_id("FED", str(date), f"speech|{url}"),
        "source": "FED",
        "event_type": "speech",
        "date": date,
        "speaker": speaker or "Fed board member",
        "title": title,
        "subtitle": None,
        "url": url,
        "text": text,
        "n_tokens": len(text.split()),
    }


def fetch_speeches(root: Path, start_year: int = 2006) -> list[dict]:
    """Fetch Fed speeches from `start_year` onward.

    Pre-2006 archives use a very different layout and add limited signal for a
    monetary-policy sentiment study; we default to modern years.
    """
    current_year = dt.date.today().year
    cache = project_cache_dir(root)
    speeches: list[dict] = []
    for year in range(start_year, current_year + 1):
        urls = _discover_speech_urls_for_year(root, year)
        logger.info(f"[FED] {year}: {len(urls)} speech URLs discovered")
        for url in urls:
            try:
                html = polite_get(url, cache_dir=cache)
            except Exception as e:
                logger.warning(f"[FED] speech {url} failed: {e}")
                continue
            row = _parse_speech_page(html, url)
            if row is not None:
                speeches.append(row)
    return speeches


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def fetch_fed_corpus(
    root: Path,
    *,
    include_speeches: bool = True,
    speech_start_year: int = 2006,
) -> pd.DataFrame:
    """Full Fed pipeline: statements + minutes + optional speeches -> parquet."""
    output_dir = root / "data" / "raw" / "fed"
    output_dir.mkdir(parents=True, exist_ok=True)

    meetings = discover_fomc_meetings(root)

    rows: list[dict] = []
    for meeting in meetings:
        stmt = fetch_statement(root, meeting.date)
        if stmt is not None:
            rows.append(stmt)
        minutes = fetch_minutes(root, meeting.date)
        if minutes is not None:
            rows.append(minutes)
    logger.info(f"[FED] {sum(1 for r in rows if r['event_type']=='statement')} statements, "
                f"{sum(1 for r in rows if r['event_type']=='minutes')} minutes")

    if include_speeches:
        rows.extend(fetch_speeches(root, start_year=speech_start_year))
        logger.info(f"[FED] {sum(1 for r in rows if r['event_type']=='speech')} speeches")

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    df = df.drop_duplicates(subset=["doc_id"]).sort_values("date").reset_index(drop=True)

    out_path = output_dir / "corpus.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"[FED] Wrote {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    fetch_fed_corpus(root)
