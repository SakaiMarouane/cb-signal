from __future__ import annotations

import datetime as dt

from cb_signal.ingest.fed import (
    _parse_meeting_dates_from_page,
    _parse_speech_page,
    _extract_press_release_body,
)


def test_parse_meeting_dates_from_historical_page():
    # Real fomchistorical{year}.htm format: no comma before the year, and a
    # "Minutes (Released ...)" aside nearby that must NOT be mistaken for a
    # meeting date.
    html = b"""
    <html><body>
      <h2>2019</h2>
      <p>January 29-30 Meeting - 2019</p>
      <p>Minutes (Released February 20, 2019): HTML</p>
      <p>April/May 30-1 Meeting - 2019</p>
      <p>Minutes (Released May 22, 2019): HTML</p>
      <p>October 4 (unscheduled) - 2019</p>
    </body></html>
    """
    dates = _parse_meeting_dates_from_page(html)
    assert dt.date(2019, 1, 30) in dates
    assert dt.date(2019, 5, 1) in dates  # cross-month meeting ends in May
    assert dt.date(2019, 10, 4) in dates
    # the minutes-release dates must not leak in as meeting dates
    assert dt.date(2019, 2, 20) not in dates
    assert dt.date(2019, 5, 22) not in dates


def test_parse_meeting_dates_from_current_calendar_page():
    # Real fomccalendars.htm format: meetings grouped under a "YYYY FOMC
    # Meetings" heading, with day ranges carrying no year of their own.
    html = b"""
    <html><body>
      <p>2026 FOMC Meetings</p>
      <p>January 27-28 Statement: PDF | HTML Minutes: PDF | HTML (Released February 18, 2026)</p>
      <p>March 17-18* Statement: PDF | HTML Minutes: PDF | HTML (Released April 08, 2026)</p>
      <p>2025 FOMC Meetings</p>
      <p>December 9-10* Statement: PDF | HTML Minutes: PDF | HTML (Released December 30, 2025)</p>
    </body></html>
    """
    dates = _parse_meeting_dates_from_page(html)
    assert dt.date(2026, 1, 28) in dates
    assert dt.date(2026, 3, 18) in dates
    assert dt.date(2025, 12, 10) in dates
    # the minutes-release dates must not leak in as meeting dates
    assert dt.date(2026, 2, 18) not in dates
    assert dt.date(2026, 4, 8) not in dates


def test_press_release_body_extraction():
    html = b"""
    <html><body>
      <h3 class="title">FOMC statement</h3>
      <div id="article">
        <p>The Committee decided to maintain the target range for the federal funds rate.</p>
        <p>Economic activity has continued to expand at a solid pace.</p>
      </div>
      <script>alert('nope')</script>
    </body></html>
    """
    title, text = _extract_press_release_body(html)
    assert title == "FOMC statement"
    assert "federal funds rate" in text
    assert "alert" not in text  # script removed


def test_speech_page_parser_returns_none_on_short_body():
    html = b"""
    <html><body>
      <h3 class="title">A short talk</h3>
      <div id="article"><p>Too short.</p></div>
    </body></html>
    """
    row = _parse_speech_page(html, "https://www.federalreserve.gov/newsevents/speech/foo20200115a.htm")
    assert row is None


def test_speech_page_parser_extracts_date_from_url():
    body = " ".join(["word"] * 150)
    html = (
        f"<html><body><h3 class='title'>Talk</h3>"
        f"<p class='speaker'>Governor Doe</p>"
        f"<div id='article'><p>{body}</p></div>"
        f"</body></html>"
    ).encode()
    url = "https://www.federalreserve.gov/newsevents/speech/doe20200115a.htm"
    row = _parse_speech_page(html, url)
    assert row is not None
    assert row["date"] == dt.date(2020, 1, 15)
    assert row["source"] == "FED"
    assert row["event_type"] == "speech"
    assert row["speaker"] == "Governor Doe"
    assert row["n_tokens"] >= 100
