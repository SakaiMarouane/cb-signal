from __future__ import annotations

import datetime as dt

from cb_signal.ingest.boe import _parse_page


def test_boe_parse_page_minutes():
    body = " ".join(["policy"] * 150)
    html = (
        f"<html><head>"
        f"<meta name='published-date' content='2024-05-09T12:00:00Z'>"
        f"</head><body>"
        f"<h1>Monetary Policy Summary and minutes - May 2024</h1>"
        f"<div class='page-content'><p>{body}</p></div>"
        f"</body></html>"
    ).encode()
    url = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2024/may-2024"
    row = _parse_page(html, url)
    assert row is not None
    assert row["source"] == "BOE"
    assert row["event_type"] == "minutes"
    assert row["date"] == dt.date(2024, 5, 9)
    assert row["speaker"] == "MPC"
    assert "May 2024" in row["title"]


def test_boe_parse_page_speech_with_time_tag():
    body = " ".join(["inflation"] * 150)
    html = (
        f"<html><body>"
        f"<h1>The economic outlook - speech by Andrew Bailey</h1>"
        f"<time datetime='2023-11-15T09:00:00Z'>15 November 2023</time>"
        f"<div class='page-content'><p>{body}</p></div>"
        f"</body></html>"
    ).encode()
    url = "https://www.bankofengland.co.uk/speech/2023/november/andrew-bailey"
    row = _parse_page(html, url)
    assert row is not None
    assert row["source"] == "BOE"
    assert row["event_type"] == "speech"
    assert row["date"] == dt.date(2023, 11, 15)
    assert "Bailey" in row["speaker"]


def test_boe_parse_page_returns_none_on_short_body():
    html = b"""
    <html><body>
      <h1>Stub</h1>
      <time datetime='2020-01-01T00:00:00Z'>1 January 2020</time>
      <div class='page-content'><p>Too short.</p></div>
    </body></html>
    """
    row = _parse_page(html, "https://www.bankofengland.co.uk/speech/2020/january/stub")
    assert row is None
