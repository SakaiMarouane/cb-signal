from __future__ import annotations

from cb_signal.ingest.ecb import (
    CANONICAL_COLUMNS,
    _extract_speaker,
    _parse_speech_page,
    _stable_doc_id,
)


def test_stable_doc_id_is_deterministic():
    a = _stable_doc_id("ECB", "2024-01-01", "Sample")
    b = _stable_doc_id("ECB", "2024-01-01", "Sample")
    assert a == b
    assert len(a) == 16


def test_stable_doc_id_differs_on_input_change():
    a = _stable_doc_id("ECB", "2024-01-01", "Sample")
    b = _stable_doc_id("ECB", "2024-01-02", "Sample")
    assert a != b


def test_extract_speaker_from_subtitle():
    assert (
        _extract_speaker("Speech by Isabel Schnabel, Member of the Executive Board, at X")
        == "Isabel Schnabel"
    )


def test_extract_speaker_returns_empty_when_no_match():
    assert _extract_speaker("Just a title with no attribution") == ""


def _sample_html(body_paragraphs: int = 60) -> bytes:
    paras = "".join(f"<p>{'word ' * 20}</p>" for _ in range(body_paragraphs))
    return (
        "<html><head>"
        '<meta property="article:published_time" content="2024-01-15">'
        "</head><body><main>"
        '<div class="title"><ul><li>SPEECH</li></ul><h1>Sample speech title</h1></div>'
        '<div class="section">'
        '<h2 class="ecb-pressContentSubtitle">Speech by Christine Lagarde, President of the ECB, '
        "at a sample venue on 15 January 2024</h2>"
        '<p class="ecb-publicationDate">Frankfurt, 15 January 2024</p>'
        f"{paras}"
        "</div>"
        "</main></body></html>"
    ).encode()


def test_parse_speech_page_extracts_expected_fields():
    row = _parse_speech_page(_sample_html(), "https://www.ecb.europa.eu/press/key/date/2024/html/x.en.html")
    assert row is not None
    assert list(row.keys()) == CANONICAL_COLUMNS
    assert row["source"] == "ECB"
    assert row["event_type"] == "speech"
    assert row["date"] == "2024-01-15"
    assert row["speaker"] == "Christine Lagarde"
    assert row["title"] == "Sample speech title"
    assert row["n_tokens"] >= 50


def test_parse_speech_page_returns_none_on_short_body():
    row = _parse_speech_page(
        _sample_html(body_paragraphs=1),
        "https://www.ecb.europa.eu/press/key/date/2024/html/x.en.html",
    )
    assert row is None


def test_parse_speech_page_returns_none_without_date():
    html = (
        "<html><head></head><body><main>"
        '<div class="title"><ul><li>SPEECH</li></ul><h1>No date</h1></div>'
        '<div class="section"><p>' + ("word " * 60) + "</p></div>"
        "</main></body></html>"
    ).encode()
    row = _parse_speech_page(html, "https://www.ecb.europa.eu/press/key/date/2024/html/x.en.html")
    assert row is None
