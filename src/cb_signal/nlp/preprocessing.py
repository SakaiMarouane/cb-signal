"""Text preprocessing for central bank communications.

Central bank speeches and statements share a few stylistic quirks that a
generic tokenizer handles poorly:

* Numeric expressions like `2.5 per cent`, `1/4 percentage point`.
* Section headers glued to the following paragraph after HTML flattening.
* Footnote markers `[1]`, `*`, that pollute sentence boundaries.
* Abbreviations `Dr.`, `Mr.`, `U.S.`, `p.p.`, `e.g.`, `i.e.` that trip
  naive sentence splitters.

We keep dependencies minimal (regex only, no spaCy in the dictionary
pipeline) so this module is cheap to run over the full corpus and safe
to import without model downloads.
"""

from __future__ import annotations

import re
import unicodedata

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "u.s", "u.k", "e.g", "i.e", "etc", "vs", "no",
    "fig", "eq", "ch", "sec", "vol", "pp", "p.p",
    "ecb", "fed", "boe", "fomc", "mpc", "imf", "oecd",
}

_FOOTNOTE_RE = re.compile(r"\[\d+\]|\(\d+\)|\bfootnote\s*\d+\b", re.IGNORECASE)
_MULTI_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")

_SENT_SPLIT_RE = re.compile(
    r"""
    (?<=[.!?])       # split after a sentence-terminal punctuation...
    (?=\s+[A-Z"'“])  # ...only when the next non-space starts a capital or an opening quote
    """,
    re.VERBOSE,
)


def normalise_unicode(text: str) -> str:
    """Strip diacritics-preserving NFC normalisation and unify quotes/dashes.

    We keep accents (ECB speeches contain them), but collapse curly quotes and
    dashes so downstream regex matching is stable.
    """
    text = unicodedata.normalize("NFC", text)
    trans = str.maketrans(
        {
            "‘": "'", "’": "'",
            "“": '"', "”": '"',
            "–": "-", "—": "-",
            " ": " ",
        }
    )
    return text.translate(trans)


def clean_document(text: str) -> str:
    """Base cleaning applied before sentence splitting.

    Removes URLs and footnote markers, collapses whitespace. Case is preserved
    at this stage; lowercasing happens inside the scorer so proper nouns can
    still be inspected downstream.
    """
    text = normalise_unicode(text)
    text = _URL_RE.sub(" ", text)
    text = _FOOTNOTE_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text)
    return text.strip()


def _looks_like_abbreviation(fragment: str) -> bool:
    """Return True if `fragment` ends with a known abbreviation.

    Called to reject false sentence boundaries introduced by tokens like `U.S.`
    when the next character happens to be uppercase.
    """
    tail = fragment.rsplit(" ", 1)[-1].rstrip(".").lower()
    return tail in _ABBREVIATIONS


def sentence_split(text: str, min_chars: int = 20) -> list[str]:
    """Split `text` into sentences using a conservative regex + abbreviation guard.

    Sentences shorter than `min_chars` are dropped: they carry no reliable
    sentiment signal (bullet stubs, section headers, page numbers).
    """
    if not text:
        return []
    cleaned = clean_document(text)

    # First pass: candidate splits.
    raw_parts = _SENT_SPLIT_RE.split(cleaned)

    # Second pass: glue back parts whose left side ends with an abbreviation.
    merged: list[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if merged and _looks_like_abbreviation(merged[-1]):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    return [s for s in merged if len(s) >= min_chars]


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")


def word_tokens(sentence: str) -> list[str]:
    """Return the lowercase word tokens of `sentence`, ignoring numbers and punctuation."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(sentence)]
