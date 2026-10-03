"""Dictionary-based hawkish / dovish scoring.

Given a document, we split it into sentences, match every hawkish/dovish
phrase in the lexicon against sliding n-grams, and flip polarity when a
negator sits within `NEGATION_WINDOW` tokens *before* the match.

The document-level output is a small tabular record:

    doc_id, n_sentences, n_tokens,
    hawk_hits, dove_hits, uncertainty_hits,
    hawk_share, dove_share, uncertainty_share,
    net_tone, net_tone_norm

Two conventions used throughout the literature:

* `net_tone = (H - D) / (H + D + eps)` — bounded in [-1, 1], stable when the
  document is short. Reported in Apel-Blix Grimaldi and follow-ups.
* `net_tone_norm = (H - D) / n_tokens * 1000` — scale-free "hits per 1000
  tokens", useful for cross-document regression when doc length varies.

Both are produced so downstream signal construction can pick whichever fits.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from cb_signal.nlp.lexicons import NEGATION_WINDOW, NEGATORS, Lexicon
from cb_signal.nlp.preprocessing import sentence_split, word_tokens


@dataclass(frozen=True)
class SentenceScore:
    """Per-sentence match summary; kept lightweight for aggregation."""
    n_tokens: int
    hawk: int
    dove: int
    uncertainty: int


def _phrase_hits(
    tokens: list[str],
    phrases: frozenset[str],
    max_n: int,
) -> list[tuple[int, str]]:
    """Return `(start_index, phrase)` for every lexicon phrase matched.

    The scan walks left-to-right and, for each position, tests n-grams of
    lengths 1..max_n. When multiple lengths match at the same position, the
    longest wins (e.g. "upside risk to inflation" beats "upside risk"), and
    the scan advances past the matched span to avoid double-counting.
    """
    hits: list[tuple[int, str]] = []
    n = len(tokens)
    i = 0
    while i < n:
        matched_len = 0
        matched_phrase: str | None = None
        # Try longest first so we consume the maximal match.
        for L in range(min(max_n, n - i), 0, -1):
            candidate = " ".join(tokens[i : i + L])
            if candidate in phrases:
                matched_len = L
                matched_phrase = candidate
                break
        if matched_phrase is not None:
            hits.append((i, matched_phrase))
            i += matched_len
        else:
            i += 1
    return hits


def _is_negated(tokens: list[str], phrase_start: int) -> bool:
    """True if a negator sits within NEGATION_WINDOW tokens before `phrase_start`."""
    lo = max(0, phrase_start - NEGATION_WINDOW)
    return any(t in NEGATORS for t in tokens[lo:phrase_start])


def score_sentence(sentence: str, lex: Lexicon) -> SentenceScore:
    """Score a single sentence: count hawk, dove, uncertainty terms with negation flip."""
    tokens = word_tokens(sentence)
    if not tokens:
        return SentenceScore(0, 0, 0, 0)

    hawk_hits = _phrase_hits(tokens, lex.hawkish, lex.max_ngram)
    dove_hits = _phrase_hits(tokens, lex.dovish, lex.max_ngram)

    hawk = 0
    dove = 0
    for pos, _phrase in hawk_hits:
        if _is_negated(tokens, pos):
            dove += 1  # "not hawkish" reads as dovish
        else:
            hawk += 1
    for pos, _phrase in dove_hits:
        if _is_negated(tokens, pos):
            hawk += 1
        else:
            dove += 1

    uncertainty = sum(1 for t in tokens if t in lex.uncertainty)

    return SentenceScore(
        n_tokens=len(tokens),
        hawk=hawk,
        dove=dove,
        uncertainty=uncertainty,
    )


def score_document(text: str, lex: Lexicon) -> dict:
    """Aggregate per-sentence scores into the document-level record."""
    sentences = sentence_split(text)
    per_sent = [score_sentence(s, lex) for s in sentences]

    n_sent = len(per_sent)
    n_tokens = sum(s.n_tokens for s in per_sent)
    hawk = sum(s.hawk for s in per_sent)
    dove = sum(s.dove for s in per_sent)
    uncertainty = sum(s.uncertainty for s in per_sent)

    eps = 1e-9
    net_tone = (hawk - dove) / (hawk + dove + eps)
    net_tone_norm = (hawk - dove) / max(n_tokens, 1) * 1000.0

    return {
        "n_sentences": n_sent,
        "n_tokens_scored": n_tokens,
        "hawk_hits": hawk,
        "dove_hits": dove,
        "uncertainty_hits": uncertainty,
        "hawk_share": hawk / max(n_tokens, 1),
        "dove_share": dove / max(n_tokens, 1),
        "uncertainty_share": uncertainty / max(n_tokens, 1),
        "net_tone": net_tone,
        "net_tone_norm": net_tone_norm,
    }


def score_corpus(corpus: pd.DataFrame, lex: Lexicon) -> pd.DataFrame:
    """Score every row of the consolidated corpus and return a joined frame.

    Input columns required: `doc_id`, `source`, `event_type`, `date`, `text`.
    Output columns: the input identifiers + all fields from `score_document`.
    """
    required = {"doc_id", "source", "event_type", "date", "text"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"corpus missing required columns: {missing}")

    logger.info(f"[NLP] scoring {len(corpus)} documents")
    records: list[dict] = []
    for row in corpus.itertuples(index=False):
        scores = score_document(row.text, lex)
        records.append(
            {
                "doc_id": row.doc_id,
                "source": row.source,
                "event_type": row.event_type,
                "date": row.date,
                **scores,
            }
        )
    out = pd.DataFrame.from_records(records)
    logger.info(
        f"[NLP] scored {len(out)} docs; "
        f"mean net_tone={out['net_tone'].mean():+.3f}, "
        f"mean hawk_share={out['hawk_share'].mean():.4f}, "
        f"mean dove_share={out['dove_share'].mean():.4f}"
    )
    return out
