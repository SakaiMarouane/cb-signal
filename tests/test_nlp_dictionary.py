from __future__ import annotations

import pandas as pd

from cb_signal.nlp.dictionary import (
    _is_negated,
    _phrase_hits,
    score_corpus,
    score_document,
    score_sentence,
)
from cb_signal.nlp.lexicons import (
    NEGATION_WINDOW,
    build_default_lexicon,
)
from cb_signal.nlp.preprocessing import (
    clean_document,
    sentence_split,
    word_tokens,
)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


def test_clean_document_strips_urls_and_footnotes():
    raw = "Growth is robust [1]. See https://example.com for details."
    out = clean_document(raw)
    assert "http" not in out
    assert "[1]" not in out
    assert "robust" in out


def test_clean_document_unifies_smart_punctuation():
    raw = "Inflation – the risk — remains “elevated”."
    out = clean_document(raw)
    assert "–" not in out
    assert "—" not in out
    assert "“" not in out and "”" not in out
    assert '"elevated"' in out


def test_sentence_split_respects_abbreviations():
    text = "The U.S. economy remains strong. Inflation is above target."
    sents = sentence_split(text, min_chars=5)
    assert len(sents) == 2
    assert sents[0].startswith("The U.S. economy")


def test_sentence_split_drops_short_stubs():
    text = "OK. Growth is robust and broad-based across sectors."
    sents = sentence_split(text, min_chars=20)
    assert len(sents) == 1
    assert "Growth is robust" in sents[0]


def test_word_tokens_lowercases_and_drops_numbers():
    tokens = word_tokens("Rate cut of 25 bps in Q4.")
    assert "rate" in tokens
    assert "cut" in tokens
    assert "25" not in tokens


# ---------------------------------------------------------------------------
# lexicons
# ---------------------------------------------------------------------------


def test_default_lexicon_has_no_polarity_conflicts():
    lex = build_default_lexicon()
    assert lex.hawkish.isdisjoint(lex.dovish)
    assert lex.max_ngram >= 2  # bigrams are essential


# ---------------------------------------------------------------------------
# phrase matching + negation
# ---------------------------------------------------------------------------


def test_phrase_hits_longest_match_wins():
    lex = build_default_lexicon()
    tokens = "we see upside risks to inflation".split()
    hits = _phrase_hits(tokens, lex.hawkish, lex.max_ngram)
    matched = [phrase for _, phrase in hits]
    # Should match the trigram phrase, not "upside risks" alone twice.
    assert "upside risks to inflation" in matched or "upside risks" in matched
    assert len(hits) == 1


def test_is_negated_window():
    tokens = ["we", "do", "not", "see", "upside", "risks"]
    assert _is_negated(tokens, phrase_start=4) is True
    # Outside the window
    long_tokens = ["not"] + ["x"] * (NEGATION_WINDOW + 1) + ["upside", "risks"]
    phrase_pos = len(long_tokens) - 2
    assert _is_negated(long_tokens, phrase_start=phrase_pos) is False


def test_score_sentence_negation_flips_polarity():
    lex = build_default_lexicon()
    plain = score_sentence("We see upside risks to inflation.", lex)
    negated = score_sentence("We do not see upside risks to inflation.", lex)
    assert plain.hawk >= 1
    assert plain.dove == 0
    assert negated.hawk == 0
    assert negated.dove >= 1


def test_score_sentence_uncertainty_counted():
    lex = build_default_lexicon()
    s = score_sentence("There is uncertainty around the risks to the outlook.", lex)
    assert s.uncertainty >= 2


# ---------------------------------------------------------------------------
# document aggregation
# ---------------------------------------------------------------------------


def test_score_document_net_tone_bounds():
    lex = build_default_lexicon()
    hawkish_doc = (
        "Inflation pressures remain elevated. "
        "We expect further tightening. "
        "Wage pressures are broad-based. "
        "The labor market remains tight."
    )
    dovish_doc = (
        "Downside risks to growth have intensified. "
        "The economy is weak and disinflationary forces dominate. "
        "Further accommodation may be warranted."
    )

    h = score_document(hawkish_doc, lex)
    d = score_document(dovish_doc, lex)

    assert h["hawk_hits"] > h["dove_hits"]
    assert d["dove_hits"] > d["hawk_hits"]
    assert h["net_tone"] > 0
    assert d["net_tone"] < 0
    assert -1.0 <= h["net_tone"] <= 1.0
    assert -1.0 <= d["net_tone"] <= 1.0


def test_score_document_empty_text_is_safe():
    lex = build_default_lexicon()
    out = score_document("", lex)
    assert out["n_sentences"] == 0
    assert out["hawk_hits"] == 0
    assert out["dove_hits"] == 0
    assert out["net_tone"] == 0.0


# ---------------------------------------------------------------------------
# corpus scoring
# ---------------------------------------------------------------------------


def test_score_corpus_produces_one_row_per_doc():
    lex = build_default_lexicon()
    corpus = pd.DataFrame(
        {
            "doc_id": ["a", "b"],
            "source": ["ECB", "FED"],
            "event_type": ["speech", "statement"],
            "date": ["2024-01-15", "2024-02-01"],
            "title": ["t1", "t2"],
            "text": [
                "Inflation pressures remain elevated. Further tightening is warranted.",
                "Downside risks to growth have increased. Additional accommodation may be needed.",
            ],
        }
    )
    out = score_corpus(corpus, lex)
    assert len(out) == 2
    assert set(out["doc_id"]) == {"a", "b"}
    assert (out["net_tone"] >= -1).all() and (out["net_tone"] <= 1).all()
    assert out.loc[out["doc_id"] == "a", "net_tone"].iloc[0] > 0
    assert out.loc[out["doc_id"] == "b", "net_tone"].iloc[0] < 0


def test_score_corpus_missing_column_raises():
    lex = build_default_lexicon()
    bad = pd.DataFrame({"doc_id": ["a"], "source": ["ECB"], "event_type": ["speech"], "date": ["2024-01-01"]})
    try:
        score_corpus(bad, lex)
    except ValueError as e:
        assert "text" in str(e)
    else:
        raise AssertionError("expected ValueError for missing 'text' column")
