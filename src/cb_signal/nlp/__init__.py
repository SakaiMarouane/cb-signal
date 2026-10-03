"""Public API for the NLP layer.

Right now only the dictionary-based scorer is exposed. FinBERT and the
fine-tuned sentence-transformer scorers land here as they are built.
"""

from cb_signal.nlp.dictionary import (
    SentenceScore,
    score_corpus,
    score_document,
    score_sentence,
)
from cb_signal.nlp.lexicons import Lexicon, build_default_lexicon, build_lexicon
from cb_signal.nlp.preprocessing import (
    clean_document,
    normalise_unicode,
    sentence_split,
    word_tokens,
)

__all__ = [
    "Lexicon",
    "SentenceScore",
    "build_default_lexicon",
    "build_lexicon",
    "clean_document",
    "normalise_unicode",
    "score_corpus",
    "score_document",
    "score_sentence",
    "sentence_split",
    "word_tokens",
]
