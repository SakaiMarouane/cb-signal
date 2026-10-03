"""Public API for the NLP layer.

Three sentiment tiers are available (see `python -m cb_signal.nlp --help`):
dictionary (`cb_signal.nlp.dictionary`), transformer (`cb_signal.nlp.finbert`),
and a fine-tuned sentence-embedding + linear head tier
(`cb_signal.nlp.finetuned`). Only the dictionary scorer is re-exported here
since the other two require optional heavy dependencies (torch,
sentence-transformers) that shouldn't be imported just by `import cb_signal.nlp`.
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
