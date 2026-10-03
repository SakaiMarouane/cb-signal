from __future__ import annotations

from cb_signal.nlp.finetuned import LABELS, build_training_examples
from cb_signal.nlp.lexicons import DOVISH, HAWKISH


def test_build_training_examples_covers_all_labels():
    examples = build_training_examples()
    labels = {e.label for e in examples}
    assert labels == set(LABELS)


def test_build_training_examples_count_matches_lexicon_times_templates():
    examples = build_training_examples()
    n_hawkish = sum(1 for e in examples if e.label == "hawkish")
    n_dovish = sum(1 for e in examples if e.label == "dovish")
    # 4 carrier templates per lexicon phrase (see _HAWK_TEMPLATES)
    assert n_hawkish == len(HAWKISH) * 4
    assert n_dovish == len(DOVISH) * 4


def test_build_training_examples_phrase_appears_in_sentence():
    examples = build_training_examples()
    hawkish_example = next(e for e in examples if e.label == "hawkish")
    assert any(p in hawkish_example.text.lower() for p in (h.lower() for h in HAWKISH))
