from __future__ import annotations

import pytest

from cb_signal.nlp.finbert import (
    ChunkingConfig,
    _chunk_ids,
    _resolve_label_indices,
)


def test_chunk_ids_short_document_returns_single_chunk():
    ids = list(range(100))
    chunks = _chunk_ids(ids, ChunkingConfig(max_tokens=512))
    assert len(chunks) == 1
    assert chunks[0] == ids


def test_chunk_ids_overlapping_windows():
    ids = list(range(1000))
    cfg = ChunkingConfig(max_tokens=512, stride=384, min_chunk_tokens=32)
    chunks = _chunk_ids(ids, cfg)
    assert len(chunks) >= 2
    # every chunk leaves room for the [CLS]/[SEP] tokens added later, or the
    # model's absolute position embeddings overflow (512 total, not content)
    for c in chunks:
        assert len(c) <= cfg.max_tokens - 2
        assert len(c) >= cfg.min_chunk_tokens
    # windows must overlap
    assert chunks[0][-1] > chunks[1][0]


def test_chunk_ids_leaves_room_for_special_tokens_on_exact_boundary():
    # A document exactly at max_tokens must still split, since [CLS]+512+[SEP]
    # would overflow a 512-position model.
    ids = list(range(512))
    chunks = _chunk_ids(ids, ChunkingConfig(max_tokens=512))
    for c in chunks:
        assert len(c) + 2 <= 512


def test_chunk_ids_drops_tiny_tail():
    ids = list(range(520))
    cfg = ChunkingConfig(max_tokens=512, stride=500, min_chunk_tokens=32)
    chunks = _chunk_ids(ids, cfg)
    # tail of 20 tokens after first 512-window should be dropped
    for c in chunks:
        assert len(c) >= cfg.min_chunk_tokens


def test_resolve_label_indices_fomc_roberta_style():
    id2label = {0: "HAWKISH", 1: "DOVISH", 2: "NEUTRAL"}
    idx = _resolve_label_indices(id2label)
    assert idx == {"hawkish": 0, "dovish": 1, "neutral": 2}


def test_resolve_label_indices_prosus_finbert_style():
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    idx = _resolve_label_indices(id2label)
    # positive maps to dovish, negative to hawkish under the documented alias set
    assert idx["dovish"] == 0
    assert idx["hawkish"] == 1
    assert idx["neutral"] == 2


def test_resolve_label_indices_rejects_unknown_labels():
    with pytest.raises(ValueError):
        _resolve_label_indices({0: "buy", 1: "sell", 2: "hold"})
