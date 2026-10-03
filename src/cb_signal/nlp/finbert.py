"""Transformer-based sentiment scoring for central bank text.

Two models are supported out of the box, both loaded via HuggingFace
`transformers`. Neither requires a GPU — a modern CPU processes ~2-4
chunks/second, so the full corpus (~5k documents at ~10 chunks each)
scores in ~1-2 hours on a laptop.

Default model
-------------
`gtfintechlab/FOMC-RoBERTa` (Shah, Paturi & Chava, 2023, "Trillion Dollar
Words: A New Financial Dataset, Task & Market Analysis") — a RoBERTa
fine-tuned on FOMC statements/minutes/speeches with three labels:
`HAWKISH`, `DOVISH`, `NEUTRAL`. This is the closest off-the-shelf model
to what CB-Signal targets and is what the LaTeX paper cites as the
transformer baseline.

Fallback model
--------------
`ProsusAI/finbert` — general financial sentiment (positive / negative /
neutral). Used only when the FOMC model cannot be downloaded (no
internet on the target machine). We remap `positive -> DOVISH` and
`negative -> HAWKISH` following the convention that "positive" for
market participants typically means expansionary policy language. This
mapping is imperfect and is why the FOMC model is the default.

Long-document handling
----------------------
BERT-family models cap input at 512 tokens. Central bank minutes are
5-15k tokens. We chunk each document into overlapping windows (stride
= 384 tokens), score each window, and aggregate probabilities:

    p_doc = mean over chunks of softmax(logits)

Then take argmax and store the full distribution so downstream signal
construction can use soft scores rather than the hard label.

Output columns (written to `data/processed/sentiment_finbert.parquet`):

    doc_id, source, event_type, date,
    p_hawkish, p_dovish, p_neutral,
    label, tone
    (tone = p_hawkish - p_dovish, in [-1, 1])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

DEFAULT_MODEL = "gtfintechlab/FOMC-RoBERTa"
FALLBACK_MODEL = "ProsusAI/finbert"

# 3-way label indices are model-specific; we resolve them at load time.
HAWK_ALIASES = {"hawkish", "hawk", "negative"}
DOV_ALIASES = {"dovish", "dove", "positive"}
NEUTRAL_ALIASES = {"neutral", "neutral-hawkish", "neutral-dovish"}


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 512
    stride: int = 384
    min_chunk_tokens: int = 32


def _resolve_label_indices(id2label: dict[int, str]) -> dict[str, int]:
    """Map the model's label names to canonical hawkish/dovish/neutral indices."""
    idx = {"hawkish": -1, "dovish": -1, "neutral": -1}
    for i, name in id2label.items():
        low = name.lower()
        if low in HAWK_ALIASES:
            idx["hawkish"] = int(i)
        elif low in DOV_ALIASES:
            idx["dovish"] = int(i)
        elif low in NEUTRAL_ALIASES:
            idx["neutral"] = int(i)
    missing = [k for k, v in idx.items() if v < 0]
    if missing:
        raise ValueError(
            f"Model id2label {id2label} does not map cleanly to hawkish/dovish/neutral; "
            f"missing: {missing}"
        )
    return idx


def _load_model(model_name: str):
    """Load a HuggingFace sequence-classification model on CPU.

    Imports happen inside the function so the module stays importable
    on machines that only have the base install (no torch).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.eval()
    label_idx = _resolve_label_indices(mdl.config.id2label)
    return tok, mdl, label_idx


def _chunk_ids(input_ids: list[int], cfg: ChunkingConfig) -> list[list[int]]:
    """Split a token id list into overlapping windows respecting the stride."""
    if len(input_ids) <= cfg.max_tokens:
        return [input_ids]
    chunks: list[list[int]] = []
    start = 0
    step = cfg.max_tokens - (cfg.max_tokens - cfg.stride)
    step = max(step, 1)
    while start < len(input_ids):
        end = min(start + cfg.max_tokens, len(input_ids))
        chunk = input_ids[start:end]
        if len(chunk) >= cfg.min_chunk_tokens:
            chunks.append(chunk)
        if end == len(input_ids):
            break
        start += cfg.stride
    return chunks


def score_document(
    text: str,
    tokenizer,
    model,
    label_idx: dict[str, int],
    cfg: ChunkingConfig,
) -> dict[str, float]:
    """Score one document; returns soft probabilities aggregated across chunks."""
    import torch

    enc = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = enc["input_ids"]
    if not input_ids:
        return {"p_hawkish": 0.0, "p_dovish": 0.0, "p_neutral": 1.0}

    chunks = _chunk_ids(input_ids, cfg)
    probs_stack = np.zeros((len(chunks), 3), dtype=np.float64)

    with torch.inference_mode():
        for i, chunk in enumerate(chunks):
            # rebuild proper attention_mask + special tokens for each window
            ids = [tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id]
            attn = [1] * len(ids)
            batch = {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attn], dtype=torch.long),
            }
            logits = model(**batch).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_stack[i, 0] = probs[label_idx["hawkish"]]
            probs_stack[i, 1] = probs[label_idx["dovish"]]
            probs_stack[i, 2] = probs[label_idx["neutral"]]

    mean_probs = probs_stack.mean(axis=0)
    return {
        "p_hawkish": float(mean_probs[0]),
        "p_dovish": float(mean_probs[1]),
        "p_neutral": float(mean_probs[2]),
    }


def score_corpus(
    corpus: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    chunking: ChunkingConfig | None = None,
) -> pd.DataFrame:
    """Score every document with a transformer classifier.

    Falls back to `FALLBACK_MODEL` if the primary model cannot be loaded
    (typically an offline machine where only the ProsusAI weights are
    already cached).
    """
    required = {"doc_id", "source", "event_type", "date", "text"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"corpus missing required columns: {missing}")

    cfg = chunking or ChunkingConfig()

    try:
        tokenizer, model, label_idx = _load_model(model_name)
        used_model = model_name
    except Exception as e:
        logger.warning(f"[FINBERT] {model_name} unavailable ({e}); falling back to {FALLBACK_MODEL}")
        tokenizer, model, label_idx = _load_model(FALLBACK_MODEL)
        used_model = FALLBACK_MODEL

    logger.info(f"[FINBERT] scoring {len(corpus)} documents with {used_model}")

    records: list[dict] = []
    for i, row in enumerate(corpus.itertuples(index=False), start=1):
        scores = score_document(row.text, tokenizer, model, label_idx, cfg)
        tone = scores["p_hawkish"] - scores["p_dovish"]
        label = max(scores.items(), key=lambda kv: kv[1])[0].removeprefix("p_")
        records.append(
            {
                "doc_id": row.doc_id,
                "source": row.source,
                "event_type": row.event_type,
                "date": row.date,
                **scores,
                "tone": tone,
                "label": label,
                "model": used_model,
            }
        )
        if i % 100 == 0:
            logger.info(f"[FINBERT] {i} / {len(corpus)} done")

    out = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    logger.info(f"[FINBERT] finished; mean tone = {out['tone'].mean():+.3f}")
    return out


def run(root: Path, model_name: str = DEFAULT_MODEL) -> pd.DataFrame:
    """Convenience wrapper: read consolidated corpus, score, persist."""
    corpus_path = root / "data" / "interim" / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. Run `python -m cb_signal.ingest` first."
        )
    corpus = pd.read_parquet(corpus_path)
    out = score_corpus(corpus, model_name=model_name)
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sentiment_finbert.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"[FINBERT] wrote {len(out)} rows to {out_path}")
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    run(root)
