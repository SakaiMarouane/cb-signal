"""CLI: `python -m cb_signal.nlp` — score the consolidated corpus.

Reads `data/interim/corpus.parquet` and applies the selected sentiment model.

Methods
-------
--method dict      Dictionary hawkish/dovish scorer. Fast, no model download.
                   Output: data/processed/sentiment_dict.parquet
--method finbert   Transformer classifier (default: ProsusAI/finbert, a
                   public model; see cb_signal.nlp.finbert for the gated
                   FOMC-specific alternative). CPU-friendly but slow.
                   Output: data/processed/sentiment_finbert.parquet
--method transformer  Fine-tuned tier: MiniLM sentence embeddings + a
                   logistic-regression head trained on a small templated
                   example set (see cb_signal.nlp.finetuned for data
                   provenance). Trains itself on first use.
                   Output: data/processed/sentiment_transformer.parquet

Flags
-----
--input   override the input parquet path
--output  override the output parquet path
--source  restrict scoring to a single source (ECB / FED / BOE)
--model   for finbert, override the HuggingFace model id
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from cb_signal.nlp.dictionary import score_corpus as score_corpus_dict
from cb_signal.nlp.finbert import DEFAULT_MODEL as FINBERT_DEFAULT
from cb_signal.nlp.finbert import score_corpus as score_corpus_finbert
from cb_signal.nlp.lexicons import build_lexicon


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the consolidated corpus with a sentiment model.")
    parser.add_argument("--method", choices=["dict", "finbert", "transformer"], default="dict",
                        help="Which sentiment method to run (default: dict).")
    parser.add_argument("--input", type=Path, default=None, help="Path to interim corpus parquet.")
    parser.add_argument("--output", type=Path, default=None, help="Path to write scored parquet.")
    parser.add_argument("--source", choices=["ECB", "FED", "BOE"], default=None,
                        help="Optional single-source filter for smoke testing.")
    parser.add_argument("--model", default=FINBERT_DEFAULT,
                        help="HuggingFace model id (finbert method only).")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    input_path = args.input or (root / "data" / "interim" / "corpus.parquet")

    default_out_names = {
        "dict": "sentiment_dict.parquet",
        "finbert": "sentiment_finbert.parquet",
        "transformer": "sentiment_transformer.parquet",
    }
    output_path = args.output or (root / "data" / "processed" / default_out_names[args.method])

    if not input_path.exists():
        logger.error(f"input corpus not found: {input_path}. Run `python -m cb_signal.ingest` first.")
        return 2

    logger.info(f"loading corpus from {input_path}")
    corpus = pd.read_parquet(input_path)
    if args.source is not None:
        corpus = corpus[corpus["source"] == args.source].reset_index(drop=True)
        logger.info(f"filtered to source={args.source}: {len(corpus)} docs")

    if args.method == "dict":
        lex = build_lexicon(root)
        logger.info(
            f"lexicon: {len(lex.hawkish)} hawkish, {len(lex.dovish)} dovish, "
            f"{len(lex.uncertainty)} uncertainty, max_ngram={lex.max_ngram}"
        )
        scored = score_corpus_dict(corpus, lex)
    elif args.method == "finbert":
        scored = score_corpus_finbert(corpus, model_name=args.model)
    else:
        from cb_signal.nlp.finetuned import score_corpus as score_corpus_finetuned

        scored = score_corpus_finetuned(root, corpus)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_path, index=False)
    logger.info(f"wrote {len(scored)} scored docs to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
