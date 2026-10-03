"""CLI entry point: `python -m cb_signal.ingest [--sources ...] [--no-speeches]`.

Runs the full Phase A pipeline end-to-end:

    ECB speeches   -> data/raw/ecb/speeches.parquet
    Fed corpus     -> data/raw/fed/corpus.parquet
    BoE corpus     -> data/raw/boe/corpus.parquet
    Rates series   -> data/raw/rates/rates.parquet
    Consolidation  -> data/interim/corpus.parquet

Each source can be enabled/disabled independently so failures in one do not
block the others.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from cb_signal.ingest.boe import fetch_boe_corpus
from cb_signal.ingest.consolidate import consolidate
from cb_signal.ingest.ecb import fetch_ecb_speeches
from cb_signal.ingest.fed import fetch_fed_corpus
from cb_signal.ingest.rates import fetch_rates

ALL_SOURCES = ["ecb", "fed", "boe", "rates"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CB-Signal ingest pipeline.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=ALL_SOURCES,
        choices=ALL_SOURCES,
        help="Which sources to fetch (default: all).",
    )
    parser.add_argument(
        "--no-speeches",
        action="store_true",
        help="Skip individual speeches from Fed and BoE (statements/minutes only).",
    )
    parser.add_argument(
        "--fed-speech-start-year",
        type=int,
        default=2006,
        help="Earliest year for Fed speech scraping (default: 2006).",
    )
    parser.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="Skip the final consolidation step.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    logger.info(f"Project root: {root}")
    logger.info(f"Sources: {args.sources}")

    errors: list[str] = []

    if "ecb" in args.sources:
        try:
            fetch_ecb_speeches(root)
        except Exception as e:
            logger.exception("[ECB] fatal error")
            errors.append(f"ECB: {e}")

    if "fed" in args.sources:
        try:
            fetch_fed_corpus(
                root,
                include_speeches=not args.no_speeches,
                speech_start_year=args.fed_speech_start_year,
            )
        except Exception as e:
            logger.exception("[FED] fatal error")
            errors.append(f"FED: {e}")

    if "boe" in args.sources:
        try:
            fetch_boe_corpus(root, include_speeches=not args.no_speeches)
        except Exception as e:
            logger.exception("[BOE] fatal error")
            errors.append(f"BOE: {e}")

    if "rates" in args.sources:
        try:
            fetch_rates(root)
        except Exception as e:
            logger.exception("[RATES] fatal error")
            errors.append(f"RATES: {e}")

    if not args.skip_consolidate:
        try:
            consolidate(root)
        except Exception as e:
            logger.exception("[CONSOLIDATE] fatal error")
            errors.append(f"CONSOLIDATE: {e}")

    if errors:
        logger.error("Ingest completed with errors:\n  - " + "\n  - ".join(errors))
        return 1
    logger.info("Ingest completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
