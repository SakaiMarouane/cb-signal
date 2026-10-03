"""Consolidate raw ECB / Fed / BoE corpora into a single interim parquet.

The result is `data/interim/corpus.parquet` with the canonical schema
(see `schema.py`). This is the input for every downstream stage: sentiment
scoring, topic modelling, signal construction, backtesting.

Any additional source added later must only append a parquet file with the
canonical columns; this consolidation step ignores per-source directory
structure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from cb_signal.ingest.schema import CANONICAL_COLUMNS

RAW_INPUTS = [
    ("ecb", "speeches.parquet"),
    ("fed", "corpus.parquet"),
    ("boe", "corpus.parquet"),
]


def consolidate(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for src_dir, filename in RAW_INPUTS:
        path = root / "data" / "raw" / src_dir / filename
        if not path.exists():
            logger.warning(f"[CONSOLIDATE] missing {path}, skipping")
            continue
        df = pd.read_parquet(path)
        missing = set(CANONICAL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        frames.append(df[CANONICAL_COLUMNS])
        logger.info(f"[CONSOLIDATE] loaded {len(df)} rows from {path}")

    if not frames:
        raise RuntimeError("No raw corpora found. Run the scrapers first.")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["doc_id"]).sort_values("date").reset_index(drop=True)

    out_dir = root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "corpus.parquet"
    out.to_parquet(out_path, index=False)

    summary = (
        out.groupby(["source", "event_type"], as_index=False)
        .size()
        .rename(columns={"size": "n_docs"})
    )
    logger.info(f"[CONSOLIDATE] wrote {len(out)} rows to {out_path}\n{summary.to_string(index=False)}")
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    consolidate(root)
