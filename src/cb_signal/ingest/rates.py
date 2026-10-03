"""Ingest daily rates data from FRED (Federal Reserve Economic Data).

FRED exposes a stable CSV endpoint per series:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}

We use it directly so no extra dependency (pandas_datareader) is required.

Series pulled by default (all daily unless noted):

    DGS2                US 2Y treasury constant-maturity yield (%)
    DGS10               US 10Y treasury constant-maturity yield (%)
    IRLTLT01DEM156N     DE 10Y long-term government bond (monthly, %)
    IRLTLT01GBM156N     UK 10Y long-term government bond (monthly, %)
    VIXCLS              VIX close (control variable)

Output: `data/raw/rates/rates.parquet` in long format
    columns: date, series, value
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from loguru import logger

from cb_signal.ingest._http import polite_get, project_cache_dir

FRED_CSV_TMPL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

DEFAULT_SERIES = [
    "DGS2",
    "DGS10",
    "IRLTLT01DEM156N",
    "IRLTLT01GBM156N",
    "VIXCLS",
]


def _fetch_series(root: Path, series_id: str) -> pd.DataFrame:
    url = FRED_CSV_TMPL.format(series=series_id)
    raw = polite_get(url, cache_dir=project_cache_dir(root))
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else "observation_date"
    df = df.rename(columns={date_col: "date", series_id: "value", series_id.lower(): "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series"] = series_id
    df = df.dropna(subset=["date", "value"])
    return df[["date", "series", "value"]]


def fetch_rates(root: Path, series: list[str] | None = None) -> pd.DataFrame:
    series = series or DEFAULT_SERIES
    output_dir = root / "data" / "raw" / "rates"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for s in series:
        try:
            df = _fetch_series(root, s)
            logger.info(f"[RATES] {s}: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")
            frames.append(df)
        except Exception as e:
            logger.warning(f"[RATES] {s} failed: {e}")

    if not frames:
        raise RuntimeError("No rates series could be fetched.")

    out = pd.concat(frames, ignore_index=True).sort_values(["series", "date"]).reset_index(drop=True)
    out_path = output_dir / "rates.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"[RATES] Wrote {len(out)} rows to {out_path}")
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    fetch_rates(root)
