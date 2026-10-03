"""Synthetic data generator for CB-Signal.

Produces the same schemas as the real ingest pipeline so downstream code
runs unchanged. Two outputs:

* `data/interim/corpus.parquet` — ECB/Fed/BoE-style documents with a
  latent hawkish/dovish tone that drifts through time (calm 2010-14,
  hawkish 2015-19 taper, dovish 2020 COVID, hawkish 2022 hiking cycle).
* `data/raw/rates/rates.parquet` — DGS2, DGS10, DE10, UK10, VIXCLS
  daily series driven by the same latent tone plus noise, so the
  sentiment signal has real forward predictive power on the synthetic
  rates panel.

The generator is seeded and reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from cb_signal.nlp.lexicons import DOVISH, HAWKISH

_RNG = np.random.default_rng(20260804)

SOURCES = [
    ("ECB", "speech", 8),
    ("ECB", "press_conference", 2),
    ("FED", "statement", 1),
    ("FED", "minutes", 1),
    ("FED", "speech", 6),
    ("BOE", "minutes", 1),
    ("BOE", "speech", 4),
]


def _regime_tone(dates: pd.DatetimeIndex) -> np.ndarray:
    """Latent hawkish tone in [-1, +1], with named regimes."""
    tone = np.zeros(len(dates))
    for i, d in enumerate(dates):
        y = d.year + (d.month - 1) / 12
        if y < 2015:
            tone[i] = -0.2  # accommodative post-GFC
        elif y < 2020:
            tone[i] = +0.4  # gradual hawkish (taper / hikes)
        elif y < 2021.5:
            tone[i] = -0.7  # COVID dovish
        elif y < 2024:
            tone[i] = +0.7  # 2022-23 hiking cycle
        else:
            tone[i] = +0.1  # 2024+ plateau
    tone += _RNG.normal(0, 0.15, len(dates))
    return np.clip(tone, -1.0, 1.0)


def _make_document(tone: float, source: str, event_type: str) -> tuple[str, str]:
    """Return (title, text) — hawk/dove phrase mix biased by `tone`."""
    p_hawk = np.clip(0.5 + 0.4 * tone, 0.05, 0.95)
    n_sent = _RNG.integers(15, 40)
    sentences: list[str] = []
    for _ in range(n_sent):
        is_hawk = _RNG.random() < p_hawk
        pool = HAWKISH if is_hawk else DOVISH
        phrase = _RNG.choice(pool)
        filler_pre = _RNG.choice(
            ["The Governing Council notes that", "We observe that",
             "Data indicate that", "The Committee judges that",
             "Recent developments show", "In the outlook,"]
        )
        filler_post = _RNG.choice(
            ["remain a key consideration for policy.",
             "warrant close attention in the coming quarters.",
             "will shape our reaction function going forward.",
             "are consistent with our medium-term assessment.",
             "reinforce the case for a data-dependent approach.",
             "have implications for the transmission of policy."]
        )
        sentences.append(f"{filler_pre} {phrase} {filler_post}")
    text = " ".join(sentences)
    title_tone = "hawkish" if tone > 0.2 else ("dovish" if tone < -0.2 else "balanced")
    title = f"{source} {event_type} — {title_tone} outlook"
    return title, text


def make_corpus(start: str = "2010-01-01", end: str = "2025-06-30") -> pd.DataFrame:
    """Generate a synthetic consolidated corpus."""
    dates = pd.date_range(start, end, freq="MS")  # month start
    records: list[dict] = []
    doc_counter = 0
    for d in dates:
        tone = _regime_tone(pd.DatetimeIndex([d]))[0]
        for source, event_type, n_per_month in SOURCES:
            for _ in range(n_per_month):
                offset_days = int(_RNG.integers(0, 28))
                dd = (d + pd.Timedelta(days=offset_days)).date()
                title, text = _make_document(tone, source, event_type)
                doc_counter += 1
                records.append(
                    {
                        "doc_id": f"{source.lower()}-{event_type}-{doc_counter:06d}",
                        "source": source,
                        "event_type": event_type,
                        "date": dd,
                        "speaker": f"{source} spokesperson",
                        "title": title,
                        "subtitle": "",
                        "url": f"https://synth.local/{source.lower()}/{doc_counter}",
                        "text": text,
                        "n_tokens": len(text.split()),
                    }
                )
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    logger.info(f"[SYNTH] generated corpus with {len(df)} documents")
    return df


def make_rates(start: str = "2010-01-01", end: str = "2025-06-30") -> pd.DataFrame:
    """Generate daily rates panel; weekly yield *changes* are driven by lagged tone.

    We put the predictive relationship on the change (not the level) so a
    sentiment-driven signal has direct forward-return power net of noise.
    """
    dates = pd.bdate_range(start, end)
    tone = _regime_tone(dates)
    tone_slow = pd.Series(tone).rolling(20, min_periods=1).mean().values
    # forward-looking predictive component on d_yield: alpha * (tone_slow[t-5])
    tone_lag = np.roll(tone_slow, 5)
    tone_lag[:5] = tone_slow[0]

    specs = {
        "DGS2":            {"start": 1.5, "alpha": 0.012, "vol": 0.012, "floor": -0.5, "cap": 8.0},
        "DGS10":           {"start": 2.5, "alpha": 0.010, "vol": 0.014, "floor": -0.5, "cap": 8.0},
        "IRLTLT01DEM156N": {"start": 1.8, "alpha": 0.008, "vol": 0.014, "floor": -0.5, "cap": 8.0},
        "IRLTLT01GBM156N": {"start": 2.2, "alpha": 0.010, "vol": 0.015, "floor": -0.5, "cap": 8.0},
        "VIXCLS":          {"start": 18.0, "alpha": -0.30, "vol": 1.2, "floor": 8.0, "cap": 80.0},
    }
    frames = []
    for series, s in specs.items():
        y = np.zeros(len(dates))
        y[0] = s["start"]
        for i in range(1, len(dates)):
            drift = s["alpha"] * tone_lag[i]
            noise = _RNG.normal(0, s["vol"])
            y[i] = np.clip(y[i - 1] + drift + noise, s["floor"], s["cap"])
        frames.append(
            pd.DataFrame({"date": dates.date, "series": series, "value": y})
        )
    out = pd.concat(frames, ignore_index=True).sort_values(["series", "date"]).reset_index(drop=True)
    logger.info(f"[SYNTH] generated rates panel: {len(out)} rows across {out['series'].nunique()} series")
    return out


def write_synthetic(root: Path) -> None:
    """Materialise both parquets in the standard project layout."""
    corpus = make_corpus()
    rates = make_rates()

    interim_dir = root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(interim_dir / "corpus.parquet", index=False)
    logger.info(f"[SYNTH] wrote {interim_dir / 'corpus.parquet'}")

    rates_dir = root / "data" / "raw" / "rates"
    rates_dir.mkdir(parents=True, exist_ok=True)
    rates.to_parquet(rates_dir / "rates.parquet", index=False)
    logger.info(f"[SYNTH] wrote {rates_dir / 'rates.parquet'}")


if __name__ == "__main__":
    write_synthetic(Path(__file__).resolve().parents[2])
