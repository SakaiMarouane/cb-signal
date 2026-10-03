"""Signal construction from document-level sentiment scores.

Inputs: `data/processed/sentiment_dict.parquet` (or finbert) + rates panel.
Outputs: `data/processed/signals.parquet` with columns:
    date, hawk_index, hawk_ema, hawk_surprise, regime, signal_2y, signal_10y

Construction steps
------------------
1. Aggregate document net_tone into a daily / weekly hawkish index by source,
   forward-fill inside a source, then average across sources.
2. Compute an EMA baseline (halflife 90 days). `hawk_surprise` = index - EMA.
3. Fit a 3-state Gaussian HMM on rolling (level, change) of the index to
   label regimes {dovish, neutral, hawkish}.
4. Build the tradable signal on the front (2Y) and long (10Y) rates as a
   z-scored `hawk_surprise` with a sign convention: hawkish surprise ->
   *short* duration (positive signal = short bonds = expected yield rise).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def _pivot_rates(rates: pd.DataFrame) -> pd.DataFrame:
    wide = rates.pivot(index="date", columns="series", values="value").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide


def build_hawk_index(sentiment: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    """Aggregate document net_tone into a weekly source-averaged hawkish index."""
    df = sentiment.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["net_tone"])
    daily = (
        df.groupby(["source", "date"])["net_tone"].mean()
        .unstack("source").sort_index()
    )
    weekly = daily.resample(freq).mean().ffill(limit=6)
    weekly["hawk_index"] = weekly.mean(axis=1)
    weekly = weekly.dropna(subset=["hawk_index"])
    return weekly


def add_ema_surprise(idx: pd.DataFrame, halflife: int = 12) -> pd.DataFrame:
    """Add hawk_ema (EMA baseline) and hawk_surprise = index - EMA (z-scored)."""
    out = idx.copy()
    out["hawk_ema"] = out["hawk_index"].ewm(halflife=halflife, adjust=False).mean()
    surp = out["hawk_index"] - out["hawk_ema"]
    # Expanding z-score to avoid look-ahead
    mu = surp.expanding(min_periods=13).mean()
    sd = surp.expanding(min_periods=13).std()
    out["hawk_surprise"] = ((surp - mu) / sd).fillna(0.0).clip(-3, 3)
    return out


def _fit_hmm_states(X: np.ndarray, n_states: int, seed: int):
    # Import order matters on this project's Windows dev environment: a
    # compiled/Cython package like hmmlearn imported before `torch` can
    # corrupt a later `torch` DLL load elsewhere in the same process (seen
    # as an unrelated-looking `OSError: ... c10.dll ...` the first time
    # anything imports torch afterwards). Harmless if torch isn't
    # installed. See cb_signal.nlp.topics for the same guard.
    with contextlib.suppress(ImportError):
        import torch  # noqa: F401
    from hmmlearn.hmm import GaussianHMM  # local import: avoid cost when not used

    model = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=200, random_state=seed)
    model.fit(X)
    return model


def _remap_by_level(model, raw: np.ndarray, n_states: int) -> np.ndarray:
    """Relabel raw HMM states to {0: dovish, ..., n_states-1: hawkish} by mean level."""
    means = np.array([model.means_[k, 0] for k in range(n_states)])
    order = np.argsort(means)  # low -> high
    remap = {int(order[k]): k for k in range(n_states)}
    return np.array([remap[int(s)] for s in raw])


def fit_regime_hmm(idx: pd.DataFrame, n_states: int = 3, seed: int = 42) -> pd.Series:
    """Fit a 3-state Gaussian HMM on (level, weekly change); return state labels
    remapped to {0: dovish, 1: neutral, 2: hawkish} by mean level.

    This fits once on the *entire* history, so early-sample labels are
    informed by the model's fit to later data — fine for a descriptive
    regime overlay (the tradable signal does not depend on `regime`; see
    `build_signals`), but not point-in-time. Use
    `fit_regime_hmm_walkforward` for an out-of-sample version of this.
    """
    level = idx["hawk_index"].values
    change = np.concatenate([[0.0], np.diff(level)])
    X = np.column_stack([level, change])
    model = _fit_hmm_states(X, n_states, seed)
    raw = model.predict(X)
    labels = _remap_by_level(model, raw, n_states)
    return pd.Series(labels, index=idx.index, name="regime")


def fit_regime_hmm_walkforward(
    idx: pd.DataFrame, n_states: int = 3, seed: int = 42, min_train_years: int = 3
) -> pd.Series:
    """Point-in-time version of `fit_regime_hmm`: refit yearly on an expanding
    window of data up to (not including) each calendar year, then label that
    year out-of-sample. The first `min_train_years` are labelled by the
    first available fit (there's no earlier data to train on) and so are
    in-sample by necessity — this matches standard walk-forward practice of
    reporting a short warm-up period.
    """
    level = idx["hawk_index"].values
    change = np.concatenate([[0.0], np.diff(level)])
    X = np.column_stack([level, change])
    years = pd.DatetimeIndex(idx.index).year
    first_year, last_year = years.min(), years.max()
    train_end_year = first_year + min_train_years

    labels = np.full(len(idx), -1, dtype=int)

    # Warm-up: in-sample fit on the first `min_train_years`, used to label
    # that same window (no earlier data exists to fit on out-of-sample).
    warm_mask = years < train_end_year
    if warm_mask.sum() >= n_states:
        model = _fit_hmm_states(X[warm_mask], n_states, seed)
        labels[warm_mask] = _remap_by_level(model, model.predict(X[warm_mask]), n_states)

    for year in range(train_end_year, last_year + 1):
        train_mask = years < year
        test_mask = years == year
        if train_mask.sum() < n_states * 5 or test_mask.sum() == 0:
            continue
        model = _fit_hmm_states(X[train_mask], n_states, seed)
        raw = model.predict(X[test_mask])
        labels[test_mask] = _remap_by_level(model, raw, n_states)

    return pd.Series(labels, index=idx.index, name="regime")


def build_signals(
    sentiment: pd.DataFrame,
    rates: pd.DataFrame,
    walk_forward_regime: bool = False,
    ema_halflife: int = 12,
    hmm_n_states: int = 3,
    hmm_seed: int = 42,
) -> pd.DataFrame:
    """Full signal build: hawkish index -> surprise -> regime -> tradable signal.

    `signal_2y`/`signal_10y` are built only from `hawk_index`/`hawk_surprise`,
    both computed with expanding (point-in-time) windows — regime labels are
    a descriptive overlay and do not feed the tradable signal, so
    `walk_forward_regime` changes the regime plot/labels but not the backtest.
    """
    idx = build_hawk_index(sentiment)
    idx = add_ema_surprise(idx, halflife=ema_halflife)
    if walk_forward_regime:
        idx["regime"] = fit_regime_hmm_walkforward(idx, n_states=hmm_n_states, seed=hmm_seed).values
    else:
        idx["regime"] = fit_regime_hmm(idx, n_states=hmm_n_states, seed=hmm_seed).values

    rw = _pivot_rates(rates).resample("W-FRI").last().ffill()
    signal_frame = idx[["hawk_index", "hawk_ema", "hawk_surprise", "regime"]].copy()

    # Composite signal: 50% level (z-scored hawk_index) + 50% surprise.
    level_z = (
        (idx["hawk_index"] - idx["hawk_index"].expanding(min_periods=13).mean())
        / idx["hawk_index"].expanding(min_periods=13).std()
    ).fillna(0.0).clip(-3, 3)
    composite = (0.5 * level_z + 0.5 * signal_frame["hawk_surprise"]).clip(-3, 3)
    # Smooth to reduce turnover
    composite = composite.rolling(4, min_periods=1).mean()

    # Sign convention: hawkish -> short duration (rates rise expected)
    signal_frame["signal_2y"] = composite
    signal_frame["signal_10y"] = 0.6 * composite  # long end less reactive

    # Attach the actual rate levels for the backtest.
    for col in ("DGS2", "DGS10", "IRLTLT01DEM156N", "IRLTLT01GBM156N", "VIXCLS"):
        if col in rw.columns:
            signal_frame[col] = rw[col].reindex(signal_frame.index).ffill()

    logger.info(
        f"[SIGNALS] {len(signal_frame)} weekly obs; "
        f"regimes: {signal_frame['regime'].value_counts().to_dict()}"
    )
    return signal_frame.reset_index().rename(columns={"index": "date", "date": "date"})


def build_from_paths(root: Path,
                     sentiment_path: Path | None = None,
                     rates_path: Path | None = None) -> pd.DataFrame:
    sentiment_path = sentiment_path or root / "data" / "processed" / "sentiment_dict.parquet"
    rates_path = rates_path or root / "data" / "raw" / "rates" / "rates.parquet"
    sentiment = pd.read_parquet(sentiment_path)
    rates = pd.read_parquet(rates_path)
    return build_signals(sentiment, rates)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    df = build_from_paths(root)
    out = root / "data" / "processed" / "signals.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info(f"[SIGNALS] wrote {out}")
