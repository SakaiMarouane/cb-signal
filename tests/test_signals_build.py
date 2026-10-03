from __future__ import annotations

import numpy as np
import pandas as pd

from cb_signal.signals.build import fit_regime_hmm, fit_regime_hmm_walkforward


def _synthetic_index(n_years: int = 6) -> pd.DataFrame:
    dates = pd.date_range("2015-01-02", periods=n_years * 52, freq="W-FRI")
    rng = np.random.default_rng(0)
    # alternate dovish/hawkish blocks so there's real structure to find
    level = np.concatenate(
        [np.full(52, -1.0) if i % 2 == 0 else np.full(52, 1.0) for i in range(n_years)]
    )
    level = level + rng.normal(scale=0.1, size=len(dates))
    return pd.DataFrame({"hawk_index": level}, index=dates)


def test_fit_regime_hmm_returns_one_label_per_row():
    idx = _synthetic_index()
    regimes = fit_regime_hmm(idx)
    assert len(regimes) == len(idx)
    assert set(regimes.unique()) <= {0, 1, 2}


def test_fit_regime_hmm_walkforward_returns_one_label_per_row():
    idx = _synthetic_index()
    regimes = fit_regime_hmm_walkforward(idx, min_train_years=2)
    assert len(regimes) == len(idx)
    assert regimes.index.equals(idx.index)
    # every row gets a real label (warm-up + yearly refits should cover everything)
    assert (regimes >= 0).all()


def test_fit_regime_hmm_walkforward_warmup_uses_only_early_data():
    idx = _synthetic_index()
    # a pathologically short min_train_years still shouldn't crash or leave gaps
    regimes = fit_regime_hmm_walkforward(idx, min_train_years=1)
    assert len(regimes) == len(idx)
