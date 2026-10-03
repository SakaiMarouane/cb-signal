from __future__ import annotations

import numpy as np
import pandas as pd

from cb_signal.backtest.engine import subperiod_stability


def _synthetic_signals(start: str = "2005-01-07", n_weeks: int = 1200) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_weeks, freq="W-FRI")
    rng = np.random.default_rng(0)
    signal = rng.normal(size=n_weeks)
    dgs2 = 2.0 + np.cumsum(rng.normal(scale=0.02, size=n_weeks))
    dgs10 = 3.0 + np.cumsum(rng.normal(scale=0.02, size=n_weeks))
    return pd.DataFrame(
        {
            "date": dates,
            "signal_2y": signal,
            "signal_10y": 0.6 * signal,
            "DGS2": dgs2,
            "DGS10": dgs10,
        }
    )


def test_subperiod_stability_returns_one_tearsheet_row_set_per_period():
    signals = _synthetic_signals()
    out = subperiod_stability(signals)
    assert not out.empty
    assert "period" in out.columns
    # each covered period contributes 3 strategy rows (short-2, short-10, 50/50)
    counts = out.groupby("period").size()
    assert (counts == 3).all()


def test_subperiod_stability_skips_periods_outside_the_sample():
    # sample starts in 2018, so GFC/ZIRP periods have no data and must be skipped
    signals = _synthetic_signals(start="2018-01-05", n_weeks=200)
    out = subperiod_stability(signals)
    assert "GFC (2007-2009)" not in set(out["period"])
