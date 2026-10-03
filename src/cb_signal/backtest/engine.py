"""Backtest utilities: metrics + a simple long/short bond-proxy engine.

The signal maps `hawk_surprise` to a duration position on 2Y and 10Y rate
proxies. We convert weekly yield changes into approximate bond returns via a
constant modified-duration approximation (D_2Y = 1.9, D_10Y = 8.5) and apply
20 bps round-trip transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DURATIONS = {"DGS2": 1.9, "DGS10": 8.5}
TC_BPS = 20.0


@dataclass(frozen=True)
class TearsheetRow:
    strategy: str
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    turnover: float
    n_periods: int


def to_bond_returns(rate_wide: pd.DataFrame, cols: list[str], durations: dict[str, float] | None = None) -> pd.DataFrame:
    """Convert yield levels (%) to weekly bond returns via -D * dY / 100."""
    durations = durations or DURATIONS
    r = pd.DataFrame(index=rate_wide.index)
    for c in cols:
        d_yield = rate_wide[c].diff() / 100.0  # yield change in decimals
        r[c] = -durations[c] * d_yield
    return r


def run_strategy(signals: pd.DataFrame,
                 leg: str = "DGS2",
                 tc_bps: float = TC_BPS,
                 durations: dict[str, float] | None = None) -> pd.DataFrame:
    """Weekly long/short strategy on one bond leg using `signal_<leg>`.

    Convention: positive signal_<leg> = expect yield UP = SHORT the bond.
    Position is `-sign(signal) * |signal|` clipped to [-1, +1].
    """
    df = signals.copy().set_index("date").sort_index()
    df.index = pd.to_datetime(df.index)

    if leg == "DGS2":
        sig_col = "signal_2y"
    elif leg == "DGS10":
        sig_col = "signal_10y"
    else:
        raise ValueError(leg)

    bond_r = to_bond_returns(df[[leg]], [leg], durations=durations)
    raw_signal = df[sig_col].clip(-3, 3) / 3.0  # scale to ~[-1, 1]
    position = (-raw_signal).shift(1).clip(-1, 1).fillna(0.0)  # T+1 execution

    gross = position * bond_r[leg]
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (tc_bps / 1e4)
    net = gross - cost

    out = pd.DataFrame({
        "position": position,
        "bond_ret": bond_r[leg],
        "gross": gross,
        "cost": cost,
        "net": net.fillna(0.0),
        "equity": (1.0 + net.fillna(0.0)).cumprod(),
        "turnover": turnover,
    }, index=df.index)
    return out


def compute_metrics(bt: pd.DataFrame, strategy: str, periods_per_year: int = 52) -> TearsheetRow:
    r = bt["net"].dropna()
    if len(r) < 5:
        return TearsheetRow(strategy, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, len(r))
    mu = r.mean() * periods_per_year
    sd = r.std() * np.sqrt(periods_per_year)
    sharpe = mu / sd if sd > 1e-9 else np.nan
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min()
    active = bt.loc[bt["position"].abs() > 1e-6, "net"]
    hit = (active > 0).mean() if len(active) else np.nan
    turnover_ann = bt["turnover"].sum() / (len(bt) / periods_per_year)
    return TearsheetRow(strategy, float(mu), float(sd), float(sharpe),
                        float(max_dd), float(hit), float(turnover_ann), len(r))


def full_backtest(
    signals: pd.DataFrame, tc_bps: float = TC_BPS, durations: dict[str, float] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both legs; return (per-leg equity frames, tearsheet)."""
    legs = ["DGS2", "DGS10"]
    equity_curves = {}
    rows = []
    for leg in legs:
        bt = run_strategy(signals, leg=leg, tc_bps=tc_bps, durations=durations)
        equity_curves[leg] = bt
        rows.append(compute_metrics(bt, strategy=f"CB-Signal short-{leg[3:]}"))
    tearsheet = pd.DataFrame([r.__dict__ for r in rows])
    # Combine legs 50/50
    combined_r = (equity_curves["DGS2"]["net"].fillna(0.0)
                  + equity_curves["DGS10"]["net"].fillna(0.0)) / 2.0
    combined = pd.DataFrame({
        "position": (equity_curves["DGS2"]["position"] + equity_curves["DGS10"]["position"]) / 2.0,
        "net": combined_r,
        "equity": (1 + combined_r).cumprod(),
        "turnover": (equity_curves["DGS2"]["turnover"] + equity_curves["DGS10"]["turnover"]) / 2.0,
    })
    equity_curves["COMBINED"] = combined
    tearsheet = pd.concat([
        tearsheet,
        pd.DataFrame([compute_metrics(combined, "CB-Signal 50/50").__dict__]),
    ], ignore_index=True)
    return equity_curves, tearsheet


# Named macro regimes for sub-period stability reporting. Boundaries are
# approximate and meant to isolate qualitatively distinct monetary-policy
# environments, not to be read as official NBER/CEPR dating.
SUBPERIODS: dict[str, tuple[str, str]] = {
    "GFC (2007-2009)": ("2007-01-01", "2009-12-31"),
    "ZIRP (2009-2015)": ("2009-01-01", "2015-12-31"),
    "COVID (2020-2021)": ("2020-01-01", "2021-12-31"),
    "Hiking cycle (2022-2023)": ("2022-01-01", "2023-12-31"),
    "Plateau (2024+)": ("2024-01-01", "2030-12-31"),
}


def subperiod_stability(
    signals: pd.DataFrame,
    periods: dict[str, tuple[str, str]] | None = None,
    tc_bps: float = TC_BPS,
    durations: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Re-run `full_backtest` on each named sub-period; one tearsheet per period.

    A period with too little data for a given leg (e.g. a signal history
    that starts after a period's end) is silently skipped for that leg.
    """
    periods = periods or SUBPERIODS
    df = signals.copy()
    df["date"] = pd.to_datetime(df["date"])

    rows = []
    for name, (start, end) in periods.items():
        sliced = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(sliced) < 10:
            continue
        _, tearsheet = full_backtest(sliced.reset_index(drop=True), tc_bps=tc_bps, durations=durations)
        tearsheet.insert(0, "period", name)
        rows.append(tearsheet)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
