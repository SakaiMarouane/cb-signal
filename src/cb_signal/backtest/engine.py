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


def to_bond_returns(rate_wide: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert yield levels (%) to weekly bond returns via -D * dY / 100."""
    r = pd.DataFrame(index=rate_wide.index)
    for c in cols:
        d_yield = rate_wide[c].diff() / 100.0  # yield change in decimals
        r[c] = -DURATIONS[c] * d_yield
    return r


def run_strategy(signals: pd.DataFrame,
                 leg: str = "DGS2",
                 tc_bps: float = TC_BPS) -> pd.DataFrame:
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

    bond_r = to_bond_returns(df[[leg]], [leg])
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


def full_backtest(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both legs; return (per-leg equity frames, tearsheet)."""
    legs = ["DGS2", "DGS10"]
    equity_curves = {}
    rows = []
    for leg in legs:
        bt = run_strategy(signals, leg=leg)
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
