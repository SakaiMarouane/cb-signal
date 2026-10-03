"""End-to-end pipeline for CB-Signal on synthetic data.

Runs: synth -> dictionary sentiment -> signals -> backtest -> figures.
Produces `reports/figures/*.png` and `reports/tearsheet.csv`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from cb_signal.backtest.engine import full_backtest
from cb_signal.nlp.dictionary import score_corpus
from cb_signal.nlp.lexicons import build_lexicon
from cb_signal.signals.build import build_signals
from cb_signal.synth import write_synthetic


def _plot_hawk_index(signals: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    df = signals.set_index("date").sort_index()
    ax.plot(df.index, df["hawk_index"], label="Hawk index (weekly)", color="#333", lw=1.0)
    ax.plot(df.index, df["hawk_ema"], label="EMA baseline (hl=12w)", color="#c33", lw=1.2)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_title("CB-Signal — hawkish index and EMA baseline")
    ax.set_ylabel("Net tone")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_regime(signals: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    df = signals.set_index("date").sort_index()
    colors = {0: "#4a7ab8", 1: "#bbbbbb", 2: "#c9564e"}
    labels = {0: "Dovish", 1: "Neutral", 2: "Hawkish"}
    for state, c in colors.items():
        m = df["regime"] == state
        ax.fill_between(df.index, 0, 1, where=m, transform=ax.get_xaxis_transform(),
                        color=c, alpha=0.35, label=labels[state])
    ax.plot(df.index, df["hawk_index"], color="black", lw=0.8)
    ax.set_title("HMM regime overlay (3-state Gaussian on level + change)")
    ax.set_ylabel("Hawk index")
    ax.legend(loc="lower left", ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_equity(curves: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for name in ("DGS2", "DGS10", "COMBINED"):
        eq = curves[name]["equity"]
        ax.plot(eq.index, eq, label=name)
    ax.axhline(1.0, color="grey", ls="--", lw=0.5)
    ax.set_title("CB-Signal equity curves (net of 20bps TC)")
    ax.set_ylabel("Equity (start=1.0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the cb-signal pipeline end to end.")
    parser.add_argument(
        "--skip-synth",
        action="store_true",
        help="Skip synthetic data generation and use whatever is already in "
        "data/interim/corpus.parquet and data/raw/rates/rates.parquet "
        "(e.g. real data from `python -m cb_signal.ingest`).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    logger.info(f"Project root: {root}")

    if args.skip_synth:
        logger.info("[1/5] skipping synthetic generation, using existing corpus + rates")
    else:
        logger.info("[1/5] generating synthetic corpus + rates panel")
        write_synthetic(root)

    logger.info("[2/5] scoring corpus with dictionary lexicon")
    corpus = pd.read_parquet(root / "data" / "interim" / "corpus.parquet")
    lex = build_lexicon(root)
    scored = score_corpus(corpus, lex)
    out_sent = root / "data" / "processed" / "sentiment_dict.parquet"
    out_sent.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out_sent, index=False)

    logger.info("[3/5] building signals (hawk index, EMA, HMM, tradable)")
    rates = pd.read_parquet(root / "data" / "raw" / "rates" / "rates.parquet")
    signals = build_signals(scored, rates)
    signals.to_parquet(root / "data" / "processed" / "signals.parquet", index=False)

    logger.info("[4/5] running backtest")
    curves, tearsheet = full_backtest(signals)
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tearsheet.to_csv(reports_dir / "tearsheet.csv", index=False)
    logger.info(f"[4/5] tearsheet:\n{tearsheet.to_string(index=False)}")

    logger.info("[5/5] rendering figures")
    _plot_hawk_index(signals, figures_dir / "01_hawk_index.png")
    _plot_regime(signals, figures_dir / "02_regime.png")
    _plot_equity(curves, figures_dir / "03_equity.png")

    for leg in ("DGS2", "DGS10", "COMBINED"):
        curves[leg].to_parquet(root / "data" / "processed" / f"equity_{leg}.parquet")

    logger.info("cb-signal pipeline complete.")


if __name__ == "__main__":
    main()
