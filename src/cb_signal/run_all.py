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

from cb_signal.backtest.engine import full_backtest, subperiod_stability
from cb_signal.config import load_config
from cb_signal.nlp.dictionary import score_corpus as score_corpus_dict
from cb_signal.nlp.lexicons import build_lexicon
from cb_signal.signals.build import build_signals
from cb_signal.synth import write_synthetic

SENTIMENT_METHODS = ("dict", "finbert", "transformer")


def _score_sentiment(method: str, root: Path, corpus: pd.DataFrame, finbert_model: str | None = None) -> pd.DataFrame:
    """Score `corpus` with the requested sentiment method.

    Returns a frame with a `net_tone` column regardless of method, so
    `signals.build.build_hawk_index` (written against the dictionary
    scorer's column name) works unchanged for every tier.
    """
    if method == "dict":
        lex = build_lexicon(root)
        return score_corpus_dict(corpus, lex)

    if method == "finbert":
        from cb_signal.nlp.finbert import DEFAULT_MODEL
        from cb_signal.nlp.finbert import score_corpus as score_corpus_finbert

        scored = score_corpus_finbert(corpus, model_name=finbert_model or DEFAULT_MODEL)
    elif method == "transformer":
        from cb_signal.nlp.finetuned import score_corpus as score_corpus_finetuned

        scored = score_corpus_finetuned(root, corpus)
    else:
        raise ValueError(f"unknown sentiment method: {method}")

    return scored.rename(columns={"tone": "net_tone"})


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

    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Run the cb-signal pipeline end to end.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config (relative to the repo root), default configs/default.yaml. "
        "Its values seed the defaults below; explicit flags still override them.",
    )
    parser.add_argument(
        "--config-override",
        nargs="*",
        default=None,
        metavar="key.path=value",
        help="Dotlist overrides applied on top of the config file, e.g. backtest.tc_bps=10.",
    )
    # First pass: only to discover --config/--config-override before setting
    # the rest of the parser's defaults from it.
    pre_args, _ = parser.parse_known_args()
    cfg = load_config(root, config_path=pre_args.config, overrides=pre_args.config_override)

    parser.add_argument(
        "--skip-synth",
        action="store_true",
        help="Skip synthetic data generation and use whatever is already in "
        "data/interim/corpus.parquet and data/raw/rates/rates.parquet "
        "(e.g. real data from `python -m cb_signal.ingest`).",
    )
    parser.add_argument(
        "--sentiment",
        choices=SENTIMENT_METHODS,
        default=cfg.sentiment.method,
        help=f"Sentiment tier to score the corpus with (default from config: {cfg.sentiment.method}). "
        "'finbert' and 'transformer' are slower and write to method-suffixed "
        "paths (sentiment_<method>.parquet, tearsheet_<method>.csv, ...) so "
        "they don't clobber the dict-method baseline outputs.",
    )
    parser.add_argument(
        "--subperiods",
        action="store_true",
        default=bool(cfg.pipeline.subperiods),
        help="Also compute a sub-period stability tearsheet (GFC/ZIRP/COVID/"
        "hiking/plateau) and write it to reports/tearsheet_subperiods<suffix>.csv.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=bool(cfg.pipeline.walk_forward),
        help="Also fit the HMM regime overlay walk-forward (yearly expanding "
        "refit, out-of-sample) instead of once on the full history, and plot "
        "it alongside the in-sample version for comparison. Note this only "
        "changes the regime plot/labels, not the tradable signal or backtest "
        "P&L: signal_2y/signal_10y are built from expanding EMA/z-score "
        "windows that are already point-in-time (see signals.build.build_signals).",
    )
    args = parser.parse_args()

    logger.info(f"Project root: {root}")

    if args.skip_synth:
        logger.info("[1/5] skipping synthetic generation, using existing corpus + rates")
    else:
        logger.info("[1/5] generating synthetic corpus + rates panel")
        write_synthetic(root)

    suffix = "" if args.sentiment == "dict" else f"_{args.sentiment}"

    logger.info(f"[2/5] scoring corpus with sentiment method={args.sentiment}")
    corpus = pd.read_parquet(root / "data" / "interim" / "corpus.parquet")
    scored = _score_sentiment(args.sentiment, root, corpus, finbert_model=cfg.sentiment.finbert_model)
    # sentiment_dict.parquet is the pre-existing name for the baseline tier;
    # keep it exactly so earlier real-data artifacts aren't orphaned.
    out_sent = root / "data" / "processed" / f"sentiment_{args.sentiment}.parquet"
    out_sent.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out_sent, index=False)

    logger.info("[3/5] building signals (hawk index, EMA, HMM, tradable)")
    rates = pd.read_parquet(root / "data" / "raw" / "rates" / "rates.parquet")
    signals = build_signals(
        scored,
        rates,
        ema_halflife=cfg.signals.ema_halflife_weeks,
        hmm_n_states=cfg.signals.hmm_n_states,
        hmm_seed=cfg.signals.hmm_seed,
    )
    signals.to_parquet(root / "data" / "processed" / f"signals{suffix}.parquet", index=False)

    durations = {"DGS2": cfg.backtest.duration_2y, "DGS10": cfg.backtest.duration_10y}
    logger.info("[4/5] running backtest")
    curves, tearsheet = full_backtest(signals, tc_bps=cfg.backtest.tc_bps, durations=durations)
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tearsheet.to_csv(reports_dir / f"tearsheet{suffix}.csv", index=False)
    logger.info(f"[4/5] tearsheet:\n{tearsheet.to_string(index=False)}")

    logger.info("[5/5] rendering figures")
    _plot_hawk_index(signals, figures_dir / f"01_hawk_index{suffix}.png")
    _plot_regime(signals, figures_dir / f"02_regime{suffix}.png")
    _plot_equity(curves, figures_dir / f"03_equity{suffix}.png")

    for leg in ("DGS2", "DGS10", "COMBINED"):
        curves[leg].to_parquet(root / "data" / "processed" / f"equity_{leg}{suffix}.parquet")

    if args.subperiods:
        logger.info("[6] sub-period stability (GFC/ZIRP/COVID/hiking/plateau)")
        sub = subperiod_stability(signals, tc_bps=cfg.backtest.tc_bps, durations=durations)
        sub_path = reports_dir / f"tearsheet_subperiods{suffix}.csv"
        sub.to_csv(sub_path, index=False)
        logger.info(f"[6] wrote {sub_path}:\n{sub.to_string(index=False)}")

    if args.walk_forward:
        logger.info("[7] walk-forward HMM regime overlay (comparison plot only)")
        signals_wf = build_signals(
            scored,
            rates,
            walk_forward_regime=True,
            ema_halflife=cfg.signals.ema_halflife_weeks,
            hmm_n_states=cfg.signals.hmm_n_states,
            hmm_seed=cfg.signals.hmm_seed,
        )
        _plot_regime(signals_wf, figures_dir / f"02_regime_walkforward{suffix}.png")
        n_changed = (signals["regime"].values != signals_wf["regime"].values).sum()
        logger.info(
            f"[7] walk-forward vs in-sample regime labels differ on "
            f"{n_changed}/{len(signals)} weeks ({n_changed / len(signals):.1%})"
        )

    logger.info("cb-signal pipeline complete.")


if __name__ == "__main__":
    main()
