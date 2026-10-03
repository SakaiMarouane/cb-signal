# CB-Signal — Central Bank Sentiment Alpha

> Extracting a quantifiable investment signal from central bank language (ECB, Fed, BoE) and testing its predictive power on rates and equity styles.

## Motivation

Central bank communications drive expectations of monetary policy and, through them, the term structure of interest rates and cross-sectional equity behaviour. This project quantifies the *hawkish–dovish* tone of ECB, Fed and BoE communications and constructs a systematic signal tested on:

- **Primary:** DE / US / UK 2Y and 10Y rates.
- **Extension:** Fama-French Europe value-vs-growth rotation.

## Methodology

1. **Ingestion** — scrape ECB speeches (official CSV), Fed FOMC statements, minutes and speeches, BoE MPC minutes and speeches.
2. **Sentiment** — three progressive approaches:
   - Baseline dictionary (Loughran-McDonald + Apel & Blix Grimaldi).
   - Pre-trained FinBERT.
   - Fine-tuned sentence-transformer on a ~500-phrase hand-labelled hawkish/dovish set.
3. **Signal construction** — hawkish surprise (deviation from EMA), BERTopic thematic decomposition, HMM regime detection.
4. **Backtest** — point-in-time discipline, transaction costs, walk-forward, sub-period stability (GFC, ZIRP, COVID, hiking cycle).
5. **Reporting** — LaTeX working paper, beamer slides, Streamlit dashboard.

## Repository layout

```
cb-signal/
├── configs/           # Hydra experiment configs
├── data/
│   ├── raw/           # scraped, immutable (git-ignored)
│   ├── interim/       # cleaned intermediate
│   └── processed/     # feature-ready
├── src/cb_signal/
│   ├── ingest/        # ECB / Fed / BoE / rates scrapers
│   ├── nlp/           # sentiment models
│   ├── signals/       # signal construction
│   ├── backtest/      # engine and metrics
│   └── utils/
├── notebooks/         # ordered 01_, 02_, ... exploration
├── tests/
├── reports/
│   ├── paper/         # LaTeX article
│   └── slides/        # beamer
├── dashboard/         # Streamlit app
└── .github/workflows/ # CI
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -e ".[dev]"
```

## Status

End-to-end runnable on **real data**: ECB, Fed and BoE communications are scraped live (`python -m cb_signal.ingest`) and run through the full sentiment → signal → backtest pipeline. A synthetic mode (`python -m cb_signal.run_all`) is also kept for offline demos and CI, generating ECB/Fed/BoE-style documents with a latent hawkish/dovish tone tied to five named regimes.

### Pipeline stages (all complete)

| Stage | Module | Output |
|-------|--------|--------|
| Ingest (real or synth) | `cb_signal.ingest` / `cb_signal.synth` | `data/interim/corpus.parquet`, `data/raw/rates/rates.parquet` |
| Dictionary sentiment | `cb_signal.nlp.dictionary` | `data/processed/sentiment_dict.parquet` |
| Signal build (index, EMA, HMM, composite) | `cb_signal.signals.build` | `data/processed/signals.parquet` |
| Backtest (2Y, 10Y, 50/50, TC 20bps) | `cb_signal.backtest.engine` | `data/processed/equity_*.parquet`, `reports/tearsheet.csv` |
| Figures | `cb_signal.run_all` | `reports/figures/*.png` |

### Real corpus

1,997–2026, 3,968 documents scraped from the official sites: 1,871 ECB speeches, 1,649 Fed statements/minutes/speeches, 448 BoE speeches and MPC minutes. Rates panel is 2Y/10Y US Treasury, DE/UK long-term yields and VIX from FRED, 1956–2026.

### Results on real data

*Weekly, full real-history sample, 20 bps round-trip TC, T+1 execution.*

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Turnover/yr |
|---|---|---|---|---|---|
| Short 10Y | +0.07% | 0.92% | 0.07 | -6.1% | 2.2 |
| 50/50 (2Y + 10Y) | -0.24% | 0.60% | -0.40 | -10.0% | 2.9 |
| Short 2Y | -0.55% | 0.34% | -1.61 | -15.2% | 3.6 |

The baseline dictionary signal does not show a robust edge on real data — a meaningfully weaker result than on the synthetic generator (where tone is constructed to drive returns, Sharpe up to 1.75). This is the honest headline finding of the current baseline, not a bug: candidate next steps are FinBERT/fine-tuned sentiment (already present as scaffolding in `cb_signal.nlp`), richer signal construction, and walk-forward validation before drawing further conclusions.

### Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate       # Windows
pip install -e .

# Real data (scrapes ECB/Fed/BoE + FRED; takes 30-90 min, polite rate-limited)
python -m cb_signal.ingest
python -m cb_signal.run_all --skip-synth

# Or, offline synthetic demo (~10s)
python -m cb_signal.run_all
```

Figures land in `reports/figures/`.
