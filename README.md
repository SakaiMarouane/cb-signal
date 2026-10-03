# CB-Signal — Central Bank Sentiment Alpha

> Extracting a quantifiable investment signal from central bank language (ECB, Fed, BoE) and testing its predictive power on rates.

## Motivation

Central bank communications drive expectations of monetary policy and, through them, the term structure of interest rates. This project quantifies the *hawkish–dovish* tone of ECB, Fed and BoE communications and constructs a systematic signal tested on DE / US / UK 2Y and 10Y rates, on real data scraped directly from the issuing institutions.

## Methodology

1. **Ingestion** — scrape ECB speeches, Fed FOMC statements/minutes/speeches, BoE MPC minutes and speeches, directly from each institution's own site (no third-party aggregator).
2. **Sentiment** — three independent tiers, cross-checked against each other:
   - Dictionary (Loughran-McDonald uncertainty subset + Apel & Blix Grimaldi / Hansen & McMahon / Bennani & Neuenkirch hawkish-dovish lexicon), with negation handling.
   - Pretrained FinBERT (`ProsusAI/finbert`; the FOMC-specific `gtfintechlab/FOMC-RoBERTa` model is supported too but is access-gated on HuggingFace).
   - A fine-tuned tier: MiniLM sentence embeddings + a logistic-regression head, trained on a small **author-curated** (not independently human-annotated) example set — see `cb_signal.nlp.finetuned` for exactly how, stated plainly rather than oversold.
3. **Signal construction** — hawkish index → EMA baseline → expanding-window (point-in-time) z-scored surprise → tradable signal. A 3-state HMM labels dovish/neutral/hawkish regimes for descriptive/diagnostic purposes (it does not feed the tradable signal). BERTopic thematic decomposition is a separate descriptive layer on top of the same corpus.
4. **Backtest** — transaction costs, full-sample tearsheet, sub-period stability (GFC, ZIRP, COVID, hiking cycle, plateau), and a walk-forward (yearly expanding refit) version of the regime overlay.
5. **Reporting** — LaTeX working paper (`reports/paper/`), beamer slides (`reports/slides/`), Streamlit dashboard (`dashboard/`).

Every item above is actually implemented and runnable — see Status below for what's been run on real data versus what you run yourself.

## Repository layout

```
cb-signal/
├── configs/           # experiment config (sentiment method, TC, HMM, EMA — see cb_signal.config)
├── data/
│   ├── raw/           # scraped, immutable (git-ignored: data/raw/.cache/)
│   ├── interim/       # cleaned intermediate
│   └── processed/     # feature-ready
├── src/cb_signal/
│   ├── ingest/        # ECB / Fed / BoE / rates scrapers
│   ├── nlp/           # dictionary, finbert, finetuned, topics
│   ├── signals/        # signal construction (EMA, HMM, walk-forward)
│   ├── backtest/       # engine, metrics, sub-period stability
│   └── config.py       # YAML config loader
├── tests/
├── reports/
│   ├── paper/          # LaTeX working paper (cb_signal.pdf)
│   ├── slides/          # beamer companion deck
│   └── figures/
├── dashboard/           # Streamlit app
└── .github/workflows/   # CI (lint + test + synthetic smoke test)
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -e ".[dev]"
# optional, for FinBERT / fine-tuned tier / BERTopic / dashboard:
pip install -e ".[nlp,dashboard]"
```

## Status

End-to-end runnable on **real data**. All five methodology items above are implemented, tested, and have been run against the real corpus at least once (see Results below); re-running any of them yourself is a single command (see Reproduce). A synthetic mode (`python -m cb_signal.run_all`) is also kept for offline demos and CI, generating ECB/Fed/BoE-style documents with a latent hawkish/dovish tone.

### Real corpus

1997–2026, 3,968 documents scraped from the official sites: 1,871 ECB speeches, 1,649 Fed statements/minutes/speeches, 448 BoE speeches and MPC minutes. Rates panel is 2Y/10Y US Treasury, DE/UK long-term yields and VIX from FRED, 1956–2026. Getting this working required real fixes to all three scrapers: ECB and BoE's old listing pages moved to JavaScript-rendered search widgets (now discovered via ECB's AddSearch API and BoE's sitemap feed respectively), and the Fed's meeting-date discovery had a pre-existing bug that mislabelled minutes-release-day press releases as FOMC statements.

### Results: dictionary tier, full real corpus (1997–2026)

*Weekly, 20 bps round-trip TC, T+1 execution.*

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Turnover/yr |
|---|---|---|---|---|---|
| Short 10Y | +0.07% | 0.92% | 0.07 | -6.1% | 2.2 |
| 50/50 (2Y + 10Y) | -0.24% | 0.60% | -0.40 | -10.0% | 2.9 |
| Short 2Y | -0.55% | 0.34% | -1.61 | -15.2% | 3.6 |

No robust full-sample edge — a meaningfully weaker result than the synthetic generator (Sharpe up to 1.75), as expected since synthetic tone is constructed to drive returns. **Sub-period breakdown** (`reports/tearsheet_subperiods.csv`) tells a more interesting story: every regime is negative *except* the 2022-2023 hiking cycle, where all three legs turn solidly positive (50/50 Sharpe 1.83). See `reports/paper/cb_signal.pdf` for the full writeup, including the hypothesis this motivates and why it should be read as suggestive (one episode) rather than proven.

### Results: cross-tier comparison, 2021–2023 window

Running FinBERT/fine-tuned-tier on the full 3,968-doc corpus is impractical on CPU (~9h at this project's measured throughput); instead all three tiers are compared on the same **contiguous 2021–2023 window** (688 docs — a random subsample would break the weekly-aggregation continuity the signal needs), which covers the pre-hiking, hiking and plateau-start periods.

| Tier | Short 10Y Sharpe | 50/50 Sharpe | Short 2Y Sharpe |
|---|---|---|---|
| Dictionary | **1.21** | **0.93** | 0.02 |
| FinBERT | 0.03 | -0.34 | -1.16 |
| Fine-tuned | -0.63 | -1.36 | -2.89 |

On this window the purpose-built dictionary lexicon beats both learned tiers, not the other way around — FinBERT is trained on general financial-news sentiment rather than central-bank language specifically, and the fine-tuned tier's training set is small and templated (see Methodology). Worth taking at face value rather than assuming "more ML" implies "better signal" here.

Full tearsheets: `reports/tearsheet_{dict,finbert,transformer}_window2123.csv`.

### Walk-forward regime check

The original regime-labelling HMM was fit once on the full history (a look-ahead issue for that descriptive label, though it does not affect backtest P&L — the tradable signal is built only from already-point-in-time EMA/z-score components). A walk-forward version (`cb_signal.signals.build.fit_regime_hmm_walkforward`, yearly expanding refit) changes 45% of weekly regime labels — see `reports/figures/02_regime_walkforward.png` vs `02_regime.png`.

### BERTopic

Fit on the full real corpus: 5 interpretable topics found (euro-area monetary policy, Fed inflation/labour, financial-stability/banking, FOMC-participant language, BoE policy) — see `data/processed/topic_summary.csv`. ~76% of documents land in the outlier bucket (HDBSCAN's default behaviour on a long, stylistically diverse corpus); reported as-is rather than tuned to look tidier.

### Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate       # Windows
pip install -e ".[dev]"

# Real data (scrapes ECB/Fed/BoE + FRED; takes 30-90 min, polite rate-limited)
python -m cb_signal.ingest
python -m cb_signal.run_all --skip-synth --subperiods --walk-forward

# Other sentiment tiers (slower; needs `pip install -e ".[nlp]"`)
python -m cb_signal.run_all --skip-synth --sentiment finbert
python -m cb_signal.run_all --skip-synth --sentiment transformer

# BERTopic
python -m cb_signal.nlp.topics

# Dashboard
streamlit run dashboard/app.py

# Or, offline synthetic demo (~10s)
python -m cb_signal.run_all
```

Figures land in `reports/figures/`. Parameters (sentiment method, EMA half-life, HMM settings, transaction costs, duration assumptions) are read from `configs/default.yaml`; override with `--config-override key.path=value`.

### Tests & CI

```bash
pytest -q        # 46+ tests, pure-Python/no network
ruff check .
```

GitHub Actions (`.github/workflows/ci.yml`) runs both plus a synthetic end-to-end smoke test on every push/PR.
