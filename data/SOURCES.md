# Data sources

Reference sheet for every dataset touched by CB-Signal. All sources are public and free.

## Central bank communications

### ECB — European Central Bank
- **Speeches** — official pipe-delimited CSV, updated after each new speech.
  URL: `https://www.ecb.europa.eu/press/key/date/html/all_speeches.en.csv`
  Coverage: 1997 → today, ~2000 documents.
  Fields: date, speakers, title, subtitle, contents.
  Access: single HTTP GET.

### Fed — Federal Reserve
- **FOMC statements** — one file per meeting since 1994, HTML.
  URL pattern: `https://www.federalreserve.gov/newsevents/pressreleases/monetary{yyyymmdd}a.htm`
  Listing: `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm`
- **FOMC minutes** — released 3 weeks after each meeting.
  URL pattern: `https://www.federalreserve.gov/monetarypolicy/fomcminutes{yyyymmdd}.htm`
- **Speeches** — listed by year at
  `https://www.federalreserve.gov/newsevents/speeches.htm` (annual archive pages).

### BoE — Bank of England
- **MPC minutes** — released alongside each Monetary Policy Report.
  URL: `https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes`
- **Speeches** — listed at `https://www.bankofengland.co.uk/news/speeches`.

## Market data

### Rates
- **US 2Y / 10Y treasury yields** — FRED series `DGS2`, `DGS10`.
- **DE 2Y / 10Y bund yields** — Bundesbank Statistics or FRED (`IRLTLT01DEM156N`).
- **UK 2Y / 10Y gilt yields** — BoE statistical database.

### Rate expectations
- **OIS surprise data** — proxied via 2Y yield change on FOMC / ECB days
  (event-window 1-day move as monetary policy surprise).

## Equity data (extension)

- **Fama-French Europe factors** — Kenneth French data library, monthly and daily.
  URL: `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`
- **VIX / MOVE / VSTOXX** — Yahoo Finance or FRED for control variables.

## Lexicons

- **Loughran-McDonald master dictionary** — financial-domain sentiment lexicon.
  URL: `https://sraf.nd.edu/loughranmcdonald-master-dictionary/`
- **Apel & Blix Grimaldi (2012) hawkish/dovish lexicon** — from the paper's appendix,
  augmented with terms from Hansen & McMahon (2016).

## Storage layout

All raw sources land under `data/raw/<source>/`. Cleaned canonical corpus lives in
`data/interim/corpus.parquet` with the schema documented in
`src/cb_signal/ingest/schema.py`.
