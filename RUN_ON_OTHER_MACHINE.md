# Running the ingest pipeline on another machine

The primary developer machine has network restrictions that block the
central bank websites. This file explains, step-by-step, how to run the
Phase A ingest on a laptop with unrestricted internet access and bring
the data back to the main working environment.

Everything described here uses **public sources only**.

---

## 1. Prerequisites on the runner machine

* Python 3.11 or 3.12 (64-bit).
* Internet access to `www.ecb.europa.eu`, `www.federalreserve.gov`,
  `www.bankofengland.co.uk`, `fred.stlouisfed.org`.
* ~500 MB free disk (raw HTML cache is the largest artefact).
* Git *(optional)* to clone / pull the code cleanly.

No GPU, no database, no cloud account.

## 2. Transferring the code

Zip the entire `cb-signal/` folder from the main machine and copy it over
(USB stick, cloud drive, GitHub — whatever works). Skip these folders,
they will be regenerated:

* `.venv/`
* `data/raw/`, `data/interim/`, `data/processed/`
* `__pycache__/`, `.pytest_cache/`

## 3. Install

Open a terminal in the copied folder:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install -e .
```

The base install (~50 MB) is enough for scraping. You do **not** need the
`[nlp]` extras on the runner machine — those are only used later for
sentiment scoring on your main machine.

Sanity check:

```bash
pytest tests/ -q
```

All tests should pass. If they don't, stop and report the failure before
scraping.

## 4. Run the ingest

Full run (all four sources + consolidation):

```bash
python -m cb_signal.ingest
```

Expected wall-clock time on a stable connection:

| Source        | Approx. duration | Approx. size |
|---------------|------------------|--------------|
| ECB speeches  | < 1 min          | ~10 MB       |
| Fed statements + minutes | 5-10 min | ~5 MB   |
| Fed speeches (2006+)     | 45-90 min | ~200 MB (raw HTML cache) |
| BoE MPC + speeches       | 20-40 min | ~100 MB |
| FRED rates               | < 1 min   | ~1 MB   |

The scraper is polite: half a second between requests to the same host,
retries with exponential backoff, on-disk cache under
`data/raw/.cache/`. Re-running is idempotent — cached pages are reused
and only new ones are fetched.

### Faster / smaller runs

Skip speeches (only statements and minutes, which are the highest-signal
documents anyway):

```bash
python -m cb_signal.ingest --no-speeches
```

Only one source:

```bash
python -m cb_signal.ingest --sources ecb rates
```

Fed speeches only from 2015 onward:

```bash
python -m cb_signal.ingest --fed-speech-start-year 2015
```

## 5. What you'll produce

After a successful full run:

```
data/
├── raw/
│   ├── .cache/          # HTML cache (can be discarded)
│   ├── ecb/speeches.parquet
│   ├── fed/corpus.parquet
│   ├── boe/corpus.parquet
│   └── rates/rates.parquet
└── interim/
    └── corpus.parquet   # merged canonical corpus (this is the deliverable)
```

The single file you must bring back is
**`data/interim/corpus.parquet`**. It contains all documents with the
canonical schema (`doc_id`, `source`, `event_type`, `date`, `speaker`,
`title`, `subtitle`, `url`, `text`, `n_tokens`) and is the input for
every downstream stage.

Also bring back **`data/raw/rates/rates.parquet`** for the backtest.

If disk allows, zip the entire `data/raw/` folder as a backup — the raw
per-source parquets are useful for debugging any bad row later.

## 6. Bringing it back

Copy `data/interim/corpus.parquet` (and ideally `data/raw/rates/`) back
to the main development machine, into the same paths under the
`cb-signal/` project directory.

Quick sanity check on the main machine:

```python
import pandas as pd
df = pd.read_parquet("data/interim/corpus.parquet")
print(df.groupby(["source", "event_type"]).size())
print(df["date"].min(), "->", df["date"].max())
```

Expected orders of magnitude:

* ECB: ~2,000 speeches
* Fed: ~250 statements, ~250 minutes, ~2,000-3,000 speeches (2006+)
* BoE: ~200 MPC packs, ~1,500-2,500 speeches

Numbers will vary as archives grow.

## 7. Troubleshooting

**Corporate proxy blocks HTTPS.** Configure `HTTPS_PROXY` before running:

```bash
# Windows
set HTTPS_PROXY=http://your.proxy:8080
set HTTP_PROXY=http://your.proxy:8080

# macOS / Linux
export HTTPS_PROXY=http://your.proxy:8080
export HTTP_PROXY=http://your.proxy:8080
```

**A single URL returns 404.** Non-fatal — the loop logs a warning and
continues. Fed and BoE occasionally move pages; a missing minute or
speech will not stop the run.

**Fed calendar page format changed.** If `discover_fomc_meetings`
returns 0 meetings, edit `_DATE_RE` in `src/cb_signal/ingest/fed.py`.

**BoE listing never terminates.** The walker stops after two
consecutive empty pages *or* after page 200. If that limit is hit,
increase it in `boe.py::_discover_listing`.

**Everything is slow.** Set `sleep_between=0.1` in `_http.py`.
Do this only if the host is not returning 429/503.
