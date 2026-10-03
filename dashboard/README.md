# CB-Signal dashboard

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

Browses the real corpus, plots the hawkish index with regime overlay per
sentiment tier, compares tearsheets (dictionary / FinBERT / fine-tuned) and
sub-period stability, and shows the BERTopic thematic breakdown. Every
section falls back to a note instead of crashing if the underlying pipeline
step hasn't been run yet — see the main README for the ingest/run_all
commands that produce each artifact.
