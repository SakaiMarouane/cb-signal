"""CB-Signal interactive dashboard.

Run with: streamlit run dashboard/app.py

Reads whatever pipeline outputs already exist under data/ and reports/ —
every section degrades gracefully (shows a note instead of crashing) if a
given artifact hasn't been produced yet, e.g. `python -m cb_signal.ingest`
and `python -m cb_signal.run_all --skip-synth [--subperiods]` haven't all
been run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

REGIME_LABELS = {0: "Dovish", 1: "Neutral", 2: "Hawkish"}
REGIME_COLORS = {0: "rgba(74,122,184,0.25)", 1: "rgba(187,187,187,0.25)", 2: "rgba(201,86,78,0.25)"}

st.set_page_config(page_title="CB-Signal Dashboard", layout="wide")


@st.cache_data
def load_parquet(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


corpus = load_parquet(ROOT / "data" / "interim" / "corpus.parquet")

st.title("CB-Signal — Central Bank Sentiment Alpha")
st.caption(
    "Real ECB/Fed/BoE communications, three independent sentiment tiers, "
    "a hawkish-index signal, and a backtest against 2Y/10Y rates."
)

if corpus is None:
    st.error(
        "No corpus found at data/interim/corpus.parquet. Run "
        "`python -m cb_signal.ingest` (real data) or "
        "`python -m cb_signal.run_all` (synthetic demo) first."
    )
    st.stop()

tab_corpus, tab_signal, tab_backtest, tab_topics = st.tabs(
    ["Corpus", "Signal & Regimes", "Backtest", "Topics"]
)

# --------------------------------------------------------------------------
# Corpus browser
# --------------------------------------------------------------------------
with tab_corpus:
    st.subheader(f"{len(corpus):,} documents, {corpus['date'].min()} to {corpus['date'].max()}")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        sources = st.multiselect("Source", sorted(corpus["source"].unique()), default=list(corpus["source"].unique()))
    with col2:
        event_types = st.multiselect(
            "Event type", sorted(corpus["event_type"].unique()), default=list(corpus["event_type"].unique())
        )
    with col3:
        query = st.text_input("Search title/speaker (case-insensitive)")

    filtered = corpus[corpus["source"].isin(sources) & corpus["event_type"].isin(event_types)]
    if query:
        mask = (
            filtered["title"].str.contains(query, case=False, na=False)
            | filtered["speaker"].str.contains(query, case=False, na=False)
        )
        filtered = filtered[mask]

    st.dataframe(
        filtered[["date", "source", "event_type", "speaker", "title", "n_tokens"]].sort_values(
            "date", ascending=False
        ),
        use_container_width=True,
        height=420,
    )

    if len(filtered) == 1:
        st.text_area("Full text", filtered.iloc[0]["text"], height=300)

# --------------------------------------------------------------------------
# Signal & regimes
# --------------------------------------------------------------------------
with tab_signal:
    method = st.selectbox(
        "Sentiment tier",
        ["dict", "finbert", "transformer"],
        format_func=lambda m: {"dict": "Dictionary", "finbert": "FinBERT", "transformer": "Fine-tuned"}[m],
    )
    suffix = "" if method == "dict" else f"_{method}"
    signals = load_parquet(ROOT / "data" / "processed" / f"signals{suffix}.parquet")

    if signals is None:
        st.warning(
            f"No signals found for method={method}. Run "
            f"`python -m cb_signal.run_all --skip-synth --sentiment {method}` first."
        )
    else:
        df = signals.copy()
        df["date"] = pd.to_datetime(df["date"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["hawk_index"], name="Hawk index", line=dict(color="#999", width=1)))
        fig.add_trace(go.Scatter(x=df["date"], y=df["hawk_ema"], name="EMA baseline", line=dict(color="crimson", width=1.5)))
        for state, label in REGIME_LABELS.items():
            mask = df["regime"] == state
            if not mask.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, "date"],
                    y=df.loc[mask, "hawk_index"],
                    mode="markers",
                    name=label,
                    marker=dict(color=REGIME_COLORS[state].replace("0.25", "0.9"), size=6),
                )
            )
        fig.update_layout(height=450, title=f"Hawkish index — {method} tier", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Regime shading uses the in-sample HMM fit by default; see "
            "reports/figures/02_regime_walkforward*.png for the point-in-time version "
            "(45% of weeks differ — see the paper, Section 3.3)."
        )

# --------------------------------------------------------------------------
# Backtest comparison across sentiment tiers
# --------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Full-sample tearsheet by sentiment tier")
    rows = []
    for m, label in [("dict", "Dictionary"), ("finbert", "FinBERT"), ("transformer", "Fine-tuned")]:
        suffix = "" if m == "dict" else f"_{m}"
        ts = load_csv(ROOT / "reports" / f"tearsheet{suffix}.csv")
        if ts is not None:
            ts = ts.copy()
            ts.insert(0, "tier", label)
            rows.append(ts)
    if rows:
        st.dataframe(pd.concat(rows, ignore_index=True), use_container_width=True)
    else:
        st.warning("No tearsheets found yet.")

    sub = load_csv(ROOT / "reports" / "tearsheet_subperiods.csv")
    if sub is not None:
        st.subheader("Sub-period stability (dictionary tier)")
        st.dataframe(sub, use_container_width=True)
    else:
        st.caption("Run with `--subperiods` to see sub-period stability here.")

    equity_combined = load_parquet(ROOT / "data" / "processed" / "equity_COMBINED.parquet")
    if equity_combined is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_combined.index, y=equity_combined["equity"], name="50/50 equity"))
        fig.update_layout(height=350, title="Combined (50/50) equity curve, dict tier", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Topics
# --------------------------------------------------------------------------
with tab_topics:
    summary = load_csv(ROOT / "data" / "processed" / "topic_summary.csv")
    if summary is None:
        st.warning("No topic summary found. Run `python -m cb_signal.nlp.topics` first.")
    else:
        st.subheader("BERTopic thematic decomposition")
        st.dataframe(summary, use_container_width=True)
        fig = go.Figure(go.Bar(x=summary["label"], y=summary["n_docs"]))
        fig.update_layout(height=400, title="Documents per topic", xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
