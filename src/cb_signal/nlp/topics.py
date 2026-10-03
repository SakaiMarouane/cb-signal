"""BERTopic thematic decomposition of the real corpus.

This is a descriptive/analysis addition, not a change to the tradable
signal: it answers "what is each document actually about" (inflation,
labour markets, financial stability, climate, ...) rather than feeding
`signals.build.build_signals`, which is unchanged. Reusing it to reweight
the trading signal would be a much larger, untested change and is out of
scope here.

Import-order note (read before touching this file)
----------------------------------------------------
On this project's Windows dev environment, importing `bertopic` (which
pulls in `hdbscan`/`umap`/`numba`) *before* `torch` corrupts a later
`torch` DLL load — it manifests as an unrelated-looking
`OSError: ... c10.dll ...` the first time anything imports `torch`
afterwards in the same process. Importing `torch` first avoids it
entirely. Every public function here does `import torch` (even though
BERTopic itself doesn't need it) as the first import specifically to force
that ordering — don't remove it even though it looks like dead code.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from cb_signal.nlp.preprocessing import clean_document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def fit_topics(
    corpus: pd.DataFrame,
    embedding_model: str = EMBEDDING_MODEL,
    min_topic_size: int = 15,
):
    """Fit BERTopic on `corpus["text"]`; returns (topic_model, topics, doc_info)."""
    import torch  # noqa: F401  (import order: see module docstring)
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    required = {"doc_id", "source", "event_type", "date", "text"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"corpus missing required columns: {missing}")

    docs = [clean_document(t) for t in corpus["text"]]
    logger.info(f"[TOPICS] embedding {len(docs)} documents with {embedding_model}")
    encoder = SentenceTransformer(embedding_model)
    embeddings = encoder.encode(docs, batch_size=64, show_progress_bar=False)

    # Use scikit-learn's own HDBSCAN rather than the standalone `hdbscan`
    # package: that package's 0.8.44 release calls
    # sklearn.utils.check_array(..., ensure_all_finite=...), a parameter
    # name that doesn't exist in the scikit-learn version pinned by this
    # project (ensure_all_finite only replaces force_all_finite in sklearn
    # >=1.6), so it raises TypeError on any fit(). sklearn's built-in
    # HDBSCAN avoids the cross-package version mismatch entirely.
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean")
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    logger.info(f"[TOPICS] fitting BERTopic (min_topic_size={min_topic_size})")
    topic_model = BERTopic(
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)

    doc_info = pd.DataFrame(
        {
            "doc_id": corpus["doc_id"].values,
            "source": corpus["source"].values,
            "event_type": corpus["event_type"].values,
            "date": corpus["date"].values,
            "topic": topics,
        }
    )
    logger.info(f"[TOPICS] found {topic_model.get_topic_info().shape[0] - 1} topics (excluding outliers)")
    return topic_model, doc_info


def topic_summary(topic_model, doc_info: pd.DataFrame, dict_sentiment: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per topic: size, top terms, and (if `dict_sentiment` is given)
    the mean dictionary net_tone of its documents."""
    info = topic_model.get_topic_info().rename(columns={"Count": "n_docs", "Name": "label"})
    info = info[info["Topic"] != -1].copy()  # drop the outlier bucket
    info["top_terms"] = info["Topic"].apply(
        lambda t: ", ".join(term for term, _ in topic_model.get_topic(t)[:8])
    )

    if dict_sentiment is not None:
        merged = doc_info.merge(dict_sentiment[["doc_id", "net_tone"]], on="doc_id", how="left")
        mean_tone = merged.groupby("topic")["net_tone"].mean()
        info["mean_net_tone"] = info["Topic"].map(mean_tone)

    return info[["Topic", "label", "n_docs", "top_terms"] + (["mean_net_tone"] if dict_sentiment is not None else [])]


def run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: read the real corpus, fit topics, persist outputs."""
    corpus_path = root / "data" / "interim" / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError(f"{corpus_path} not found. Run `python -m cb_signal.ingest` first.")
    corpus = pd.read_parquet(corpus_path)

    topic_model, doc_info = fit_topics(corpus)

    dict_sentiment_path = root / "data" / "processed" / "sentiment_dict.parquet"
    dict_sentiment = pd.read_parquet(dict_sentiment_path) if dict_sentiment_path.exists() else None
    summary = topic_summary(topic_model, doc_info, dict_sentiment)

    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_info.to_parquet(out_dir / "topics.parquet", index=False)
    summary.to_csv(out_dir / "topic_summary.csv", index=False)
    logger.info(f"[TOPICS] wrote {out_dir / 'topics.parquet'} and {out_dir / 'topic_summary.csv'}")
    return doc_info, summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    run(root)
