"""A small fine-tuned sentiment tier: sentence embeddings + a linear head.

Data provenance (read this before citing this as a research dataset)
----------------------------------------------------------------------
There is no human-annotated hawkish/dovish corpus in this project. The
"training set" built by `build_training_examples()` is templated: each
phrase from `cb_signal.nlp.lexicons.HAWKISH` / `DOVISH` is dropped into a
handful of short carrier sentences to make it look like natural prose, and
a separate list of genuinely neutral central-bank boilerplate sentences is
written by hand to cover the neutral class. This is a legitimate way to
bootstrap a lightweight classifier from an existing curated lexicon, but it
is author-curated, not independently annotated — the paper and README
describe it this way rather than as a "hand-labelled" research dataset.

Why a linear head on frozen embeddings, not full fine-tuning
--------------------------------------------------------------
With ~500 examples, backpropagating through a whole transformer would
almost certainly overfit and destabilise on CPU. Embedding each example
with a small pretrained sentence-transformer (`all-MiniLM-L6-v2`, 384-dim,
~80MB) and fitting a multinomial logistic regression on top ("linear
probing") is the standard, well-behaved choice for this data regime, and
is what "fine-tuned" means in this module — not end-to-end weight updates.

Output columns (written to `data/processed/sentiment_transformer.parquet`):

    doc_id, source, event_type, date,
    p_hawkish, p_dovish, p_neutral,
    label, tone
    (tone = p_hawkish - p_dovish, in [-1, 1])
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from cb_signal.nlp.lexicons import DOVISH, HAWKISH
from cb_signal.nlp.preprocessing import sentence_split

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_HAWK_TEMPLATES = [
    "The Committee noted {phrase} in its assessment of the outlook.",
    "Members pointed to {phrase} when discussing the economy.",
    "The statement referred to {phrase} as a key consideration.",
    "Policymakers highlighted {phrase} going forward.",
]
_DOVE_TEMPLATES = _HAWK_TEMPLATES  # same carrier sentences, opposite phrase list

# Hand-written, genuinely neutral central-bank boilerplate: procedural or
# administrative sentences that carry no hawkish/dovish tone.
_NEUTRAL_EXAMPLES = [
    "The meeting was attended by all members of the Committee.",
    "The next policy meeting is scheduled for the following month.",
    "Minutes of the meeting will be published in three weeks.",
    "The Chair opened the session and reviewed the agenda.",
    "Staff presented the latest economic projections to the Committee.",
    "The Committee thanked staff for their analysis.",
    "Members discussed procedural matters relating to communication.",
    "The press conference followed the policy announcement.",
    "Committee members reviewed data compiled since the previous meeting.",
    "The decision was taken by a vote among eligible members.",
    "A transcript of the press conference is available on the website.",
    "The Committee's mandate is set out in the founding treaty.",
    "Today I would like to thank the organisers for their invitation.",
    "It is a pleasure to be here with you this afternoon.",
    "Let me now turn to the second part of my remarks.",
    "I will structure my remarks around three main themes.",
    "The annual report sets out the institution's activities in detail.",
    "The next scheduled publication date is listed on the calendar.",
    "Representatives from member countries took part in the discussion.",
    "The working group will report back at the next meeting.",
    "This concludes my prepared remarks; I am happy to take questions.",
    "The conference was held jointly with several partner institutions.",
    "I would like to begin by describing the structure of this talk.",
    "The data referenced in this speech are publicly available online.",
    "The secretariat will circulate a summary after the meeting.",
    "Attendance at today's event exceeded expectations.",
    "The agenda included three items for discussion.",
    "I will now hand over to my colleague for the second presentation.",
    "The committee's composition rotates according to established rules.",
    "A revised version of the report will be issued next quarter.",
]

LABELS = ("hawkish", "dovish", "neutral")


@dataclass(frozen=True)
class TrainingExample:
    text: str
    label: str


def build_training_examples() -> list[TrainingExample]:
    """Build the templated + hand-written example set described above."""
    examples: list[TrainingExample] = []
    for phrase in HAWKISH:
        for tmpl in _HAWK_TEMPLATES:
            examples.append(TrainingExample(tmpl.format(phrase=phrase), "hawkish"))
    for phrase in DOVISH:
        for tmpl in _DOVE_TEMPLATES:
            examples.append(TrainingExample(tmpl.format(phrase=phrase), "dovish"))
    for sent in _NEUTRAL_EXAMPLES:
        examples.append(TrainingExample(sent, "neutral"))
    return examples


def write_training_examples_csv(root: Path) -> Path:
    """Persist the training set to disk for transparency/inspection."""
    out_path = root / "data" / "raw" / "labels" / "hawkish_dovish_examples.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    examples = build_training_examples()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for ex in examples:
            writer.writerow([ex.text, ex.label])
    logger.info(f"[FINETUNED] wrote {len(examples)} training examples to {out_path}")
    return out_path


def _model_dir(root: Path) -> Path:
    return root / "data" / "models" / "finetuned_classifier"


def train(root: Path, embedding_model: str = EMBEDDING_MODEL):
    """Embed the training examples and fit a logistic-regression head.

    Persists the classifier (joblib) and the embedding model name under
    `data/models/finetuned_classifier/` and returns the fitted classifier.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    examples = build_training_examples()
    write_training_examples_csv(root)

    logger.info(f"[FINETUNED] embedding {len(examples)} training examples with {embedding_model}")
    encoder = SentenceTransformer(embedding_model)
    X = encoder.encode([e.text for e in examples], batch_size=64, show_progress_bar=False)
    y = [e.label for e in examples]

    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    logger.info(f"[FINETUNED] trained logistic-regression head; train accuracy={train_acc:.3f}")

    import joblib

    model_dir = _model_dir(root)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "classifier.joblib")
    (model_dir / "embedding_model.txt").write_text(embedding_model, encoding="utf-8")
    logger.info(f"[FINETUNED] saved classifier to {model_dir}")
    return clf


def _load(root: Path):
    import joblib
    from sentence_transformers import SentenceTransformer

    model_dir = _model_dir(root)
    classifier_path = model_dir / "classifier.joblib"
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"{classifier_path} not found. Run `cb_signal.nlp.finetuned.train(root)` first."
        )
    embedding_model = (model_dir / "embedding_model.txt").read_text(encoding="utf-8").strip()
    clf = joblib.load(classifier_path)
    encoder = SentenceTransformer(embedding_model)
    return encoder, clf


def score_corpus(root: Path, corpus: pd.DataFrame, batch_size: int = 128) -> pd.DataFrame:
    """Score every document by averaging sentence-level classifier output.

    Trains the classifier on first use if it hasn't been trained yet.
    """
    required = {"doc_id", "source", "event_type", "date", "text"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"corpus missing required columns: {missing}")

    if not (_model_dir(root) / "classifier.joblib").exists():
        logger.info("[FINETUNED] no trained classifier found; training now")
        train(root)
    encoder, clf = _load(root)
    class_order = list(clf.classes_)

    logger.info(f"[FINETUNED] scoring {len(corpus)} documents")

    # Split every document into sentences, keep a doc_id back-reference, and
    # embed everything in one batched pass for speed.
    all_sentences: list[str] = []
    owner: list[int] = []  # index into corpus.itertuples() position
    for i, text in enumerate(corpus["text"]):
        sents = sentence_split(text) or [text[:500]]
        all_sentences.extend(sents)
        owner.extend([i] * len(sents))

    embeddings = encoder.encode(
        all_sentences, batch_size=batch_size, show_progress_bar=False
    )
    probs = clf.predict_proba(embeddings)  # (n_sentences, n_classes) in class_order

    n_docs = len(corpus)
    sums = np.zeros((n_docs, len(class_order)))
    counts = np.zeros(n_docs)
    owner_arr = np.asarray(owner)
    for j in range(len(class_order)):
        np.add.at(sums[:, j], owner_arr, probs[:, j])
    np.add.at(counts, owner_arr, 1)
    counts[counts == 0] = 1
    doc_probs = sums / counts[:, None]

    idx = {name: class_order.index(name) for name in LABELS}
    records = []
    for i, row in enumerate(corpus.itertuples(index=False)):
        p_hawkish = float(doc_probs[i, idx["hawkish"]])
        p_dovish = float(doc_probs[i, idx["dovish"]])
        p_neutral = float(doc_probs[i, idx["neutral"]])
        tone = p_hawkish - p_dovish
        label = max(
            (("hawkish", p_hawkish), ("dovish", p_dovish), ("neutral", p_neutral)),
            key=lambda kv: kv[1],
        )[0]
        records.append(
            {
                "doc_id": row.doc_id,
                "source": row.source,
                "event_type": row.event_type,
                "date": row.date,
                "p_hawkish": p_hawkish,
                "p_dovish": p_dovish,
                "p_neutral": p_neutral,
                "tone": tone,
                "label": label,
                "model": EMBEDDING_MODEL,
            }
        )

    out = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    logger.info(f"[FINETUNED] finished; mean tone = {out['tone'].mean():+.3f}")
    return out


def run(root: Path) -> pd.DataFrame:
    """Convenience wrapper: read consolidated corpus, score, persist."""
    corpus_path = root / "data" / "interim" / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. Run `python -m cb_signal.ingest` first."
        )
    corpus = pd.read_parquet(corpus_path)
    out = score_corpus(root, corpus)
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sentiment_transformer.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"[FINETUNED] wrote {len(out)} rows to {out_path}")
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    run(root)
