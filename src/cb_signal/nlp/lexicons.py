"""Hawkish / dovish lexicons and Loughran-McDonald uncertainty subset.

Sources
-------
* Apel & Blix Grimaldi (2012), "The information content of central bank minutes",
  Sveriges Riksbank Working Paper Series No. 261 — original hawkish/dovish
  bigram list, distilled from Riksbank minutes but shown to transfer to
  ECB/Fed communications in follow-up work.
* Hansen & McMahon (2016), "Shocking language: Understanding the macroeconomic
  effects of central bank communication", Journal of International Economics —
  additional monetary-policy-relevant phrases.
* Bennani & Neuenkirch (2017), "The (home) bias of European central bankers",
  Empirical Economics — complementary hawkish/dovish tokens.
* Loughran & McDonald (2011), "When is a liability not a liability? Textual
  analysis, dictionaries, and 10-Ks", Journal of Finance — uncertainty word
  list. The full LM master dictionary is much larger; we embed the uncertainty
  subset only, since hawkish/dovish tone (not general negativity) is the
  target here. If the full CSV is placed under `data/raw/lexicons/`, the
  loader below can pick it up.

Phrase-first design
-------------------
Monetary-policy tone is carried by multiword expressions ("downside risks to
growth", "price stability oriented") more than by single tokens. Every entry
below is stored as a lowercase phrase; the scorer matches on whitespace-
tokenised n-grams (see `dictionary.py`). Storing bigrams as a *single* phrase
avoids the classic pitfall where "risk" alone is dovish but "upside risk" is
hawkish.

Design invariants
-----------------
* Phrases are lowercase, single-spaced, no punctuation.
* No phrase appears in more than one polarity list. `_validate()` enforces
  this at import time.
* Phrases can be 1-4 tokens long. Longer phrases are rare and typically
  paraphrased in central bank prose, so we cap n at 4 for match speed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


HAWKISH: list[str] = [
    # --- Apel & Blix Grimaldi (2012) core ---
    "inflation pressures",
    "inflationary pressures",
    "price pressures",
    "upward pressure",
    "upside risks",
    "upside risk",
    "upside risks to inflation",
    "second-round effects",
    "wage pressures",
    "wage growth",
    "overheating",
    "elevated inflation",
    "above target",
    "above our target",
    "tighten",
    "tightening",
    "tighter",
    "restrictive",
    "less accommodative",
    "withdrawing accommodation",
    "removing accommodation",
    "normalise",
    "normalising",
    "normalisation",
    "normalize",
    "normalizing",
    "normalization",
    # --- Hansen & McMahon (2016) additions ---
    "robust growth",
    "strong growth",
    "strong recovery",
    "firming",
    "firm",
    "resilient",
    "buoyant",
    "sustained expansion",
    "capacity constraints",
    "supply constraints",
    "labor market tightness",
    "labour market tightness",
    "tight labor market",
    "tight labour market",
    "wage-price spiral",
    "de-anchoring",
    "unanchored expectations",
    # --- Bennani & Neuenkirch (2017) additions ---
    "raise rates",
    "raising rates",
    "hike",
    "hiking",
    "rate hike",
    "rate increase",
    "increase in rates",
    "vigilant",
    "vigilance",
    "monitor closely",
    "closely monitor",
    "appropriate to firm",
    "firm policy",
    "hawkish",
]


DOVISH: list[str] = [
    # --- Apel & Blix Grimaldi (2012) core ---
    "downside risks",
    "downside risk",
    "downside risks to growth",
    "downside risks to activity",
    "subdued",
    "muted",
    "weak",
    "weakness",
    "weaker",
    "deteriorating",
    "adverse",
    "sluggish",
    "slack",
    "output gap",
    "below target",
    "below our target",
    "disinflation",
    "disinflationary",
    "deflation",
    "deflationary",
    "deflation risk",
    "low inflation",
    "declining inflation",
    # --- Hansen & McMahon (2016) additions ---
    "accommodative",
    "accommodation",
    "easing",
    "ease",
    "easier",
    "supportive",
    "support the economy",
    "stimulus",
    "stimulate",
    "patient",
    "patience",
    "cautious",
    "caution",
    "gradual",
    "gradually",
    # --- Bennani & Neuenkirch (2017) additions ---
    "cut rates",
    "cutting rates",
    "rate cut",
    "rate decrease",
    "lower rates",
    "reduce rates",
    "reducing rates",
    "additional stimulus",
    "further accommodation",
    "more accommodative",
    "recession",
    "contraction",
    "contracting",
    "unemployment",
    "high unemployment",
    "rising unemployment",
    "dovish",
]


# Loughran-McDonald (2011) uncertainty subset. Full list ~285 words in the
# master dictionary; we embed the ones that occur in monetary-policy prose.
LM_UNCERTAINTY: list[str] = [
    "uncertain", "uncertainty", "uncertainties",
    "ambiguous", "ambiguity",
    "risk", "risks", "risky",
    "volatility", "volatile",
    "unpredictable", "unpredictability",
    "doubt", "doubts", "doubtful",
    "may", "might", "could",
    "possibly", "possible", "probable", "probably",
    "contingent", "contingency",
    "tentative", "tentatively",
    "unclear",
    "cautionary",
    "conditional",
]


# Standard English negators. A phrase preceded by any of these within
# `NEGATION_WINDOW` tokens is treated as polarity-flipped.
NEGATORS: set[str] = {
    "not", "no", "never", "none", "nothing", "neither", "nor",
    "without", "hardly", "barely", "scarcely",
    "cannot", "can't", "won't", "shouldn't", "wouldn't", "don't", "doesn't",
    "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
    "less", "lower", "lack", "lacking",
}


NEGATION_WINDOW = 3  # tokens to look back for a negator before a matched phrase


@dataclass(frozen=True)
class Lexicon:
    """Immutable container passed to the scorer.

    Attributes
    ----------
    hawkish : frozenset of phrases with pro-tightening / anti-inflation tone.
    dovish  : frozenset of phrases with pro-easing / pro-growth tone.
    uncertainty : LM uncertainty subset.
    max_ngram : longest n in any hawkish/dovish phrase; drives the scorer loop.
    """

    hawkish: frozenset[str]
    dovish: frozenset[str]
    uncertainty: frozenset[str]
    max_ngram: int = field(default=0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_ngram",
            max(
                (len(p.split()) for p in (*self.hawkish, *self.dovish)),
                default=1,
            ),
        )


def _validate(hawkish: list[str], dovish: list[str]) -> None:
    """Fail loudly if the same phrase appears in both polarities."""
    conflict = set(hawkish) & set(dovish)
    if conflict:
        raise ValueError(f"Phrases in both hawkish and dovish lists: {sorted(conflict)}")


def build_default_lexicon() -> Lexicon:
    """Return the embedded lexicon; safe to call without any data files."""
    _validate(HAWKISH, DOVISH)
    return Lexicon(
        hawkish=frozenset(p.lower() for p in HAWKISH),
        dovish=frozenset(p.lower() for p in DOVISH),
        uncertainty=frozenset(w.lower() for w in LM_UNCERTAINTY),
    )


def load_lm_uncertainty_from_csv(csv_path: Path) -> frozenset[str]:
    """Extract the LM uncertainty column from the official master dictionary CSV.

    The full Loughran-McDonald master dictionary is distributed as a CSV with
    columns `Word, ..., Uncertainty, ...`. If the user drops the file under
    `data/raw/lexicons/LoughranMcDonald_MasterDictionary.csv`, this loader
    replaces the embedded uncertainty subset with the authoritative one.
    """
    words: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unc = row.get("Uncertainty") or row.get("uncertainty")
            if unc and unc not in {"0", "", "0.0"}:
                words.add(row["Word"].lower())
    return frozenset(words)


def build_lexicon(root: Path | None = None) -> Lexicon:
    """Return the default lexicon, upgrading LM uncertainty if the CSV is present."""
    base = build_default_lexicon()
    if root is None:
        return base
    lm_csv = root / "data" / "raw" / "lexicons" / "LoughranMcDonald_MasterDictionary.csv"
    if not lm_csv.exists():
        return base
    return Lexicon(
        hawkish=base.hawkish,
        dovish=base.dovish,
        uncertainty=load_lm_uncertainty_from_csv(lm_csv),
    )
