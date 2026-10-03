from __future__ import annotations

from pathlib import Path

from cb_signal.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_config_reads_default_yaml():
    cfg = load_config(ROOT)
    assert cfg.sentiment.method == "dict"
    assert cfg.backtest.tc_bps == 20.0
    assert cfg.signals.hmm_n_states == 3


def test_load_config_applies_dotlist_overrides():
    cfg = load_config(ROOT, overrides=["backtest.tc_bps=5", "sentiment.method=finbert"])
    assert cfg.backtest.tc_bps == 5
    assert cfg.sentiment.method == "finbert"
    # untouched values stay at their file defaults
    assert cfg.signals.hmm_n_states == 3
