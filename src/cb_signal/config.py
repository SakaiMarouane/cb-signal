"""Experiment configuration loader (Hydra/OmegaConf-backed).

Keeps `configs/default.yaml` as the single source of truth for parameters
that `cb_signal.run_all` would otherwise hardcode (sentiment method, EMA
halflife, HMM settings, transaction costs, duration assumptions). This is a
thin OmegaConf loader rather than a full `@hydra.main` application, so it
composes with the existing argparse CLI in `run_all.py` instead of replacing
it: config values become the defaults, and explicit CLI flags still win.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG_PATH = Path("configs") / "default.yaml"


def load_config(root: Path, config_path: Path | None = None, overrides: list[str] | None = None) -> DictConfig:
    """Load the YAML config, applying optional `key.path=value` dotlist overrides."""
    path = root / (config_path or DEFAULT_CONFIG_PATH)
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg
