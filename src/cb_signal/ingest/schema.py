"""Shared schema definitions and helpers for the ingest pipelines."""

from __future__ import annotations

from typing import Literal

Source = Literal["ECB", "FED", "BOE"]
EventType = Literal["speech", "statement", "minutes", "press_conference"]

CANONICAL_COLUMNS: list[str] = [
    "doc_id",
    "source",
    "event_type",
    "date",
    "speaker",
    "title",
    "subtitle",
    "url",
    "text",
    "n_tokens",
]
