"""Polite HTTP helper shared by all scrapers.

Features:
    - Configurable User-Agent (scraper is identified).
    - Retry with exponential backoff on 5xx / connection errors.
    - Rate limiting (sleep between requests to the same host).
    - On-disk response cache keyed by URL so re-runs are cheap.
    - Transparent handling of both HTML and CSV endpoints.

The cache lives under `data/raw/.cache/` and is keyed by a hash of the URL.
Delete the folder to force a full re-fetch.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from loguru import logger

DEFAULT_UA = "cb-signal-research/0.1 (academic project; contact: local)"
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 2.0  # seconds; multiplied on each retry
DEFAULT_SLEEP = 0.5  # seconds between successive requests to same host


_last_hit: dict[str, float] = {}


def _cache_path(url: str, cache_dir: Path) -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()
    return cache_dir / h[:2] / f"{h}.bin"


def _sleep_polite(host: str, min_gap: float) -> None:
    now = time.monotonic()
    last = _last_hit.get(host, 0.0)
    delta = now - last
    if delta < min_gap:
        time.sleep(min_gap - delta)
    _last_hit[host] = time.monotonic()


def polite_get(
    url: str,
    *,
    cache_dir: Path | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    sleep_between: float = DEFAULT_SLEEP,
    user_agent: str = DEFAULT_UA,
    force: bool = False,
) -> bytes:
    """GET a URL with retries, rate limiting, and on-disk caching.

    Returns the raw response bytes. Raises after `retries` failed attempts.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = _cache_path(url, cache_dir)
        if cache_file.exists() and not force:
            return cache_file.read_bytes()

    host = urlparse(url).netloc
    headers = {"User-Agent": user_agent, "Accept": "*/*"}

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        _sleep_polite(host, sleep_between)
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 404:
                # not retryable and semantically meaningful
                resp.raise_for_status()
            if resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} on {url}")
            resp.raise_for_status()
            data = resp.content

            if cache_dir is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(data)
            return data
        except requests.HTTPError as e:
            # non-retryable client errors bubble up immediately
            if e.response is not None and 400 <= e.response.status_code < 500:
                raise
            last_exc = e
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e

        wait = backoff * attempt
        logger.warning(f"[{attempt}/{retries}] {url} failed: {last_exc}. Retrying in {wait:.1f}s")
        time.sleep(wait)

    assert last_exc is not None
    raise last_exc


def project_cache_dir(root: Path) -> Path:
    return root / "data" / "raw" / ".cache"
