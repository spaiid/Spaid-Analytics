"""Shared HTTP layer: polite rate limiting, retries, and an on-disk cache.

Every external source in this app goes through `fetch`. That keeps the
politeness rules (SEC's 10 req/s, Stooq's daily cap) in one place instead of
scattered across adapters, and makes reruns cheap because responses are cached
to disk keyed by URL.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import httpx

from spaid.config.settings import CACHE, SETTINGS, USER_AGENT

log = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Token bucket: at most `calls` requests per `period` seconds."""

    calls: int
    period: float

    def __post_init__(self) -> None:
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._hits and now - self._hits[0] > self.period:
                    self._hits.popleft()
                if len(self._hits) < self.calls:
                    self._hits.append(now)
                    return
                wait = self.period - (now - self._hits[0]) + 0.01
            time.sleep(max(0.0, wait))


# Per-host politeness. SEC publishes 10 req/s; we sit under it deliberately.
LIMITS: dict[str, RateLimit] = {
    "data.sec.gov": RateLimit(8, 1.0),
    "www.sec.gov": RateLimit(8, 1.0),
    "stooq.com": RateLimit(3, 1.0),
    "stooq.pl": RateLimit(3, 1.0),
    "fred.stlouisfed.org": RateLimit(10, 1.0),
    "alfred.stlouisfed.org": RateLimit(10, 1.0),
    "en.wikipedia.org": RateLimit(5, 1.0),
}
_DEFAULT_LIMIT = RateLimit(5, 1.0)


class FetchError(RuntimeError):
    """Raised when a URL cannot be retrieved after retries."""


def _cache_path(url: str, suffix: str) -> Path:
    digest = hashlib.blake2b(url.encode(), digest_size=16).hexdigest()
    return CACHE / f"{digest}{suffix}"


def fetch(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    cache_hours: float | None = 12.0,
    suffix: str = ".bin",
    timeout: float | None = None,
    browser_ua: bool = False,
    retries: int | None = None,
) -> bytes:
    """GET `url`, honouring the per-host rate limit and the on-disk cache.

    `cache_hours=None` disables caching. A cached body newer than `cache_hours`
    is returned without touching the network.
    """
    path = _cache_path(url, suffix)
    if cache_hours is not None and path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600.0
        if age_h < cache_hours:
            return path.read_bytes()

    from spaid.config.settings import BROWSER_UA

    host = httpx.URL(url).host or ""
    limit = LIMITS.get(host, _DEFAULT_LIMIT)
    hdrs = {
        "User-Agent": BROWSER_UA if browser_ua else USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }
    if headers:
        hdrs.update(headers)

    attempts = retries if retries is not None else SETTINGS.http.retries
    last: Exception | None = None

    for attempt in range(attempts):
        limit.acquire()
        try:
            resp = httpx.get(
                url,
                headers=hdrs,
                timeout=timeout or SETTINGS.http.timeout,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                body = resp.content
                if cache_hours is not None:
                    path.write_bytes(body)
                return body
            if resp.status_code in (429, 503):
                wait = min(30.0, 2.0 * (2**attempt))
                log.warning("%s throttled us (%s); backing off %.0fs", host, resp.status_code, wait)
                time.sleep(wait)
                last = FetchError(f"{url} -> HTTP {resp.status_code}")
                continue
            raise FetchError(f"{url} -> HTTP {resp.status_code}")
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
            time.sleep(min(8.0, 1.5 * (2**attempt)))

    # A stale cache entry beats no data at all.
    if path.exists():
        log.warning("using stale cache for %s after %d failed attempts", url, attempts)
        return path.read_bytes()
    raise FetchError(f"failed to fetch {url}: {last}")


def fetch_text(url: str, **kw) -> str:
    kw.setdefault("suffix", ".txt")
    return fetch(url, **kw).decode("utf-8", errors="replace")


def fetch_json(url: str, **kw):
    import json

    kw.setdefault("suffix", ".json")
    return json.loads(fetch(url, **kw))
