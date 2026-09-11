"""Access to SEC XBRL company facts.

`data.sec.gov`'s per-company API is rate limited and will hand out 429s (and
temporary blocks) long before 500 companies are fetched. The bulk archive on
`www.sec.gov` is not rate limited, but it is 1.4 GB.

We get the best of both by treating the remote ZIP as a random-access file:
`HTTPRangeFile` serves `zipfile` over HTTP range requests, so we read the
central directory (~2 MB) once and then pull only the ~500 members we care
about, about 1 MB each. No 1.4 GB download, no rate limits.
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import httpx

from spaid.config.settings import CACHE, USER_AGENT

log = logging.getLogger(__name__)

BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
FACTS_CACHE = CACHE / "companyfacts"
FACTS_CACHE.mkdir(parents=True, exist_ok=True)


class HTTPRangeFile(io.RawIOBase):
    """A seekable read-only file backed by HTTP range requests."""

    def __init__(self, url: str, client: httpx.Client, headers: dict[str, str] | None = None):
        self.url = url
        self.client = client
        self.headers = headers or {}
        resp = client.head(url, headers=self.headers, follow_redirects=True)
        resp.raise_for_status()
        if resp.headers.get("accept-ranges", "").lower() != "bytes":
            log.warning("%s does not advertise range support; trying anyway", url)
        self._size = int(resp.headers["content-length"])
        self._pos = 0
        self.request_count = 0
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self._size + offset
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        if size <= 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + size, self._size) - 1
        headers = dict(self.headers)
        headers["Range"] = f"bytes={self._pos}-{end}"
        resp = self.client.get(self.url, headers=headers, follow_redirects=True)
        resp.raise_for_status()
        data = resp.content
        self.request_count += 1
        self.bytes_fetched += len(data)
        self._pos += len(data)
        return data

    def readinto(self, buf) -> int:  # type: ignore[override]
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)


class CompanyFacts:
    """Reads company facts JSON, preferring the local cache."""

    def __init__(self, *, cache_days: float = 7.0):
        self.cache_days = cache_days
        self._client: httpx.Client | None = None
        self._zip: zipfile.ZipFile | None = None
        self._raw: HTTPRangeFile | None = None

    def _ensure_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._client = httpx.Client(timeout=120.0)
            self._raw = HTTPRangeFile(BULK_URL, self._client, {"User-Agent": USER_AGENT})
            buffered = io.BufferedReader(self._raw, buffer_size=1 << 20)
            self._zip = zipfile.ZipFile(buffered)
            log.info(
                "opened SEC bulk archive: %d members, %.1f MB of range reads",
                len(self._zip.namelist()), self._raw.bytes_fetched / 1e6,
            )
        return self._zip

    def _cache_path(self, cik: int) -> Path:
        return FACTS_CACHE / f"CIK{cik:010d}.json"

    def get(self, cik: int) -> dict | None:
        import time

        path = self._cache_path(cik)
        if path.exists():
            age_days = (time.time() - path.stat().st_mtime) / 86400.0
            if age_days < self.cache_days:
                try:
                    return json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass

        member = f"CIK{cik:010d}.json"
        try:
            data = self._ensure_zip().read(member)
        except KeyError:
            log.warning("no SEC facts for CIK %d", cik)
            return None
        except Exception as exc:  # noqa: BLE001 - network/zip errors are all recoverable
            log.warning("failed reading %s: %s", member, exc)
            return None

        path.write_bytes(data)
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            log.warning("malformed JSON for CIK %d", cik)
            return None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._zip = None
        self._raw = None
        self._client = None

    def __enter__(self) -> "CompanyFacts":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def stats(self) -> dict:
        if self._raw is None:
            return {"requests": 0, "mb": 0.0}
        return {"requests": self._raw.request_count, "mb": round(self._raw.bytes_fetched / 1e6, 1)}
