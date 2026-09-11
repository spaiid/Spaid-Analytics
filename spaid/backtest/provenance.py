"""The four coordinates that make a backtest result reproducible.

A number from a backtest is worthless unless you can say exactly what produced
it. Four things have to be pinned, and none of them is optional:

* **Which code** -- the git commit, plus whether the tree was dirty when the run
  happened. A result from uncommitted code is marked as such rather than being
  attributed to the last commit, which it was not produced by.
* **Which data** -- a checksum over the curated tables' shape and freshness, so
  that refetching prices produces a visibly different data version rather than
  silently changing yesterday's result.
* **Which universe** -- the membership reconstruction, which is itself versioned.
* **Which strategy** -- the frozen configuration checksum.

They are computed here, in one place, so that every run records them the same
way and a run that cannot determine one of them says `unknown` instead of
omitting it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import UTC, datetime

from spaid.config.settings import ROOT
from spaid.storage import store

log = logging.getLogger(__name__)

# The tables a backtest reads. A change to any of them changes the answer, so
# all of them are in the data version; tables the backtester never touches are
# deliberately not, or every unrelated refresh would invalidate the history.
DATA_TABLES: tuple[str, ...] = (
    "prices",
    "fundamentals",
    "observations",
    "companies",
    "securities",
    "security_master",
    "universe_membership",
)


def code_commit() -> str:
    """The current commit, suffixed `+dirty` when the tree has uncommitted changes."""
    try:
        rev = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("could not read the git commit (%s); recording it as unknown", exc)
        return "unknown"
    return f"{rev}+dirty" if status else rev


def data_version() -> str:
    """A checksum over the curated tables the backtest depends on.

    Built from each table's row count, column count and last-updated stamp
    rather than its bytes: hashing 600 MB of parquet on every run would cost
    more than it is worth, and these three change whenever the content does.
    """
    parts: list[dict] = []
    for name in DATA_TABLES:
        info = store.info(name)
        if info is None:
            parts.append({"table": name, "present": False})
            continue
        parts.append(
            {
                "table": name,
                "rows": info.rows,
                "columns": info.columns,
                "updated_at": info.updated_at,
                "date_min": info.date_min,
                "date_max": info.date_max,
            }
        )
    digest = hashlib.blake2b(
        json.dumps(parts, sort_keys=True).encode(), digest_size=16
    ).hexdigest()
    return f"data-{digest[:12]}"


def universe_version() -> str:
    """The most recent historical-universe reconstruction, or the current list."""
    membership = store.read("universe_membership")
    if membership is None or membership.is_empty():
        return "unknown"
    versions = [
        v for v in membership["universe_version"].unique().to_list() if v is not None
    ]
    historical = [v for v in versions if v != "current-list"]
    if historical:
        return sorted(historical)[-1]
    return "current-list"


def checksum(payload) -> str:
    """A stable checksum over any JSON-serialisable configuration."""
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()


def run_id(prefix: str = "bt") -> str:
    """A run identifier that sorts chronologically and never collides."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"{prefix}-{stamp}"


def coordinates() -> dict:
    """Everything needed to reproduce a run, as one dictionary."""
    return {
        "code_commit": code_commit(),
        "data_version": data_version(),
        "universe_version": universe_version(),
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
