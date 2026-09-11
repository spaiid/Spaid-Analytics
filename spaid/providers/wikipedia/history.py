"""Point-in-time S&P 500 membership, reconstructed from page revisions.

The constituents page states who is in the index *today*. Ranking a historical
date against today's membership is the survivorship bias that makes a backtest
meaningless: the companies that were dropped are exactly the ones that did
badly, and leaving them out is equivalent to knowing in advance which names
would survive.

The page's own edit history is the fix. Wikipedia keeps every revision, so the
table as it stood on any past date is retrievable, and a sequence of those
tables *is* the membership history: a company present in the March snapshot and
absent from the April one left the index between those two dates. This is an
observational reconstruction, not an official index file, and the difference is
recorded rather than glossed over:

* **Entry and exit dates are bracketed, not exact.** A monthly snapshot grid
  locates a change within a month. The bracket is stored (`entry_observed_after`
  / `entry_observed_by`) so the backtester can refuse to trade inside an
  ambiguous window instead of pretending to a precision it does not have.
* **The table's own "date added" column sharpens entries.** Where a snapshot
  reports a date added that falls inside the bracket, the entry is exact and is
  marked `confirmed`.
* **Exits carry no reason.** The page once kept a changes table with reasons --
  acquired, merged, bankrupt -- but it no longer does, and older revisions of it
  are a different reconstruction problem. An exit is therefore recorded with a
  reason of `unknown`, never guessed at.

What this module cannot supply is prices for the companies it discovers. That
gap is measured in `spaid.pipeline.security_master` and is what keeps the
backtest labelled exploratory.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise

import polars as pl

from spaid.config.settings import RAW
from spaid.providers.http import fetch_json

log = logging.getLogger(__name__)

API = "https://en.wikipedia.org/w/api.php"
PAGE = "List of S&P 500 companies"
SOURCE = "wikipedia_revisions"

# Parsed snapshots are kept as the raw layer's own copy. A backtest has to be
# reproducible months later, and re-deriving membership from a page that keeps
# being edited would not be.
SNAPSHOT_DIR = RAW / "wikipedia_sp500_snapshots"

# The revision at which the table gained a CIK column. Before it, a snapshot
# identifies companies by ticker and name only, which cannot be joined to
# filings with any confidence -- tickers are reused across companies.
CIK_COLUMN_FROM = date(2014, 7, 1)

_TICKER_COLUMNS = ("Symbol", "Ticker symbol", "Ticker")
_NAME_COLUMNS = ("Security", "Company")
_SECTOR_COLUMNS = ("GICS Sector",)
_INDUSTRY_COLUMNS = ("GICS Sub-Industry", "GICS Sub Industry")
_ADDED_COLUMNS = ("Date added", "Date first added")


@dataclass(frozen=True)
class Revision:
    revid: int
    timestamp: datetime

    @property
    def day(self) -> date:
        return self.timestamp.date()


def to_yahoo(symbol: str) -> str:
    """Wikipedia writes class shares with a dot; Yahoo uses a hyphen."""
    return str(symbol).strip().upper().replace(".", "-")


def _column(frame, candidates: tuple[str, ...]) -> str | None:
    """First matching column name, tolerating the footnote markers Wikipedia adds."""
    lookup = {str(c): str(c) for c in frame.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    # "Date first added[3][4]" and similar.
    for raw in lookup:
        base = re.sub(r"\[.*?\]", "", raw).strip()
        if base in candidates:
            return raw
    return None


def list_revisions(start: date, end: date) -> list[Revision]:
    """Every revision of the constituents page between two dates, newest first.

    One request per 500 revisions rather than one per snapshot date, because the
    page is edited far more often than we sample it.
    """
    out: list[Revision] = []
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": PAGE,
        "rvlimit": "500",
        "rvprop": "ids|timestamp",
        "rvstart": f"{end.isoformat()}T23:59:59Z",
        "rvend": f"{start.isoformat()}T00:00:00Z",
        "rvdir": "older",
        "format": "json",
        "formatversion": "2",
    }
    cont: dict[str, str] = {}
    while True:
        payload = fetch_json(_encode(params | cont), cache_hours=24.0 * 7)
        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            break
        for rev in pages[0].get("revisions", []):
            out.append(
                Revision(
                    revid=int(rev["revid"]),
                    timestamp=datetime.fromisoformat(
                        rev["timestamp"].replace("Z", "+00:00")
                    ),
                )
            )
        if "continue" not in payload:
            break
        cont = {k: str(v) for k, v in payload["continue"].items()}

    out.sort(key=lambda r: r.timestamp, reverse=True)
    log.info(
        "constituents page: %d revisions between %s and %s", len(out), start, end
    )
    return out


def _encode(params: dict[str, str]) -> str:
    from urllib.parse import urlencode

    return f"{API}?{urlencode(params)}"


def revision_in_effect(revisions: list[Revision], as_of: date) -> Revision | None:
    """The last revision saved on or before `as_of` -- the page as it then read."""
    cutoff = datetime.combine(as_of, datetime.max.time()).replace(tzinfo=UTC)
    for rev in revisions:  # newest first
        if rev.timestamp <= cutoff:
            return rev
    return None


def parse_revision(revid: int) -> pl.DataFrame:
    """The constituents table from one revision, normalised.

    Cached on disk by revision id. A revision is immutable, so the cache never
    needs invalidating and a rebuild months later reads exactly the same bytes.
    """
    import pandas as pd

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    cached = SNAPSHOT_DIR / f"rev_{revid}.parquet"
    if cached.exists():
        return pl.read_parquet(cached)

    payload = fetch_json(
        _encode(
            {
                "action": "parse",
                "oldid": str(revid),
                "prop": "text",
                "format": "json",
                "formatversion": "2",
            }
        ),
        cache_hours=24.0 * 365,
    )
    html = payload.get("parse", {}).get("text")
    if not html:
        raise RuntimeError(f"revision {revid} returned no parsed text")

    tables = pd.read_html(io.StringIO(html))
    frame = None
    for candidate in tables:
        if _column(candidate, _TICKER_COLUMNS) and len(candidate) > 300:
            frame = candidate
            break
    if frame is None:
        raise RuntimeError(f"revision {revid}: no constituents table found")

    ticker_col = _column(frame, _TICKER_COLUMNS)
    name_col = _column(frame, _NAME_COLUMNS)
    sector_col = _column(frame, _SECTOR_COLUMNS)
    industry_col = _column(frame, _INDUSTRY_COLUMNS)
    added_col = _column(frame, _ADDED_COLUMNS)
    cik_col = "CIK" if "CIK" in frame.columns else None

    n = len(frame)
    data = {
        "ticker": [to_yahoo(s) for s in frame[ticker_col].astype(str)],
        "name": (
            frame[name_col].astype(str).tolist() if name_col else [""] * n
        ),
        "sector": (
            frame[sector_col].astype(str).tolist() if sector_col else [""] * n
        ),
        "industry": (
            frame[industry_col].astype(str).tolist() if industry_col else [""] * n
        ),
        "cik": (
            [_int_or_none(c) for c in frame[cik_col]] if cik_col else [None] * n
        ),
        "date_added": (
            [_date_or_none(x) for x in frame[added_col]] if added_col else [None] * n
        ),
    }
    out = (
        pl.DataFrame(
            data,
            schema={
                "ticker": pl.Utf8,
                "name": pl.Utf8,
                "sector": pl.Utf8,
                "industry": pl.Utf8,
                "cik": pl.Int64,
                "date_added": pl.Date,
            },
        )
        .filter(pl.col("ticker").str.len_chars() > 0)
        .unique(subset=["ticker"], keep="first")
        .sort("ticker")
    )
    out.write_parquet(cached, compression="zstd")
    return out


def _int_or_none(value) -> int | None:
    try:
        text = re.sub(r"[^0-9]", "", str(value))
        return int(text) if text else None
    except (TypeError, ValueError):
        return None


def _date_or_none(value) -> date | None:
    text = str(value).strip()[:10]
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(text[: len(datetime.now().strftime(fmt))], fmt).date()
        except ValueError:
            continue
    return None


def month_ends(start: date, end: date) -> list[date]:
    """The last calendar day of every month in the range, inclusive."""
    out: list[date] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        nxt = date(cursor.year + (cursor.month // 12), (cursor.month % 12) + 1, 1)
        last = nxt - timedelta(days=1)
        if start <= last <= end:
            out.append(last)
        cursor = nxt
    return out


def snapshots(
    start: date,
    end: date,
    *,
    grid: list[date] | None = None,
) -> dict[date, pl.DataFrame]:
    """The constituents table as it read on each date in the grid."""
    grid = grid or month_ends(start, end)
    revisions = list_revisions(start - timedelta(days=120), end)
    if not revisions:
        raise RuntimeError("no revisions of the constituents page could be listed")

    out: dict[date, pl.DataFrame] = {}
    seen: dict[int, pl.DataFrame] = {}
    for as_of in grid:
        rev = revision_in_effect(revisions, as_of)
        if rev is None:
            log.warning("no revision of the constituents page existed at %s", as_of)
            continue
        if rev.revid not in seen:
            try:
                seen[rev.revid] = parse_revision(rev.revid)
            except Exception as exc:  # a single bad revision must not lose the run
                log.warning("revision %d (%s) could not be parsed: %s", rev.revid, as_of, exc)
                continue
        table = seen[rev.revid]
        if table.height < 400:
            log.warning(
                "revision %d at %s lists only %d companies; treating it as damaged "
                "and skipping the snapshot",
                rev.revid, as_of, table.height,
            )
            continue
        out[as_of] = table.with_columns(
            pl.lit(as_of).alias("snapshot_date"),
            pl.lit(rev.revid).alias("revid"),
        )

    log.info(
        "membership snapshots: %d dates from %s to %s, %d distinct revisions",
        len(out),
        min(out) if out else "-",
        max(out) if out else "-",
        len({df["revid"][0] for df in out.values()}),
    )
    return out


@dataclass(frozen=True)
class MembershipSpan:
    """One continuous period during which a security was an index member."""

    ticker: str
    cik: int | None
    name: str
    sector: str
    industry: str
    entry_observed_after: date | None  # last snapshot where it was absent
    entry_observed_by: date  # first snapshot where it was present
    entry_date: date | None  # exact, where the table states one
    exit_observed_after: date | None  # last snapshot where it was present
    exit_observed_by: date | None  # first snapshot where it was absent
    still_member: bool
    n_snapshots: int


def build_spans(snaps: dict[date, pl.DataFrame]) -> list[MembershipSpan]:
    """Turn a sequence of snapshots into membership spans with bracketed edges.

    Keyed on ticker because that is the only identifier every snapshot carries.
    The security master resolves tickers to companies afterwards, and records
    where that resolution is uncertain.
    """
    if not snaps:
        return []

    dates = sorted(snaps)
    present: dict[str, set[date]] = {}
    detail: dict[str, dict] = {}
    for as_of in dates:
        for row in snaps[as_of].iter_rows(named=True):
            present.setdefault(row["ticker"], set()).add(as_of)
            # Latest observation of the descriptive fields wins; a renamed
            # company should read under its current name.
            detail[row["ticker"]] = row

    spans: list[MembershipSpan] = []
    for ticker, days in present.items():
        ordered = sorted(days)
        info = detail[ticker]
        # Split into runs of consecutive snapshot dates: a company can leave the
        # index and return, and merging those into one span would let the
        # backtest hold it through a period when it was not a member.
        runs: list[list[date]] = [[ordered[0]]]
        for prev, cur in pairwise(ordered):
            gap = dates.index(cur) - dates.index(prev)
            if gap == 1:
                runs[-1].append(cur)
            else:
                runs.append([cur])

        for run in runs:
            first, last = run[0], run[-1]
            i_first, i_last = dates.index(first), dates.index(last)
            before = dates[i_first - 1] if i_first > 0 else None
            after = dates[i_last + 1] if i_last + 1 < len(dates) else None

            stated = info.get("date_added")
            exact = None
            if stated is not None:
                # Only trust the stated date when it is consistent with what the
                # snapshots actually observed.
                upper = first
                lower = before or date(1900, 1, 1)
                if lower <= stated <= upper:
                    exact = stated
                elif before is None and stated <= first:
                    # Already a member at the first snapshot; the stated date is
                    # the real entry and predates our window.
                    exact = stated

            spans.append(
                MembershipSpan(
                    ticker=ticker,
                    cik=info.get("cik"),
                    name=info.get("name") or "",
                    sector=info.get("sector") or "",
                    industry=info.get("industry") or "",
                    entry_observed_after=before,
                    entry_observed_by=first,
                    entry_date=exact,
                    exit_observed_after=last if after is not None else None,
                    exit_observed_by=after,
                    still_member=after is None,
                    n_snapshots=len(run),
                )
            )

    spans.sort(key=lambda s: (s.ticker, s.entry_observed_by))
    n_exits = sum(1 for s in spans if not s.still_member)
    log.info(
        "membership spans: %d spans over %d tickers, %d observed exits",
        len(spans), len({s.ticker for s in spans}), n_exits,
    )
    return spans


def spans_frame(spans: list[MembershipSpan]) -> pl.DataFrame:
    """The spans as a frame, for inspection and for the security master."""
    if not spans:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8, "cik": pl.Int64, "name": pl.Utf8,
                "sector": pl.Utf8, "industry": pl.Utf8,
                "entry_observed_after": pl.Date, "entry_observed_by": pl.Date,
                "entry_date": pl.Date, "exit_observed_after": pl.Date,
                "exit_observed_by": pl.Date, "still_member": pl.Boolean,
                "n_snapshots": pl.Int32,
            }
        )
    return pl.DataFrame([vars(s) for s in spans]).with_columns(
        pl.col("n_snapshots").cast(pl.Int32)
    )
