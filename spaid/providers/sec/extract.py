"""Turn SEC company-facts JSON into canonical observation rows.

This layer is deliberately dumb. It does not compute trailing-twelve-month
figures, choose between competing tags, or repair anything: it copies facts out
of the filing with every field that establishes provenance, and stops. All
judgement happens downstream in `spaid.pipeline.fundamentals`, where it can be
revised without re-fetching a gigabyte of filings.

What it *does* enforce is the three-dates discipline:

* `period_end`  -- the period the number describes
* `filed`       -- the date the filing was submitted
* `available_at`-- the first session on which we could have acted on it

Collapsing those into one date is the single most effective way to build a
backtest that cannot be traded, so they are separate columns and every one is
mandatory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta

import polars as pl

from spaid.providers.sec.concepts import TAG_TO_CONCEPTS, WANTED_TAGS
from spaid.storage.schema import OBSERVATIONS, coerce

log = logging.getLogger(__name__)

SOURCE = "sec_companyfacts"
ACCEPTED_UNITS = frozenset({"USD", "shares", "USD/shares", "pure"})

# Facts reported for a window longer than this are not a period result we can
# use (multi-year cumulative disclosures appear in some filings).
MAX_DURATION_DAYS = 400


def company_id_for(cik: int) -> str:
    """The stable identifier every table joins on.

    Ticker symbols are reassigned -- `FB` became `META`, and a delisted ticker
    can be handed to an unrelated company years later. The SEC's Central Index
    Key does not move, so it is the identity and the ticker is a label.
    """
    return f"CIK{int(cik):010d}"


def _as_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


class TradingCalendar:
    """Maps a calendar date to the first session on or after it.

    Used to set `available_at`. A filing submitted after Tuesday's close is
    knowable Tuesday evening but not actionable until Wednesday's open, so the
    convention here is the next session strictly after the filing date. That
    costs one day of signal and removes a whole class of look-ahead argument.
    """

    def __init__(self, sessions: Sequence[date]):
        self._sessions = sorted(set(sessions))
        if not self._sessions:
            raise ValueError("trading calendar requires at least one session")

    def next_session_after(self, d: date) -> date:
        import bisect

        i = bisect.bisect_right(self._sessions, d)
        if i >= len(self._sessions):
            # Beyond the last known session: fall back to the next weekday, so
            # freshly filed data is not silently dropped between price refreshes.
            nxt = d + timedelta(days=1)
            while nxt.weekday() >= 5:
                nxt += timedelta(days=1)
            return nxt
        return self._sessions[i]

    @classmethod
    def from_prices(cls, prices: pl.DataFrame, ticker: str = "SPY") -> TradingCalendar:
        sub = prices.filter(pl.col("ticker") == ticker)
        if sub.is_empty():
            sub = prices
        return cls(sub["date"].unique().sort().to_list())


def _primary_concept(tag: str) -> str | None:
    """The concept this tag most specifically represents.

    A tag can serve several concepts -- the pre-tax income tag is also the
    fallback for operating income -- so the observation records the best single
    match for readability while the resolver downstream still works from `tag`
    and each concept's own preference list.
    """
    candidates = TAG_TO_CONCEPTS.get(tag)
    if not candidates:
        return None
    return min(candidates, key=lambda kv: kv[1])[0]


def extract_company(
    facts: dict,
    *,
    cik: int,
    ticker: str | None = None,
    calendar: TradingCalendar | None = None,
    collected_at: datetime | None = None,
) -> list[dict]:
    """Flatten one company-facts document into observation rows.

    Rows whose reporting date precedes the period they claim to describe are
    dropped: a filing cannot report a quarter that has not ended, and treating
    such a row as the newest available figure freezes a mis-dated value in place
    for months.
    """
    collected_at = collected_at or datetime.now(UTC)
    company_id = company_id_for(cik)
    out: list[dict] = []
    dropped_future = 0

    for taxonomy, tags in (facts.get("facts") or {}).items():
        for tag, node in tags.items():
            qualified = f"{taxonomy}:{tag}"
            # The concept map stores bare us-gaap tags and namespace-qualified
            # ones for other taxonomies (dei:EntityCommonStockSharesOutstanding).
            lookup = tag if tag in WANTED_TAGS else qualified
            if lookup not in WANTED_TAGS:
                continue
            concept = _primary_concept(lookup)

            for unit, rows in (node.get("units") or {}).items():
                if unit not in ACCEPTED_UNITS:
                    continue
                for r in rows:
                    val = r.get("val")
                    end = _as_date(r.get("end"))
                    filed = _as_date(r.get("filed"))
                    if val is None or end is None or filed is None:
                        continue
                    if end > filed:
                        dropped_future += 1
                        continue

                    start = _as_date(r.get("start"))
                    is_duration = start is not None
                    duration = (end - start).days if is_duration else None
                    if duration is not None and (
                        duration <= 0 or duration > MAX_DURATION_DAYS
                    ):
                        continue

                    available = (
                        calendar.next_session_after(filed) if calendar else filed
                    )
                    out.append(
                        {
                            "company_id": company_id,
                            "cik": int(cik),
                            "ticker": ticker,
                            "taxonomy": taxonomy,
                            "tag": lookup,
                            "concept": concept,
                            "value": float(val),
                            "unit": unit,
                            "period_start": start,
                            "period_end": end,
                            "period_type": "duration" if is_duration else "instant",
                            "duration_days": duration,
                            "fiscal_year": int(r["fy"]) if r.get("fy") is not None else None,
                            "fiscal_period": r.get("fp"),
                            "form": r.get("form"),
                            "accession": r.get("accn"),
                            "filed": filed,
                            "available_at": available,
                            "revision": None,  # assigned once the company is complete
                            "is_restatement": None,
                            "source": SOURCE,
                            "collected_at": collected_at,
                        }
                    )

    if dropped_future:
        log.debug(
            "%s: dropped %d facts whose period ends after the filing date",
            company_id, dropped_future,
        )
    return out


def assign_revisions(df: pl.DataFrame) -> pl.DataFrame:
    """Number each disclosure of a period and flag genuine restatements.

    Revision 0 is what the market saw first. A later filing repeating the same
    number is still revision 1 but is not a restatement; only a changed value is.
    Keeping both lets the value-trap detector ask "was this quarter revised
    downward?", which is one of its more reliable signals.
    """
    if df.is_empty():
        return df

    key = ["company_id", "tag", "unit", "period_start", "period_end"]
    out = (
        df.sort([*key, "filed", "accession"])
        .with_columns(
            pl.int_range(pl.len()).over(key).cast(pl.Int32).alias("revision"),
            pl.col("value").first().over(key).alias("_first_value"),
        )
        .with_columns(
            (
                (pl.col("revision") > 0)
                & (
                    (pl.col("value") - pl.col("_first_value")).abs()
                    > (pl.col("_first_value").abs() * 1e-6 + 1e-6)
                )
            ).alias("is_restatement")
        )
        .drop("_first_value")
    )
    return out


def build_observations(
    companies: Iterable[tuple[int, str | None]],
    *,
    facts_loader,
    calendar: TradingCalendar | None = None,
    progress_every: int = 50,
) -> pl.DataFrame:
    """Extract observations for many companies.

    `facts_loader` is any callable taking a CIK and returning the company-facts
    document or None, which keeps this function testable without a network or a
    1.4 GB archive.
    """
    from spaid.providers.sec.predecessors import predecessor_for

    collected_at = datetime.now(UTC)
    rows: list[dict] = []
    missing: list[int] = []
    inherited: list[str] = []
    companies = list(companies)

    for i, (cik, ticker) in enumerate(companies, start=1):
        facts = facts_loader(cik)
        company_rows: list[dict] = []
        if facts:
            company_rows.extend(
                extract_company(
                    facts, cik=cik, ticker=ticker, calendar=calendar, collected_at=collected_at
                )
            )

        # A company that reorganised into a new registrant keeps its history
        # under the old identifier. Carry it forward under the *successor's*
        # company_id so the two are one continuous series downstream.
        link = predecessor_for(cik)
        if link is not None:
            prior = facts_loader(link.predecessor_cik)
            if prior:
                carried = extract_company(
                    prior,
                    cik=link.predecessor_cik,
                    ticker=ticker,
                    calendar=calendar,
                    collected_at=collected_at,
                )
                successor_id = company_id_for(cik)
                for row in carried:
                    row["company_id"] = successor_id
                    row["source"] = f"{SOURCE}:predecessor"
                company_rows.extend(carried)
                inherited.append(
                    f"{ticker or successor_id} inherited {len(carried)} facts from "
                    f"{link.predecessor_name}"
                )
            else:
                log.warning(
                    "no facts available for predecessor CIK %d of %s",
                    link.predecessor_cik, ticker or cik,
                )

        if not company_rows:
            missing.append(cik)
            continue
        rows.extend(company_rows)
        if progress_every and i % progress_every == 0:
            log.info("observations: %d/%d companies, %d rows", i, len(companies), len(rows))

    for note in inherited:
        log.info("predecessor history: %s", note)

    if not rows:
        raise RuntimeError("no observations extracted from any company")

    df = pl.DataFrame(rows, infer_schema_length=None)
    df = coerce(df, OBSERVATIONS)
    df = assign_revisions(df)

    # The primary key includes the accession, so a filing that reports the same
    # tag and period twice (different dimensions collapsed to the same face
    # value) would collide. Keep the last, which is the filing's own final word.
    before = df.height
    df = df.unique(
        subset=["company_id", "tag", "period_end", "period_start", "accession"], keep="last"
    ).sort(["company_id", "tag", "period_end", "filed"])
    if before != df.height:
        log.info("observations: collapsed %d duplicate facts", before - df.height)

    if missing:
        log.warning(
            "no SEC facts for %d companies (first few: %s)",
            len(missing), missing[:8],
        )
    df = coerce(df, OBSERVATIONS)
    log.info(
        "observations: %d rows, %d companies, %d distinct tags",
        df.height, df["company_id"].n_unique(), df["tag"].n_unique(),
    )
    return df
