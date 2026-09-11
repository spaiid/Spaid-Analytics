"""Analyst estimates, revisions and earnings dates.

The product brief asks for forward growth estimates, estimate revisions,
earnings-surprise history and upcoming catalysts. Yahoo supplies all four
without a key, which makes them available to a private tool that would otherwise
have to pay for a consensus feed.

One property of this source shapes how it may be used: **there is no vintage
history.** Yahoo publishes the consensus as it stands right now, not as it stood
in 2019. So this table accumulates forward from the first time it is collected,
and the backtester must treat estimates as simply unavailable before that date.
Back-filling today's consensus into a 2019 backtest would be look-ahead of the
worst kind -- the consensus already knows what happened.

The revision counts (`up_30d`, `down_30d`) are the exception that makes this
worth collecting at all: they are a *current* statement about recent analyst
behaviour, and they are one of the few genuinely forward-looking inputs
available at this price point.
"""

from __future__ import annotations

import logging
import math
import warnings
from datetime import UTC, date, datetime

import polars as pl

from spaid.storage.schema import EARNINGS_EVENTS, ESTIMATES, coerce

log = logging.getLogger(__name__)

SOURCE = "yahoo"

# Yahoo's period labels, in the order the UI wants them.
PERIODS = ("0q", "+1q", "0y", "+1y")


def _clean(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _int(value) -> int | None:
    f = _clean(value)
    return None if f is None else int(f)


def _frame_rows(df, period_index_name: str = "period") -> dict[str, dict]:
    """Turn a small pandas frame indexed by period label into a dict of dicts."""
    if df is None or getattr(df, "empty", True):
        return {}
    try:
        return {str(idx): row.to_dict() for idx, row in df.iterrows()}
    except Exception:  # noqa: BLE001 - the shape varies by yfinance version
        return {}


def fetch_for_ticker(
    ticker: str, *, company_id: str, as_of: date, collected_at: datetime
) -> tuple[list[dict], list[dict]]:
    """Estimates and earnings events for one ticker.

    Returns empty lists rather than raising when Yahoo has no coverage: a
    company with no analyst following is a real state of the world, and the
    confidence system is what should react to it.
    """
    import yfinance as yf

    warnings.filterwarnings("ignore")
    tk = yf.Ticker(ticker)

    estimates: list[dict] = []
    events: list[dict] = []

    try:
        eps = _frame_rows(tk.earnings_estimate)
        rev = _frame_rows(tk.revenue_estimate)
        revisions = _frame_rows(tk.eps_revisions)
    except Exception as exc:  # noqa: BLE001
        log.debug("%s: no estimate frames (%s)", ticker, type(exc).__name__)
        eps, rev, revisions = {}, {}, {}

    for period in PERIODS:
        for metric, table in (("eps", eps), ("revenue", rev)):
            row = table.get(period)
            if not row:
                continue
            rvn = revisions.get(period, {}) if metric == "eps" else {}
            consensus = _clean(row.get("avg"))
            if consensus is None:
                continue
            estimates.append(
                {
                    "company_id": company_id,
                    "ticker": ticker,
                    "as_of": as_of,
                    "metric": metric,
                    "period": period,
                    "period_end_estimate": None,
                    "consensus": consensus,
                    "low": _clean(row.get("low")),
                    "high": _clean(row.get("high")),
                    "n_analysts": _int(row.get("numberOfAnalysts")),
                    "year_ago": _clean(row.get("yearAgoEps") or row.get("yearAgoRevenue")),
                    "growth": _clean(row.get("growth")),
                    "up_7d": _int(rvn.get("upLast7days")),
                    "up_30d": _int(rvn.get("upLast30days")),
                    "down_7d": _int(rvn.get("downLast7Days") or rvn.get("downLast7days")),
                    "down_30d": _int(rvn.get("downLast30days")),
                    "source": SOURCE,
                    "collected_at": collected_at,
                }
            )

    try:
        hist = tk.get_earnings_dates(limit=24)
    except Exception as exc:  # noqa: BLE001
        log.debug("%s: no earnings dates (%s)", ticker, type(exc).__name__)
        hist = None

    if hist is not None and not getattr(hist, "empty", True):
        for idx, row in hist.iterrows():
            try:
                event_date = idx.date()
            except AttributeError:
                continue
            actual = _clean(row.get("Reported EPS"))
            events.append(
                {
                    "company_id": company_id,
                    "ticker": ticker,
                    "event_date": event_date,
                    "is_future": actual is None,
                    "eps_estimate": _clean(row.get("EPS Estimate")),
                    "eps_actual": actual,
                    "surprise_pct": _clean(row.get("Surprise(%)")),
                    "source": SOURCE,
                    "collected_at": collected_at,
                }
            )

    return estimates, events


def fetch_estimates(
    securities: pl.DataFrame,
    *,
    as_of: date | None = None,
    limit: int | None = None,
    progress_every: int = 50,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Collect estimates and earnings events for every primary security."""
    as_of = as_of or datetime.now(UTC).date()
    collected_at = datetime.now(UTC)

    rows = (
        securities.filter(pl.col("is_primary"))
        .select(["company_id", "ticker"])
        .unique()
        .sort("ticker")
    )
    if limit:
        rows = rows.head(limit)

    all_est: list[dict] = []
    all_evt: list[dict] = []
    failures = 0

    for i, r in enumerate(rows.iter_rows(named=True), start=1):
        try:
            est, evt = fetch_for_ticker(
                r["ticker"],
                company_id=r["company_id"],
                as_of=as_of,
                collected_at=collected_at,
            )
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the run
            failures += 1
            log.debug("%s: estimates failed (%s)", r["ticker"], exc)
            continue
        all_est.extend(est)
        all_evt.extend(evt)
        if progress_every and i % progress_every == 0:
            log.info(
                "estimates: %d/%d tickers, %d estimate rows, %d events",
                i, rows.height, len(all_est), len(all_evt),
            )

    est_df = (
        coerce(pl.DataFrame(all_est, infer_schema_length=None), ESTIMATES)
        .unique(subset=["company_id", "as_of", "metric", "period"], keep="last")
        .sort(["company_id", "metric", "period"])
        if all_est
        else coerce(pl.DataFrame(schema={"company_id": pl.Utf8}), ESTIMATES)
    )
    evt_df = (
        coerce(pl.DataFrame(all_evt, infer_schema_length=None), EARNINGS_EVENTS)
        .unique(subset=["company_id", "event_date"], keep="last")
        .sort(["company_id", "event_date"])
        if all_evt
        else coerce(pl.DataFrame(schema={"company_id": pl.Utf8}), EARNINGS_EVENTS)
    )

    log.info(
        "estimates: %d rows for %d companies; %d earnings events; %d tickers failed",
        est_df.height, est_df["company_id"].n_unique(), evt_df.height, failures,
    )
    return est_df, evt_df
