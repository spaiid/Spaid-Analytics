"""Current-moment security facts that are not in the filings.

Three things the SEC's aggregate interface cannot give us:

* **A consolidated share count for dual-class issuers.** Visa, Berkshire and
  others report share counts per class, and the aggregate interface carries only
  undimensioned facts, so those companies have no usable share count in the
  filings at all. Without one there is no market capitalisation, and without that
  there is no valuation. This is the concrete reason the app has a second market
  data provider rather than a single source.
* **Forward consensus.** Next-year earnings and the analyst target price are not
  filed by anyone.
* **Quote data.** Bid, ask and the implied spread, which the risk system needs to
  judge what a position will cost to establish and unwind.

None of this has history, so none of it may be read by the backtester. It
describes right now, and the table records that with a single `as_of` date.
"""

from __future__ import annotations

import logging
import math
import warnings
from datetime import UTC, date, datetime

import polars as pl

from spaid.storage.schema import SECURITY_SNAPSHOT, coerce

log = logging.getLogger(__name__)

SOURCE = "yahoo"


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def fetch_snapshot(
    securities: pl.DataFrame,
    *,
    as_of: date | None = None,
    limit: int | None = None,
    progress_every: int = 50,
) -> pl.DataFrame:
    """One row per primary security with today's provider facts."""
    import yfinance as yf

    warnings.filterwarnings("ignore")
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

    out: list[dict] = []
    failures = 0

    for i, r in enumerate(rows.iter_rows(named=True), start=1):
        ticker = r["ticker"]
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the run
            failures += 1
            log.debug("%s: snapshot failed (%s)", ticker, type(exc).__name__)
            continue

        bid, ask = _num(info.get("bid")), _num(info.get("ask"))
        spread_bps = None
        if bid and ask and ask > bid > 0:
            mid = (bid + ask) / 2.0
            spread_bps = (ask - bid) / mid * 1e4

        out.append(
            {
                "company_id": r["company_id"],
                "ticker": ticker,
                "as_of": as_of,
                "price": _num(info.get("currentPrice") or info.get("regularMarketPrice")),
                "shares_outstanding": _num(info.get("sharesOutstanding")),
                "market_cap": _num(info.get("marketCap")),
                "enterprise_value": _num(info.get("enterpriseValue")),
                "forward_eps": _num(info.get("forwardEps")),
                "forward_pe": _num(info.get("forwardPE")),
                "trailing_eps": _num(info.get("trailingEps")),
                "beta": _num(info.get("beta")),
                "dividend_yield": _num(info.get("dividendYield")),
                "target_mean_price": _num(info.get("targetMeanPrice")),
                "n_analysts": int(info["numberOfAnalystOpinions"])
                if info.get("numberOfAnalystOpinions")
                else None,
                "bid": bid,
                "ask": ask,
                "spread_bps": spread_bps,
                "float_shares": _num(info.get("floatShares")),
                "source": SOURCE,
                "collected_at": collected_at,
            }
        )
        if progress_every and i % progress_every == 0:
            log.info("snapshot: %d/%d tickers", i, rows.height)

    if not out:
        raise RuntimeError("no security snapshots returned")

    df = coerce(pl.DataFrame(out, infer_schema_length=None), SECURITY_SNAPSHOT)
    df = df.unique(subset=["company_id", "as_of"], keep="last").sort("ticker")
    log.info(
        "snapshot: %d securities (%d with a share count, %d failed)",
        df.height, int(df["shares_outstanding"].is_not_null().sum()), failures,
    )
    return df
