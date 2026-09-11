"""Daily price history.

Yahoo, through the `yfinance` client, is the price backbone. Two findings from
the earlier build are worth preserving because they cost real time to establish:
Yahoo's raw chart endpoints answer 429 to a plain HTTP client but the library
succeeds because it establishes a cookie and crumb session first; and Stooq, the
obvious alternative, silently truncates after a couple of dozen requests a day
and restates its history on every dividend, so cached data is not reproducible.

The central discipline here is keeping two price series apart:

* ``close_raw``  -- what the stock actually traded at that day, on that day's
  share basis. Eligibility screens and market capitalisation use this.
* ``close_adj``  -- the total-return series, restated for splits and dividends
  onto today's basis. Returns and momentum use this.

Confusing them is not cosmetic. Nvidia traded at $22 in mid-2015; its adjusted
close reads $0.54 because of later splits. Screening on the adjusted price
removes the best performer of the decade from seven years of history, and does
so retroactively every time a new split happens.
"""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, date, datetime

import polars as pl

from spaid.config.settings import SETTINGS
from spaid.storage.schema import PRICES, coerce

log = logging.getLogger(__name__)

SOURCE = "yahoo"
BATCH = 100


def _download_batch(tickers: list[str], start: date) -> pl.DataFrame | None:
    import pandas as pd
    import yfinance as yf

    warnings.filterwarnings("ignore")
    raw = yf.download(
        tickers,
        start=start.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw is None or raw.empty:
        return None

    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers[:1]])

    long = (
        raw.stack(level=1, future_stack=True)
        .rename_axis(index=["date", "ticker"])
        .reset_index()
    )
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Adj Close": "close_adj",
        "Close": "close_raw",
        "Volume": "volume",
        "Stock Splits": "split_ratio",
        "Dividends": "dividend",
    }
    long = long.rename(columns=rename)
    keep = [
        c
        for c in [
            "date", "ticker", "open", "high", "low",
            "close_adj", "close_raw", "volume", "split_ratio", "dividend",
        ]
        if c in long.columns
    ]
    long = long[keep].dropna(subset=["close_raw"])
    if long.empty:
        return None

    return pl.from_pandas(long).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
        pl.col("close_raw").cast(pl.Float64),
        pl.col("close_adj").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )


def add_split_factor(df: pl.DataFrame) -> pl.DataFrame:
    """Product of every split occurring strictly *after* each date, per ticker.

    A share count reported in a 2024 filing is stated on that filing's basis. To
    compare it with a price series restated onto today's basis, multiply it by
    the splits that have happened since. Applying the factor for the wrong date
    is what made Nvidia's market capitalisation read $300bn for two months after
    its 2024 split.
    """
    if "split_ratio" not in df.columns:
        return df.with_columns(pl.lit(1.0).alias("split_factor"))

    return (
        df.sort(["ticker", "date"])
        .with_columns(
            pl.when(pl.col("split_ratio").is_null() | (pl.col("split_ratio") <= 0))
            .then(1.0)
            .otherwise(pl.col("split_ratio"))
            .alias("_ratio")
        )
        .with_columns(
            pl.col("_ratio").cum_prod().over("ticker").alias("_cum_inclusive"),
            pl.col("_ratio").product().over("ticker").alias("_total"),
        )
        .with_columns((pl.col("_total") / pl.col("_cum_inclusive")).alias("split_factor"))
        .drop("_ratio", "_cum_inclusive", "_total")
    )


def fetch_prices(tickers: list[str], *, start: date | None = None) -> pl.DataFrame:
    """Download daily bars for `tickers` and normalise them to the price schema."""
    start = start or SETTINGS.universe.history_start
    wanted = list(dict.fromkeys(tickers + list(SETTINGS.universe.extra_tickers)))
    collected_at = datetime.now(UTC)

    frames: list[pl.DataFrame] = []
    for i in range(0, len(wanted), BATCH):
        chunk = wanted[i : i + BATCH]
        got = _download_batch(chunk, start)
        if got is None:
            log.warning("prices: empty response for batch %d-%d", i, i + len(chunk))
            continue
        frames.append(got)
        log.info("prices: batch %d-%d -> %d rows", i, i + len(chunk), got.height)

    if not frames:
        raise RuntimeError("no price data returned for any ticker")

    df = pl.concat(frames, how="diagonal_relaxed")
    df = (
        df.filter(pl.col("close_raw") > 0)
        .unique(subset=["ticker", "date"], keep="last")
        .sort(["ticker", "date"])
    )
    df = add_split_factor(df)

    # Traded value from the *raw* close. Using the dividend-adjusted close makes
    # long-standing dividend payers look progressively less liquid the further
    # back you look, which biases a liquidity screen against exactly the kind of
    # company this app is meant to find.
    df = df.with_columns(
        (pl.col("close_raw") * pl.col("volume")).alias("dollar_volume"),
        pl.lit(SOURCE).alias("source"),
        pl.lit(collected_at).alias("collected_at"),
    )

    out = coerce(df, PRICES)
    log.info(
        "prices: %d rows, %d tickers, %s to %s",
        out.height, out["ticker"].n_unique(), out["date"].min(), out["date"].max(),
    )
    return out


def trading_sessions(prices: pl.DataFrame, benchmark: str | None = None) -> list[date]:
    """The application's trading calendar: dates the benchmark traded."""
    benchmark = benchmark or SETTINGS.universe.benchmark
    sub = prices.filter(pl.col("ticker") == benchmark)
    if sub.is_empty():
        sub = prices
    return sub["date"].unique().sort().to_list()


def median_dollar_volume(prices: pl.DataFrame, *, days: int = 60) -> pl.DataFrame:
    """Recent median traded value per ticker, used to pick primary share classes."""
    if prices.is_empty():
        return pl.DataFrame(schema={"ticker": pl.Utf8, "dollar_volume": pl.Float64})
    last = prices["date"].max()
    cutoff = last - __import__("datetime").timedelta(days=days * 2)
    return (
        prices.filter(pl.col("date") >= cutoff)
        .group_by("ticker")
        .agg(pl.col("dollar_volume").median().alias("dollar_volume"))
        .sort("dollar_volume", descending=True)
    )
