"""Treasury yields, for classifying the rate environment of a historical period.

One series, one purpose: the regime analysis has to be able to say whether a
stretch of the backtest happened under high or low rates, and asserting that
from memory is not a measurement. The Federal Reserve's own series is free, has
no key, and goes back further than anything else here.

It is written to the raw layer rather than to a canonical table because that is
what it is -- a provider's payload, normalised only enough to be read back. It
is not a price, and forcing it into the price schema would make a yield look
like a close.
"""

from __future__ import annotations

import io
import logging
from datetime import UTC, date, datetime

import polars as pl

from spaid.config.settings import RAW
from spaid.providers.http import fetch_text

log = logging.getLogger(__name__)

SOURCE = "fred"
SERIES = "DGS10"  # ten-year constant-maturity Treasury yield, per cent
PATH = RAW / "fred_dgs10.parquet"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"


def fetch(series: str = SERIES, *, start: date = date(2010, 1, 1), cache_hours: float = 24.0) -> pl.DataFrame:
    """Download one FRED series as a date/value frame.

    Missing observations are published as a full stop, which parses to null and
    is left as null: a market holiday is not a zero interest rate.
    """
    text = fetch_text(
        URL.format(series=series, start=start.isoformat()),
        cache_hours=cache_hours,
        suffix=".csv",
    )
    raw = pl.read_csv(io.StringIO(text), null_values=["."], try_parse_dates=True)
    columns = raw.columns
    date_col = columns[0]
    value_col = next((c for c in columns[1:] if c.lower() != date_col.lower()), columns[-1])

    out = (
        raw.select(
            pl.col(date_col).cast(pl.Date).alias("date"),
            pl.col(value_col).cast(pl.Float64, strict=False).alias("value"),
        )
        .drop_nulls("value")
        .sort("date")
        .with_columns(
            pl.lit(series).alias("series"),
            pl.lit(datetime.now(UTC)).alias("collected_at"),
        )
    )
    log.info(
        "%s: %d observations from %s to %s",
        series, out.height, out["date"].min(), out["date"].max(),
    )
    return out


def refresh(*, force: bool = False) -> pl.DataFrame:
    """Fetch and cache the ten-year yield, reusing the stored copy when fresh."""
    if PATH.exists() and not force:
        stored = pl.read_parquet(PATH)
        if not stored.is_empty():
            return stored
    out = fetch()
    PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(PATH, compression="zstd")
    return out


def load() -> pl.DataFrame | None:
    """The stored series, or None when it has never been fetched."""
    if not PATH.exists():
        return None
    return pl.read_parquet(PATH)
