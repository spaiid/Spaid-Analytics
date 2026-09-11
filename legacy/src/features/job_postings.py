import polars as pl
from common.io import read_parquet_partition, write_parquet_partition
from entity.resolver import resolve_ticker

def job_posting_features(dt: str) -> str:
    # Read normalized rows for this date
    df = read_parquet_partition("normalized", "job_postings", dt)
    if df is None or df.is_empty():
        return ""

    # Map to tickers and keep confident matches
    df = resolve_ticker(df).filter(
        (pl.col("ticker").is_not_null()) & (pl.col("confidence") >= 0.8)
    )

    # Aggregate to per-ticker/day
    daily = df.group_by(["ticker", "dt"]).agg(pl.count().alias("postings"))

    # Emit a simple feature table (you can add more later: 7d sums, z-scores, etc.)
    out = (
        daily
        .with_columns([
            pl.lit("job_postings").alias("feature_id"),
            pl.col("postings").cast(pl.Int64).alias("value"),
        ])
        .select(["ticker", "dt", "feature_id", "value"])
    )

    return write_parquet_partition(out, "features", "job_postings", dt)
