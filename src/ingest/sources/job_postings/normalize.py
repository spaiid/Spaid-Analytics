import polars as pl
from src.common.io import read_parquet_partition, write_parquet_partition
from src.common.hashing import stable_row_hash

def normalize_job_postings(dt: str) -> str:
    df = read_parquet_partition("raw", "job_postings", dt)
    if df is None or df.is_empty():
        return ""
    # Casts + fill
    out = (
        df.with_columns([
            pl.col("platform").cast(pl.Utf8),
            pl.col("brand").cast(pl.Utf8),
            pl.col("domain").cast(pl.Utf8),
            pl.col("posting_id").cast(pl.Utf8),
            pl.col("title").cast(pl.Utf8),
            pl.col("location").cast(pl.Utf8),
            pl.col("department").cast(pl.Utf8),
            pl.col("url").cast(pl.Utf8),
            pl.col("posted_at").cast(pl.Utf8),
            pl.col("updated_at").cast(pl.Utf8),
            pl.lit(dt).alias("dt"),
        ])
    )
    # Deterministic identity: prefer platform+domain+posting_id; fallback to title+location
    out = out.with_columns([
        pl.when(pl.col("posting_id").is_not_null())
          .then(pl.struct(["platform","domain","posting_id"]))
          .otherwise(pl.struct(["platform","domain","title","location"]))
          .map_elements(stable_row_hash)
          .alias("row_id")
    ]).unique(subset=["row_id"])
    # Keep only normalized columns
    out = out.select(["dt","platform","brand","domain","posting_id","title","location","department","url","posted_at","updated_at","row_id"])
    return write_parquet_partition(out, "normalized", "job_postings", dt)
