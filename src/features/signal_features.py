from __future__ import annotations
import os, glob
from datetime import date, timedelta
import polars as pl

def _root() -> str:
    return os.environ.get("SF_DATA_ROOT", "_data")

def _paths(kind: str, start: str, end: str):
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    cur = d0
    while cur <= d1:
        yield os.path.join(_root(), "signalgraph", f"kind={kind}", f"asof={cur.isoformat()}", "*.parquet")
        cur += timedelta(days=1)

def load_signals(kind: str, start: str, end: str) -> pl.DataFrame:
    files: list[str] = []
    for pat in _paths(kind, start, end):
        files.extend(glob.glob(pat))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(p) for p in files], how="diagonal_relaxed")

def build_daily_features(start: str, end: str) -> pl.DataFrame:
    price = load_signals("market.price", start, end)
    vol = load_signals("market.volume", start, end)
    rv = load_signals("market.volatility", start, end)

    if price.is_empty():
        return pl.DataFrame()

    base = (
        price.select(["asof", "entity", "value"])
        .rename({"value": "close"})
        .with_columns(pl.col("close").cast(pl.Float64))
    )
    vol = (
        vol.select(["asof", "entity", "value"])
        .rename({"value": "volume"})
        .with_columns(pl.col("volume").cast(pl.Float64))
    )
    rv = (
        rv.select(["asof", "entity", "value"])
        .rename({"value": "rv_range_close"})
        .with_columns(pl.col("rv_range_close").cast(pl.Float64))
    )

    # Join & order
    out = (
        base.join(vol, on=["asof", "entity"], how="left")
        .join(rv, on=["asof", "entity"], how="left")
        .sort(["entity", "asof"])
    )

    # ---- Pass 1: core features ----
    out = out.with_columns(
        [
            # returns & momentum (explicit shift form for portability)
            (pl.col("close") / pl.col("close").shift(1).over("entity") - 1).alias("ret_1d"),
            (pl.col("close") / pl.col("close").shift(5).over("entity") - 1).alias("ret_5d"),
            (pl.col("close") / pl.col("close").shift(20).over("entity") - 1).alias("ret_20d"),
            (pl.col("close") / pl.col("close").shift(60).over("entity") - 1).alias("ret_60d"),
            # moving averages
            pl.col("close").rolling_mean(10).over("entity").alias("ma10"),
            pl.col("close").rolling_mean(20).over("entity").alias("ma20"),
            pl.col("close").rolling_mean(50).over("entity").alias("ma50"),
            (pl.col("close") / (pl.col("close").rolling_mean(20).over("entity") + 1e-12) - 1).alias(
                "px_over_ma20"
            ),
            # volume structure
            pl.col("volume").rolling_mean(20).over("entity").alias("vol_ma20"),
            (pl.col("volume") / (pl.col("volume").rolling_mean(20).over("entity") + 1e-12) - 1).alias(
                "vol_rel_20"
            ),
            # range-based vol smoothing
            pl.col("rv_range_close").rolling_mean(20).over("entity").alias("rv_rc_20"),
        ]
    )

    # ---- Pass 2: needs ret_1d computed above ----
    out = out.with_columns(
        [
            pl.col("ret_1d").rolling_std(20).over("entity").alias("vol_cc_20"),
            pl.col("ret_1d").rolling_std(60).over("entity").alias("vol_cc_60"),
        ]
    )

    # ---- RSI(14): temporary gain/loss columns, then drop ----
    out = out.with_columns(
        [
            pl.when(pl.col("ret_1d") > 0)
            .then(pl.col("ret_1d"))
            .otherwise(0.0)
            .rolling_mean(14)
            .over("entity")
            .alias("_gain14"),

            pl.when(pl.col("ret_1d") < 0)
            .then(-pl.col("ret_1d"))
            .otherwise(0.0)
            .rolling_mean(14)
            .over("entity")
            .alias("_loss14"),
        ]
    )

    out = out.with_columns(
        [
            (
                100
                - (100 / (1 + (pl.col("_gain14") / (pl.col("_loss14") + 1e-12))))
            ).alias("rsi14")
        ]
    ).drop(["_gain14", "_loss14"])

    # ---- MACD using SMAs as EMA stand-ins ----
    out = out.with_columns(
        [
            pl.col("close").rolling_mean(12).over("entity").alias("ema12"),
            pl.col("close").rolling_mean(26).over("entity").alias("ema26"),
        ]
    )
    out = out.with_columns(
        [
            (pl.col("ema12") - pl.col("ema26")).alias("macd"),
            pl.col("macd").rolling_mean(9).over("entity").alias("macd_signal"),
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"),
        ]
    )

    return out



def add_forward_return(df: pl.DataFrame, horizon: int = 20) -> pl.DataFrame:
    if df.is_empty():
        return df
    return (df.with_columns([
                pl.col("close").shift(-horizon).over("entity").alias("close_fwd")
            ])
            .with_columns([
                ((pl.col("close_fwd")/pl.col("close")) - 1.0).alias(f"ret_fwd_{horizon}d")
            ])
            .drop("close_fwd"))
