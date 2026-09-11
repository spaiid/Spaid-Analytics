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

    # --- core columns ---
    base = (
        price.select(["asof", "entity", "value"])
             .rename({"value": "close"})
             .with_columns([
                 pl.col("asof").cast(pl.Date),
                 pl.col("entity").cast(pl.Utf8),
                 pl.col("close").cast(pl.Float64),
             ])
    )
    vol = (
        vol.select(["asof", "entity", "value"])
           .rename({"value": "volume"})
           .with_columns([
               pl.col("asof").cast(pl.Date),
               pl.col("entity").cast(pl.Utf8),
               pl.col("volume").cast(pl.Float64),
           ])
    )
    rv = (
        rv.select(["asof", "entity", "value"])
          .rename({"value": "rv_range_close"})
          .with_columns([
              pl.col("asof").cast(pl.Date),
              pl.col("entity").cast(pl.Utf8),
              pl.col("rv_range_close").cast(pl.Float64),
          ])
    )

    # Join & order
    out = (
        base.join(vol, on=["asof", "entity"], how="left")
            .join(rv, on=["asof", "entity"], how="left")
            .sort(["entity", "asof"])
    )

    # ---- Pass 1: core features (short/medium) ----
    out = out.with_columns(
        [
            # returns & momentum
            (pl.col("close") / pl.col("close").shift(1).over("entity") - 1).alias("ret_1d"),
            (pl.col("close") / pl.col("close").shift(5).over("entity") - 1).alias("ret_5d"),
            (pl.col("close") / pl.col("close").shift(20).over("entity") - 1).alias("ret_20d"),
            (pl.col("close") / pl.col("close").shift(60).over("entity") - 1).alias("ret_60d"),

            # moving averages (SMA)
            pl.col("close").rolling_mean(10).over("entity").alias("ma10"),
            pl.col("close").rolling_mean(20).over("entity").alias("ma20"),
            pl.col("close").rolling_mean(50).over("entity").alias("ma50"),

            # price vs MA
            (pl.col("close") / (pl.col("close").rolling_mean(20).over("entity") + 1e-12) - 1).alias("px_over_ma20"),

            # volume structure
            pl.col("volume").rolling_mean(20).over("entity").alias("vol_ma20"),
            (pl.col("volume") / (pl.col("volume").rolling_mean(20).over("entity") + 1e-12) - 1).alias("vol_rel_20"),

            # range-based vol smoothing (short)
            pl.col("rv_range_close").rolling_mean(20).over("entity").alias("rv_rc_20"),
        ]
    )

    # ---- Pass 2: vol from ret_1d ----
    out = out.with_columns(
        [
            pl.col("ret_1d").rolling_std(20).over("entity").alias("vol_cc_20"),
            pl.col("ret_1d").rolling_std(60).over("entity").alias("vol_cc_60"),
        ]
    )

    # ---- RSI(14) ----
    out = out.with_columns(
        [
            pl.when(pl.col("ret_1d") > 0).then(pl.col("ret_1d")).otherwise(0.0)
              .rolling_mean(14).over("entity").alias("_gain14"),
            pl.when(pl.col("ret_1d") < 0).then(-pl.col("ret_1d")).otherwise(0.0)
              .rolling_mean(14).over("entity").alias("_loss14"),
        ]
    )
    out = out.with_columns(
        [
            (100 - (100 / (1 + (pl.col("_gain14") / (pl.col("_loss14") + 1e-12))))).alias("rsi14"),
        ]
    ).drop(["_gain14", "_loss14"])

    # ---- MACD (compute in steps; don't reference new cols in same call) ----

    # 1) EMAs (SMA stand-ins)
    out = out.with_columns([
        pl.col("close").rolling_mean(12).over("entity").alias("ema12"),
        pl.col("close").rolling_mean(26).over("entity").alias("ema26"),
    ])

    # 2) MACD line
    out = out.with_columns(
        (pl.col("ema12") - pl.col("ema26")).alias("macd")
    )

    # 3) Signal line (9-period SMA of MACD)
    out = out.with_columns(
        pl.col("macd").rolling_mean(9).over("entity").alias("macd_signal")
    )

    # 4) Histogram (MACD − Signal)
    out = out.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")
    )

    # ======================================================================
    # Long-horizon additions (useful for 90d target)
    # ======================================================================

    # Longer lookback returns
    out = out.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(90).over("entity") - 1).alias("ret_90d"),
            (pl.col("close") / pl.col("close").shift(120).over("entity") - 1).alias("ret_120d"),
            (pl.col("close") / pl.col("close").shift(180).over("entity") - 1).alias("ret_180d"),
        ]
    )

    # Longer MAs and price vs longer MAs
    out = out.with_columns(
        [
            pl.col("close").rolling_mean(100).over("entity").alias("ma100"),
            pl.col("close").rolling_mean(200).over("entity").alias("ma200"),
            (pl.col("close") / (pl.col("close").rolling_mean(100).over("entity") + 1e-12) - 1).alias("px_over_ma100"),
            (pl.col("close") / (pl.col("close").rolling_mean(200).over("entity") + 1e-12) - 1).alias("px_over_ma200"),
        ]
    )

    # Longer vol windows (close-to-close and range-close)
    out = out.with_columns(
        [
            pl.col("ret_1d").rolling_std(90).over("entity").alias("vol_cc_90"),
            pl.col("rv_range_close").rolling_mean(60).over("entity").alias("rv_rc_60"),
            pl.col("rv_range_close").rolling_mean(90).over("entity").alias("rv_rc_90"),
        ]
    )

    return out

def add_forward_return(df: pl.DataFrame, horizon: int = 20) -> pl.DataFrame:
    if df.is_empty():
        return df
    return (
        df.with_columns(
            pl.col("close").shift(-horizon).over("entity").alias("close_fwd")
        )
        .with_columns(
            ((pl.col("close_fwd") / pl.col("close")) - 1.0).alias(f"ret_fwd_{horizon}d")
        )
        .drop("close_fwd")
    )
