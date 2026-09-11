"""Price-and-volume features.

Every window here is strictly backward-looking and includes the current bar, so
a feature dated `t` uses nothing the market had not printed by the close of
`t`. The forward target lives in `assemble.py` and is the only column that
looks ahead.

Note on moving averages: these are genuine exponential means where the name
says exponential. (The previous codebase labelled simple rolling means `ema12`
and `ema26` and then differenced them into a "MACD", which was not a MACD.)
"""
from __future__ import annotations

import logging

import polars as pl

from spaid.data.universe import benchmark_ticker

log = logging.getLogger(__name__)

EPS = 1e-12


def _rolling_beta(ret: str, mkt: str, window: int) -> pl.Expr:
    """Cov(r, m) / Var(m) over a trailing window, per ticker."""
    mean_r = pl.col(ret).rolling_mean(window).over("ticker")
    mean_m = pl.col(mkt).rolling_mean(window).over("ticker")
    mean_rm = (pl.col(ret) * pl.col(mkt)).rolling_mean(window).over("ticker")
    mean_mm = (pl.col(mkt) ** 2).rolling_mean(window).over("ticker")
    cov = mean_rm - mean_r * mean_m
    var = mean_mm - mean_m**2
    return cov / (var + EPS)


def build_technical(prices: pl.DataFrame) -> pl.DataFrame:
    """Return one row per ticker-date with the technical feature block."""
    bench = benchmark_ticker()

    market = (
        prices.filter(pl.col("ticker") == bench)
        .select(["date", "close"])
        .sort("date")
        .with_columns((pl.col("close") / pl.col("close").shift(1) - 1.0).alias("mkt_ret"))
        .select(["date", "mkt_ret"])
    )

    df = (
        prices.filter(pl.col("ticker") != bench)
        .sort(["ticker", "date"])
        .join(market, on="date", how="left")
    )

    # --- daily returns, the base of nearly everything else -----------------
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("ret_1d")
    )

    # --- momentum ----------------------------------------------------------
    df = df.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(5).over("ticker") - 1.0).alias("ret_5d"),
            (pl.col("close") / pl.col("close").shift(63).over("ticker") - 1.0).alias("ret_63d"),
            # 12-1 and 6-1: skip the most recent month, which is dominated by
            # short-term reversal rather than momentum
            (
                pl.col("close").shift(21).over("ticker") / pl.col("close").shift(252).over("ticker") - 1.0
            ).alias("mom_12_1"),
            (
                pl.col("close").shift(21).over("ticker") / pl.col("close").shift(126).over("ticker") - 1.0
            ).alias("mom_6_1"),
        ]
    )

    # --- trend -------------------------------------------------------------
    df = df.with_columns(
        [
            pl.col("close").rolling_mean(50).over("ticker").alias("_ma50"),
            pl.col("close").rolling_mean(200).over("ticker").alias("_ma200"),
            pl.col("close").rolling_max(252).over("ticker").alias("_hi252"),
            (pl.col("ret_1d") > 0).cast(pl.Float64).rolling_mean(63).over("ticker").alias("up_day_ratio"),
        ]
    ).with_columns(
        [
            (pl.col("close") / (pl.col("_ma50") + EPS) - 1.0).alias("px_over_ma50"),
            (pl.col("close") / (pl.col("_ma200") + EPS) - 1.0).alias("px_over_ma200"),
            (pl.col("_ma50") / (pl.col("_ma200") + EPS) - 1.0).alias("ma_50_200"),
            (pl.col("close") / (pl.col("_hi252") + EPS) - 1.0).alias("near_52w_high"),
        ]
    )

    # --- volatility & market sensitivity ------------------------------------
    ann = 252**0.5
    df = df.with_columns(
        [
            (pl.col("ret_1d").rolling_std(63).over("ticker") * ann).alias("vol_63d"),
            (
                pl.when(pl.col("ret_1d") < 0).then(pl.col("ret_1d")).otherwise(None)
                .rolling_std(126, min_samples=20).over("ticker") * ann
            ).alias("downside_vol_126d"),
            _rolling_beta("ret_1d", "mkt_ret", 252).alias("beta_252d"),
        ]
    )

    # residual return stream, then residual momentum and idiosyncratic vol
    df = df.with_columns(
        (pl.col("ret_1d") - pl.col("beta_252d") * pl.col("mkt_ret")).alias("_resid")
    ).with_columns(
        [
            pl.col("_resid").rolling_sum(126).over("ticker").alias("_resid_sum_126"),
            (pl.col("_resid").rolling_std(126).over("ticker") * ann).alias("idio_vol_126d"),
        ]
    ).with_columns(
        # 6-1 construction on the residual stream too
        (
            pl.col("_resid_sum_126") - pl.col("_resid").rolling_sum(21).over("ticker")
        ).alias("resid_mom_126d")
    )

    # one-year drawdown (negative number; closer to zero is better)
    df = df.with_columns(
        (pl.col("close") / (pl.col("close").rolling_max(252).over("ticker") + EPS) - 1.0)
        .rolling_min(252)
        .over("ticker")
        .alias("max_drawdown_252d")
    )

    # --- liquidity ----------------------------------------------------------
    df = df.with_columns(
        [
            pl.col("dollar_volume").rolling_median(21).over("ticker").alias("_dv21"),
            (
                (pl.col("ret_1d").abs() / (pl.col("dollar_volume") + EPS) * 1e9)
                .rolling_mean(63)
                .over("ticker")
            ).alias("amihud_illiq"),
        ]
    ).with_columns(
        [
            (pl.col("_dv21") + 1.0).log().alias("log_dollar_volume"),
            (
                pl.col("dollar_volume").rolling_mean(21).over("ticker")
                / (pl.col("dollar_volume").rolling_mean(252).over("ticker") + EPS)
                - 1.0
            ).alias("volume_trend"),
        ]
    )

    keep = [
        "date", "ticker", "close", "ret_1d", "dollar_volume", "_dv21",
        "ret_5d", "ret_63d", "mom_12_1", "mom_6_1", "resid_mom_126d",
        "px_over_ma50", "px_over_ma200", "ma_50_200", "near_52w_high", "up_day_ratio",
        "vol_63d", "downside_vol_126d", "beta_252d", "idio_vol_126d", "max_drawdown_252d",
        "log_dollar_volume", "amihud_illiq", "volume_trend",
    ]
    out = df.select([c for c in keep if c in df.columns]).rename({"_dv21": "dollar_volume_21d"})
    log.info("technical: %d rows, %d tickers", out.height, out["ticker"].n_unique())
    return out
