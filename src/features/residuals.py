# src/features/residuals.py
from __future__ import annotations
import polars as pl

def _ensure_date(df: pl.DataFrame) -> pl.DataFrame:
    if df.schema.get("asof") == pl.Date:
        return df
    if df.schema.get("asof") == pl.Datetime:
        return df.with_columns(pl.col("asof").dt.date().alias("asof"))
    return df.with_columns(pl.col("asof").str.to_date(format="%Y-%m-%d").alias("asof"))

def add_market_sector_returns(
    df: pl.DataFrame,
    sector_col: str | None = None,
) -> pl.DataFrame:
    """Adds cross-sectional daily market return and (optionally) sector return columns."""
    df = _ensure_date(df).sort(["asof", "entity"])

    # market equal-weight daily return (mean of entity-level ret_1d per date)
    mkt = (
        df.select(["asof", "ret_1d"])
          .group_by("asof").agg(pl.col("ret_1d").mean().alias("mkt_ret_1d"))
    )
    out = df.join(mkt, on="asof", how="left")

    if sector_col and sector_col in df.columns:
        sec = (
            df.select(["asof", sector_col, "ret_1d"])
              .group_by(["asof", sector_col])
              .agg(pl.col("ret_1d").mean().alias("sector_ret_1d"))
        )
        out = out.join(sec, on=["asof", sector_col], how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("sector_ret_1d"))
    return out

def add_residual_features(
    df: pl.DataFrame,
    window: int = 60,
    sector_col: str | None = None,
    prefix: str = "resid",
) -> pl.DataFrame:
    """
    Rolling OLS with intercept:
      y = ret_1d, X = [mkt_ret_1d] (+ sector_ret_1d if provided)

    Produces (per entity):
      - beta_mkt_{W}d, beta_sec_{W}d (if sector provided), alpha_{W}d
      - resid_1d (daily regression residual)
      - resid_cum_20d, resid_cum_60d (cumulative residual momentum)
      - resid_vol_20d, resid_vol_60d (residual volatility)
    """
    assert "ret_1d" in df.columns, "ret_1d must exist (built in signal_features)"
    out = add_market_sector_returns(df, sector_col)

    # rolling means (for intercept handling)
    def _roll_mean(col: str) -> pl.Expr:
        return pl.col(col).rolling_mean(window).over("entity")

    # helpers
    y   = "ret_1d"
    x1  = "mkt_ret_1d"
    x2  = "sector_ret_1d" if (sector_col and sector_col in out.columns) else None

    # center variables over rolling window (with intercept, this is the stable way)
    out = out.with_columns([
        _roll_mean(y).alias("_ybar"),
        _roll_mean(x1).alias("_x1bar"),
    ])
    if x2:
        out = out.with_columns(_roll_mean(x2).alias("_x2bar"))

    # centered series
    out = out.with_columns([
        (pl.col(y)  - pl.col("_ybar")).alias("_yc"),
        (pl.col(x1) - pl.col("_x1bar")).alias("_x1c"),
    ])
    if x2:
        out = out.with_columns((pl.col(x2) - pl.col("_x2bar")).alias("_x2c"))

    # rolling covariances/variances
    out = out.with_columns([
        (pl.col("_x1c") * pl.col("_x1c")).rolling_mean(window).over("entity").alias("_v11"),
        (pl.col("_x1c") * pl.col("_yc")).rolling_mean(window).over("entity").alias("_c1y"),
    ])
    if x2:
        out = out.with_columns([
            (pl.col("_x2c") * pl.col("_x2c")).rolling_mean(window).over("entity").alias("_v22"),
            (pl.col("_x1c") * pl.col("_x2c")).rolling_mean(window).over("entity").alias("_v12"),
            (pl.col("_x2c") * pl.col("_yc")).rolling_mean(window).over("entity").alias("_c2y"),
        ])

    # betas (handle 1D or 2D regressors)
    if not x2:
        # single regressor
        out = out.with_columns(
            (pl.col("_c1y") / (pl.col("_v11") + 1e-12)).alias(f"beta_mkt_{window}d")
        )
        # alpha = ybar - beta * xbar
        out = out.with_columns(
            (pl.col("_ybar") - pl.col(f"beta_mkt_{window}d") * pl.col("_x1bar")).alias(f"alpha_{window}d")
        )
        # residual
        out = out.with_columns(
            (pl.col(y) - (pl.col(f"alpha_{window}d") + pl.col(f"beta_mkt_{window}d") * pl.col(x1))).alias(f"{prefix}_1d")
        )
    else:
        # 2D regression: solve (X'X)^{-1} X'y using rolling moments
        # determinant
        out = out.with_columns(
            (pl.col("_v11") * pl.col("_v22") - pl.col("_v12") * pl.col("_v12") + 1e-18).alias("_det")
        )
        beta1 = (pl.col("_c1y") * pl.col("_v22") - pl.col("_v12") * pl.col("_c2y")) / pl.col("_det")
        beta2 = (pl.col("_c2y") * pl.col("_v11") - pl.col("_v12") * pl.col("_c1y")) / pl.col("_det")

        out = out.with_columns([
            beta1.alias(f"beta_mkt_{window}d"),
            beta2.alias(f"beta_sec_{window}d"),
        ])
        # alpha = ybar - b1*x1bar - b2*x2bar
        out = out.with_columns(
            (pl.col("_ybar")
             - pl.col(f"beta_mkt_{window}d") * pl.col("_x1bar")
             - pl.col(f"beta_sec_{window}d") * pl.col("_x2bar")).alias(f"alpha_{window}d")
        )
        # residual
        out = out.with_columns(
            (pl.col(y)
             - (pl.col(f"alpha_{window}d")
                + pl.col(f"beta_mkt_{window}d") * pl.col(x1)
                + pl.col(f"beta_sec_{window}d") * pl.col(x2))
            ).alias(f"{prefix}_1d")
        )

    # residual momentum & risk features
    out = out.with_columns([
        pl.col(f"{prefix}_1d").rolling_mean(20).over("entity").alias(f"{prefix}_cum_20d"),
        pl.col(f"{prefix}_1d").rolling_mean(60).over("entity").alias(f"{prefix}_cum_60d"),
        pl.col(f"{prefix}_1d").rolling_std(20).over("entity").alias(f"{prefix}_vol_20d"),
        pl.col(f"{prefix}_1d").rolling_std(60).over("entity").alias(f"{prefix}_vol_60d"),
    ])

    # cleanup temps
    drop_tmp = [c for c in out.columns if c.startswith("_")]
    return out.drop(drop_tmp)
