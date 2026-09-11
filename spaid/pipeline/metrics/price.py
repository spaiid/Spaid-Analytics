"""Price-derived metrics: momentum, trend, relative strength, risk and liquidity.

Every window here is strictly backward-looking and includes the current bar, so
a metric dated `t` uses nothing the market had not printed by the close of `t`.

Returns are computed from the total-return series (`close_adj`), because a
momentum signal should not show a gap on every ex-dividend date. Screens and
market capitalisation use the traded price (`close_raw`). The two are kept
rigorously apart.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from spaid.config.settings import SECTOR_ETF, SETTINGS

log = logging.getLogger(__name__)

EPS = 1e-12
ANNUALISE = float(np.sqrt(SETTINGS.trading_days_per_year))

# Beta is estimated over three years rather than one. A one-year window is too
# short to separate a stock's market sensitivity from a single year's idiosyncratic
# path, and the estimates it produces are not stable enough to set a discount rate.
BETA_WINDOW = SETTINGS.trading_days_per_year * 3

# Weight on the raw estimate in the Blume adjustment; the remainder shrinks
# toward the market beta of one.
BLUME_WEIGHT = 0.67


# ---------------------------------------------------------------------------
# Pure numeric kernels (tested directly)
# ---------------------------------------------------------------------------


def rolling_max_drawdown(close: np.ndarray, window: int) -> np.ndarray:
    """Worst peak-to-trough decline *within* each trailing window.

    The naive formulation -- rolling minimum of (price / rolling maximum) -- is
    not this. It lets the trough compare against a peak up to two windows back,
    so a stock that fell 30% eighteen months ago and has recovered steadily still
    reports a 30% one-year drawdown. The peak must be constrained to the same
    window as the trough, which is what the cumulative maximum along each window
    row does here.
    """
    n = close.size
    out = np.full(n, np.nan)
    if n < window or window < 2:
        return out

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(close, window)
    peaks = np.maximum.accumulate(windows, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(peaks > 0, windows / peaks - 1.0, np.nan)
    out[window - 1 :] = np.nanmin(drawdowns, axis=1)
    return out


def downside_deviation(returns: np.ndarray, window: int, *, min_samples: int = 40) -> np.ndarray:
    """Root-mean-square of returns below zero, annualised.

    This is the textbook semideviation, `sqrt(mean(min(r, 0)^2))`, not the
    standard deviation of the subset of negative returns. The distinction
    matters: a stock whose down days are uniformly -1% has near-zero dispersion
    among its down days but real downside risk, and the dispersion measure
    reports it as safe.
    """
    n = returns.size
    out = np.full(n, np.nan)
    if n < window:
        return out

    from numpy.lib.stride_tricks import sliding_window_view

    negative = np.minimum(np.nan_to_num(returns, nan=0.0), 0.0) ** 2
    valid = (~np.isnan(returns)).astype(float)

    windows = sliding_window_view(negative, window)
    counts = sliding_window_view(valid, window).sum(axis=1)
    means = windows.sum(axis=1) / np.maximum(counts, 1.0)
    result = np.sqrt(means) * ANNUALISE
    result[counts < min_samples] = np.nan
    out[window - 1 :] = result
    return out


def downside_beta(
    returns: np.ndarray, market: np.ndarray, window: int, *, min_samples: int = 40
) -> np.ndarray:
    """Beta measured only on days the market fell.

    What a holder actually cares about is how the stock behaves when the market
    is falling, which is not always what full-sample beta says.
    """
    n = returns.size
    out = np.full(n, np.nan)
    if n < window:
        return out

    from numpy.lib.stride_tricks import sliding_window_view

    mask = (market < 0) & ~np.isnan(returns) & ~np.isnan(market)
    r = np.where(mask, np.nan_to_num(returns, nan=0.0), 0.0)
    m = np.where(mask, np.nan_to_num(market, nan=0.0), 0.0)
    w = mask.astype(float)

    rw = sliding_window_view(r, window)
    mw = sliding_window_view(m, window)
    ww = sliding_window_view(w, window)

    counts = ww.sum(axis=1)
    safe = np.maximum(counts, 1.0)
    mean_r = rw.sum(axis=1) / safe
    mean_m = mw.sum(axis=1) / safe
    cov = (rw * mw).sum(axis=1) / safe - mean_r * mean_m
    var = (mw * mw).sum(axis=1) / safe - mean_m * mean_m

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(var > EPS, cov / var, np.nan)
    beta[counts < min_samples] = np.nan
    out[window - 1 :] = beta
    return out


def amihud_illiquidity(
    returns: np.ndarray, dollar_volume: np.ndarray, window: int, *, min_samples: int = 40
) -> np.ndarray:
    """Average absolute return per million dollars traded.

    Days with no reported volume are excluded rather than divided by. Yahoo
    occasionally reports zero volume alongside a real price move; dividing by a
    small epsilon produces a value around 1e19 which then contaminates the
    rolling mean for the next three months and pins the company at the bottom of
    the liquidity ranking.
    """
    n = returns.size
    out = np.full(n, np.nan)
    if n < window:
        return out

    from numpy.lib.stride_tricks import sliding_window_view

    usable = (dollar_volume > 0) & ~np.isnan(returns) & ~np.isnan(dollar_volume)
    ratio = np.where(usable, np.abs(np.nan_to_num(returns)) / np.where(usable, dollar_volume, 1.0), 0.0)
    ratio = ratio * 1e6
    weights = usable.astype(float)

    rw = sliding_window_view(ratio, window)
    ww = sliding_window_view(weights, window)
    counts = ww.sum(axis=1)
    means = rw.sum(axis=1) / np.maximum(counts, 1.0)
    means[counts < min_samples] = np.nan
    out[window - 1 :] = means
    return out


# ---------------------------------------------------------------------------
# Frame-level construction
# ---------------------------------------------------------------------------


def _benchmark_returns(prices: pl.DataFrame, ticker: str, alias: str) -> pl.DataFrame:
    sub = (
        prices.filter(pl.col("ticker") == ticker)
        .select(["date", "close_adj"])
        .sort("date")
    )
    if sub.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, alias: pl.Float64, f"{alias}_close": pl.Float64})
    return sub.with_columns(
        (pl.col("close_adj") / pl.col("close_adj").shift(1) - 1.0).alias(alias),
        pl.col("close_adj").alias(f"{alias}_close"),
    ).select(["date", alias, f"{alias}_close"])


def build_price_metrics(
    prices: pl.DataFrame,
    *,
    sectors: pl.DataFrame | None = None,
    benchmark: str | None = None,
) -> pl.DataFrame:
    """One row per ticker-date with every price-derived metric.

    `sectors` maps ticker to GICS sector so that each name can be compared with
    its own sector exchange-traded fund, which is what separates a good company
    from a good sector.
    """
    benchmark = benchmark or SETTINGS.universe.benchmark
    market = _benchmark_returns(prices, benchmark, "mkt_ret")

    etf_tickers = sorted(set(SECTOR_ETF.values()))
    etf_closes = (
        prices.filter(pl.col("ticker").is_in(etf_tickers))
        .select(["date", "ticker", "close_adj"])
        .rename({"ticker": "sector_etf", "close_adj": "sector_close"})
    )

    excluded = set(etf_tickers) | {benchmark, SETTINGS.universe.momentum_benchmark}
    df = (
        prices.filter(~pl.col("ticker").is_in(list(excluded)))
        .sort(["ticker", "date"])
        .join(market, on="date", how="left")
    )

    if sectors is not None and not sectors.is_empty():
        mapping = sectors.select(["ticker", "sector"]).with_columns(
            pl.col("sector")
            .replace_strict(SECTOR_ETF, default=None)
            .alias("sector_etf")
        )
        df = df.join(mapping.select(["ticker", "sector_etf"]), on="ticker", how="left")
        df = df.join(etf_closes, on=["date", "sector_etf"], how="left")
    else:
        df = df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("sector_etf"),
            pl.lit(None, dtype=pl.Float64).alias("sector_close"),
        )

    px = pl.col("close_adj")
    df = df.with_columns((px / px.shift(1).over("ticker") - 1.0).alias("ret_1d"))

    # ---- momentum ---------------------------------------------------------
    df = df.with_columns(
        (px.shift(21).over("ticker") / px.shift(252).over("ticker") - 1.0).alias("mom_12_1"),
        (px.shift(21).over("ticker") / px.shift(126).over("ticker") - 1.0).alias("mom_6_1"),
        (px / px.shift(63).over("ticker") - 1.0).alias("ret_3m"),
        (px / px.shift(126).over("ticker") - 1.0).alias("_ret_6m"),
        (px / px.shift(252).over("ticker") - 1.0).alias("_ret_12m"),
    )

    # ---- relative strength ------------------------------------------------
    df = df.with_columns(
        (
            pl.col("mkt_ret_close") / pl.col("mkt_ret_close").shift(126).over("ticker") - 1.0
        ).alias("_mkt_6m"),
        (
            pl.col("sector_close") / pl.col("sector_close").shift(126).over("ticker") - 1.0
        ).alias("_sector_6m"),
    ).with_columns(
        (pl.col("_ret_6m") - pl.col("_mkt_6m")).alias("rel_strength_spy"),
        (pl.col("_ret_6m") - pl.col("_sector_6m")).alias("rel_strength_sector"),
    )

    # ---- trend ------------------------------------------------------------
    df = df.with_columns(
        px.rolling_mean(50).over("ticker").alias("_ma50"),
        px.rolling_mean(200).over("ticker").alias("_ma200"),
        px.rolling_max(252).over("ticker").alias("_high252"),
    ).with_columns(
        (px / (pl.col("_ma50") + EPS) - 1.0).alias("px_over_ma50"),
        (px / (pl.col("_ma200") + EPS) - 1.0).alias("px_over_ma200"),
        (px / (pl.col("_high252") + EPS) - 1.0).alias("near_52w_high"),
        (
            pl.col("_ma200") / (pl.col("_ma200").shift(63).over("ticker") + EPS) - 1.0
        ).alias("ma200_slope"),
    )

    # ---- momentum consistency --------------------------------------------
    # Share of the last twelve months in which the stock beat the market,
    # measured on monthly blocks so one explosive week cannot stand in for a
    # year of steady outperformance.
    df = df.with_columns(
        (pl.col("ret_1d") - pl.col("mkt_ret")).alias("_excess_1d")
    ).with_columns(
        pl.col("_excess_1d").rolling_sum(21).over("ticker").alias("_excess_21d")
    ).with_columns(
        (pl.col("_excess_21d") > 0)
        .cast(pl.Float64)
        .rolling_mean(252, min_samples=126)
        .over("ticker")
        .alias("momentum_consistency")
    )

    # ---- volatility -------------------------------------------------------
    df = df.with_columns(
        (pl.col("ret_1d").rolling_std(252, min_samples=120).over("ticker") * ANNUALISE).alias(
            "volatility"
        ),
        (pl.col("ret_1d").rolling_std(63, min_samples=40).over("ticker") * ANNUALISE).alias(
            "volatility_3m"
        ),
    )

    # ---- liquidity --------------------------------------------------------
    df = df.with_columns(
        pl.col("dollar_volume").rolling_median(60, min_samples=20).over("ticker").alias(
            "dollar_volume_median"
        ),
    )

    # ---- per-ticker numeric kernels --------------------------------------
    df = _apply_numeric_kernels(df)

    keep = [
        "ticker", "date", "close_raw", "close_adj", "ret_1d",
        "mom_12_1", "mom_6_1", "ret_3m",
        "rel_strength_spy", "rel_strength_sector",
        "px_over_ma50", "px_over_ma200", "ma200_slope",
        "near_52w_high", "momentum_consistency",
        "volatility", "volatility_3m", "beta", "beta_raw", "downside_beta",
        "downside_deviation", "max_drawdown_1y", "max_drawdown_5y",
        "dollar_volume", "dollar_volume_median", "amihud",
        "split_factor",
    ]
    out = df.select([c for c in keep if c in df.columns])
    log.info("price metrics: %d rows, %d tickers", out.height, out["ticker"].n_unique())
    return out


def _apply_numeric_kernels(df: pl.DataFrame) -> pl.DataFrame:
    """Run the windowed numpy kernels per ticker and stitch the results back."""
    year = SETTINGS.trading_days_per_year
    pieces: list[pl.DataFrame] = []

    for (ticker,), group in df.group_by(["ticker"], maintain_order=True):
        g = group.sort("date")
        close = g["close_adj"].to_numpy().astype(float)
        rets = g["ret_1d"].to_numpy().astype(float)
        mkt = g["mkt_ret"].to_numpy().astype(float)
        dv = g["dollar_volume"].fill_null(0.0).to_numpy().astype(float)

        pieces.append(
            g.with_columns(
                pl.Series("max_drawdown_1y", rolling_max_drawdown(close, year)),
                pl.Series("max_drawdown_5y", rolling_max_drawdown(close, year * 5)),
                pl.Series("downside_deviation", downside_deviation(rets, year)),
                pl.Series("downside_beta", downside_beta(rets, mkt, BETA_WINDOW)),
                pl.Series("amihud", amihud_illiquidity(rets, dv, 63)),
                pl.Series("beta_raw", _rolling_beta(rets, mkt, BETA_WINDOW)),
            )
        )

    if not pieces:
        return df
    out = pl.concat(pieces, how="vertical_relaxed")

    # Blume adjustment. A raw regression beta is a noisy estimate of a quantity
    # that is known to revert toward one, and over a single year the noise
    # dominates: Coca-Cola prints a *negative* beta, which would make a
    # defensive staple look like a hedge against the market it plainly is not.
    # Shrinking two thirds of the way toward the raw estimate is the standard
    # correction and is what makes beta usable in a cost-of-capital calculation.
    return out.with_columns(
        pl.when(pl.col("beta_raw").is_not_null())
        .then(BLUME_WEIGHT * pl.col("beta_raw") + (1.0 - BLUME_WEIGHT))
        .otherwise(None)
        .alias("beta")
    )


def _rolling_beta(returns: np.ndarray, market: np.ndarray, window: int, *, min_samples: int = 120) -> np.ndarray:
    """Covariance over variance on a trailing window, both population moments."""
    n = returns.size
    out = np.full(n, np.nan)
    if n < window:
        return out

    from numpy.lib.stride_tricks import sliding_window_view

    valid = ~np.isnan(returns) & ~np.isnan(market)
    r = np.where(valid, np.nan_to_num(returns), 0.0)
    m = np.where(valid, np.nan_to_num(market), 0.0)
    w = valid.astype(float)

    rw = sliding_window_view(r, window)
    mw = sliding_window_view(m, window)
    ww = sliding_window_view(w, window)

    counts = ww.sum(axis=1)
    safe = np.maximum(counts, 1.0)
    mean_r = rw.sum(axis=1) / safe
    mean_m = mw.sum(axis=1) / safe
    cov = (rw * mw).sum(axis=1) / safe - mean_r * mean_m
    var = (mw * mw).sum(axis=1) / safe - mean_m * mean_m

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(var > EPS, cov / var, np.nan)
    beta[counts < min_samples] = np.nan
    out[window - 1 :] = beta
    return out
