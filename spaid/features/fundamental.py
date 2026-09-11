"""Valuation, quality, growth and capital-use features.

The join that matters is `join_asof` on the **filing date**. For every
ticker-date we take the most recent fundamentals whose 10-Q or 10-K was already
public, so a feature dated 2019-05-02 uses the numbers a reader could have
pulled from EDGAR that morning -- not the quarter that had ended but not yet
been filed.

Market capitalisation deserves its own note. It is computed from the
*split-adjusted but not dividend-adjusted* close, multiplied by the reported
share count restated onto today's share basis via `split_factor`. Using the
total-return close here would understate market cap by the whole dividend
history and quietly inflate every yield.
"""
from __future__ import annotations

import logging

import polars as pl

log = logging.getLogger(__name__)

EPS = 1e-9
YEAR = 252  # trading days


def _wide_fundamentals(fundamentals: pl.DataFrame) -> pl.DataFrame:
    """One row per (ticker, filed) with a column per concept, forward-filled."""
    wide = (
        fundamentals.pivot(
            on="concept", index=["ticker", "filed"], values="value", aggregate_function="last"
        )
        .sort(["ticker", "filed"])
    )
    concepts = [c for c in wide.columns if c not in ("ticker", "filed")]
    # A filing that reports revenue but not inventory should not blank out the
    # inventory we already knew about, hence the forward fill.
    return wide.with_columns([pl.col(c).forward_fill().over("ticker") for c in concepts])


def _safe_div(num: pl.Expr, den: pl.Expr, *, positive_den: bool = False) -> pl.Expr:
    guard = (den.abs() > EPS) if not positive_den else (den > EPS)
    return pl.when(guard).then(num / den).otherwise(None)


def build_fundamental(prices: pl.DataFrame, fundamentals: pl.DataFrame) -> pl.DataFrame:
    wide = _wide_fundamentals(fundamentals)

    panel = (
        prices.select(["date", "ticker", "close_split", "split_factor"])
        .sort(["date", "ticker"])
    )

    joined = (
        panel.sort("date")
        .join_asof(
            wide.sort("filed"),
            left_on="date",
            right_on="filed",
            by="ticker",
            strategy="backward",     # only filings already public on `date`
        )
        .sort(["ticker", "date"])
    )

    have = set(joined.columns)

    def col(name: str) -> pl.Expr:
        return pl.col(name) if name in have else pl.lit(None, dtype=pl.Float64)

    # --- market cap ---------------------------------------------------------
    joined = joined.with_columns(
        _safe_div(
            pl.col("close_split") * col("shares") * pl.col("split_factor"),
            pl.lit(1.0),
        ).alias("market_cap")
    )

    # --- derived building blocks -------------------------------------------
    joined = joined.with_columns(
        [
            (col("operating_cash_flow") - col("capex")).alias("_fcf"),
            (col("debt_long").fill_null(0.0) + col("debt_current").fill_null(0.0)).alias("_debt"),
            _safe_div(col("operating_income"), col("revenue")).alias("operating_margin"),
        ]
    )

    mcap = pl.col("market_cap")
    joined = joined.with_columns(
        [
            # Value
            _safe_div(col("net_income"), mcap, positive_den=True).alias("earnings_yield"),
            _safe_div(col("revenue"), mcap, positive_den=True).alias("sales_yield"),
            _safe_div(pl.col("_fcf"), mcap, positive_den=True).alias("fcf_yield"),
            _safe_div(col("equity"), mcap, positive_den=True).alias("book_to_price"),
            # Quality
            _safe_div(col("net_income"), col("equity"), positive_den=True).alias("roe"),
            _safe_div(col("net_income"), col("assets"), positive_den=True).alias("roa"),
            _safe_div(col("gross_profit"), col("revenue")).alias("gross_margin"),
            _safe_div(col("net_income") - col("operating_cash_flow"), col("assets"), positive_den=True)
            .alias("accruals"),
            _safe_div(pl.col("_debt"), col("equity"), positive_den=True).alias("leverage"),
            _safe_div(col("current_assets"), col("current_liabilities"), positive_den=True)
            .alias("current_ratio"),
            # Capital use — buybacks and dividends are cash outflows, so they
            # arrive positive-signed already in the SEC data.
            _safe_div(col("buyback"), mcap, positive_den=True).alias("buyback_yield"),
            _safe_div(col("dividends_paid"), mcap, positive_den=True).alias("dividend_yield"),
            _safe_div(col("capex"), col("revenue"), positive_den=True).alias("capex_intensity"),
            _safe_div(col("rnd"), col("revenue"), positive_den=True).alias("rnd_intensity"),
        ]
    )

    # --- growth: this year's TTM against the same figure a year ago ---------
    joined = joined.sort(["ticker", "date"]).with_columns(
        [
            col("revenue").shift(YEAR).over("ticker").alias("_rev_lag"),
            col("net_income").shift(YEAR).over("ticker").alias("_ni_lag"),
            pl.col("_fcf").shift(YEAR).over("ticker").alias("_fcf_lag"),
            pl.col("operating_margin").shift(YEAR).over("ticker").alias("_om_lag"),
        ]
    ).with_columns(
        [
            _safe_div(col("revenue") - pl.col("_rev_lag"), pl.col("_rev_lag").abs())
            .alias("revenue_growth"),
            # abs() in the denominator so a swing out of a loss reads as growth
            _safe_div(col("net_income") - pl.col("_ni_lag"), pl.col("_ni_lag").abs())
            .alias("earnings_growth"),
            _safe_div(pl.col("_fcf") - pl.col("_fcf_lag"), pl.col("_fcf_lag").abs())
            .alias("fcf_growth"),
            (pl.col("operating_margin") - pl.col("_om_lag")).alias("margin_trend"),
        ]
    )

    out_cols = [
        "date", "ticker", "market_cap",
        "earnings_yield", "sales_yield", "fcf_yield", "book_to_price",
        "roe", "roa", "gross_margin", "operating_margin", "accruals", "leverage", "current_ratio",
        "revenue_growth", "earnings_growth", "fcf_growth", "margin_trend",
        "buyback_yield", "dividend_yield", "capex_intensity", "rnd_intensity",
    ]
    out = joined.select([c for c in out_cols if c in joined.columns])

    covered = out.select(
        [(pl.col(c).is_not_null().mean() * 100).alias(c) for c in out_cols if c not in ("date", "ticker")]
    ).to_dicts()[0]
    log.info("fundamental: %d rows; coverage %%: %s",
             out.height, {k: round(v, 1) for k, v in sorted(covered.items(), key=lambda x: -x[1])})
    return out
