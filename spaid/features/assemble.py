"""Assemble the modelling panel.

Three things happen here and the order matters:

1. **Eligibility.** A name is only scored on dates when it was actually in the
   index and actually tradeable. Screening on `date_added` removes the half of
   survivorship bias we can remove: the backtest never buys a company before
   the index had picked it.

2. **Target.** Forward excess return over the benchmark, arithmetic throughout.
   (The previous codebase built the target in log space and then compounded it
   arithmetically in the backtest, which quietly mismatched every number.)

3. **Cross-sectional normalisation.** Winsorise, z-score and median-fill are all
   computed *within a single date*, across names. That is what makes them safe:
   a statistic computed from date `t` alone cannot import information from the
   future, so it needs no train/test fitting. The previous codebase imputed with
   a median taken over the whole sample including the test period, which did.
"""
from __future__ import annotations

import logging

import polars as pl

from spaid.config import SETTINGS
from spaid.data.universe import benchmark_ticker
from spaid.features.registry import FEATURES, NAMES

log = logging.getLogger(__name__)


def build_panel(
    prices: pl.DataFrame,
    technical: pl.DataFrame,
    fundamental: pl.DataFrame,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    """Join every block into one ticker-date panel with the forward target."""
    cfg = SETTINGS

    panel = (
        technical.join(fundamental, on=["date", "ticker"], how="left")
        .join(universe.select(["ticker", "name", "sector", "date_added"]), on="ticker", how="left")
    )

    panel = _add_target(panel, prices)

    # ---- eligibility -------------------------------------------------------
    before = panel.height
    panel = panel.filter(
        (pl.col("date_added").is_null() | (pl.col("date") >= pl.col("date_added")))
        & (pl.col("close") >= cfg.universe.min_price)
        & (pl.col("dollar_volume_21d") >= cfg.universe.min_dollar_volume)
    )
    log.info(
        "panel: %d -> %d rows after eligibility (index membership, price, liquidity)",
        before, panel.height,
    )

    # drop dates too thin to rank across
    counts = panel.group_by("date").agg(pl.len().alias("n"))
    keep_dates = counts.filter(pl.col("n") >= cfg.model.min_names_per_date)["date"]
    panel = panel.filter(pl.col("date").is_in(keep_dates))

    return panel.sort(["date", "ticker"])


def _add_target(panel: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Forward `horizon`-day return, in excess of the benchmark."""
    h = SETTINGS.model.horizon_days
    bench = benchmark_ticker()

    market = (
        prices.filter(pl.col("ticker") == bench)
        .select(["date", "close"])
        .sort("date")
        .with_columns(
            (pl.col("close").shift(-h) / pl.col("close") - 1.0).alias("mkt_fwd_ret")
        )
        .select(["date", "mkt_fwd_ret"])
    )

    out = (
        panel.sort(["ticker", "date"])
        .with_columns(
            (pl.col("close").shift(-h).over("ticker") / pl.col("close") - 1.0).alias("fwd_ret")
        )
        .join(market, on="date", how="left")
        .with_columns((pl.col("fwd_ret") - pl.col("mkt_fwd_ret")).alias("target"))
    )
    return out


def cross_sectional_normalize(
    panel: pl.DataFrame,
    *,
    features: tuple[str, ...] = NAMES,
    sector_neutral: bool = True,
) -> pl.DataFrame:
    """Winsorise, z-score and median-fill each feature within each date.

    All three are same-date operations, so none of them can leak. Sector
    neutrality is applied by demeaning within (date, sector) before the z-score,
    which stops the score from collapsing into a bet on one sector.

    Two columns are emitted per feature:
      ``z_<feature>``      the direction-adjusted z-score, missing filled with 0
      ``has_<feature>``    1.0 when the raw value was present, else 0.0

    The `has_` flags matter: filling a gap with 0 makes a name look merely
    average on that feature, which is a real assumption and needs to stay
    visible downstream rather than silently counting as evidence.
    """
    cfg = SETTINGS.model
    lo, hi = cfg.winsorize_pct, 1.0 - cfg.winsorize_pct
    present = [f for f in features if f in panel.columns]
    directions = {f.name: f.direction for f in FEATURES}

    out = panel.with_columns(
        [pl.col(f).is_not_null().cast(pl.Float64).alias(f"has_{f}") for f in present]
    )

    # 1) winsorise within date, so one blown-up value cannot dominate the z-score
    out = out.with_columns(
        [
            pl.col(f)
            .clip(
                pl.col(f).quantile(lo).over("date"),
                pl.col(f).quantile(hi).over("date"),
            )
            .alias(f)
            for f in present
        ]
    )

    # 2) sector-demean within date
    if sector_neutral:
        out = out.with_columns(
            [
                (pl.col(f) - pl.col(f).median().over(["date", "sector"])).alias(f)
                for f in present
            ]
        )

    # 3) z-score within date, then direction-adjust and fill gaps with the
    #    cross-sectional average (zero, post-standardisation)
    out = out.with_columns(
        [
            (
                (pl.col(f) - pl.col(f).mean().over("date"))
                / (pl.col(f).std().over("date") + 1e-9)
                * directions.get(f, 1)
            )
            .clip(-4.0, 4.0)          # keep a single outlier from swamping a group
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias(f"z_{f}")
            for f in present
        ]
    )

    return out


def feature_columns(panel: pl.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith("z_")]


def coverage_columns(panel: pl.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith("has_")]
