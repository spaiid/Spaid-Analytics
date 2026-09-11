"""Forward-looking metrics from analyst estimates and earnings events.

These are the only genuinely forward-looking inputs the app has. Everything else
describes what a company has already done; revisions describe what the people
who cover it now think it will do, and a change in that view has historically
carried more information than its level.

The honesty constraint that shapes this module: the estimates source publishes
only a current snapshot, with no vintage history. So

* revision *counts* (how many analysts moved up or down in the last month) are
  usable immediately, because they are a statement about the recent past that
  the provider computes for us;
* revision *deltas* measured by comparing our own stored snapshots only become
  available once we have been collecting for long enough, and return null until
  then rather than zero;
* nothing here may be used by the backtester at historical dates.

`estimates_available_from` records when collection began, and the confidence
system reads it so that a name scored without any forward evidence is visibly
less certain rather than silently equivalent.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

log = logging.getLogger(__name__)

EPS = 1e-9


def revision_balance(
    estimates: pl.DataFrame, *, as_of: date | None = None, period: str = "+1y"
) -> pl.DataFrame:
    """Net direction of recent analyst earnings revisions, scaled to [-1, +1].

    Defined as (upgrades - downgrades) / (upgrades + downgrades) over the past
    thirty days. A company nobody revised returns null, not zero: "no analyst
    changed their mind" and "analysts were evenly split" are different states,
    and only the second is evidence.
    """
    if estimates.is_empty():
        return pl.DataFrame(schema={"company_id": pl.Utf8, "eps_revision_3m": pl.Float64})

    as_of = as_of or estimates["as_of"].max()
    sub = estimates.filter(
        (pl.col("as_of") == as_of)
        & (pl.col("metric") == "eps")
        & (pl.col("period") == period)
    )
    if sub.is_empty():
        return pl.DataFrame(schema={"company_id": pl.Utf8, "eps_revision_3m": pl.Float64})

    up = pl.col("up_30d").fill_null(0)
    down = pl.col("down_30d").fill_null(0)
    total = up + down
    return sub.select(
        "company_id",
        pl.when(total > 0)
        .then((up - down) / total)
        .otherwise(None)
        .cast(pl.Float64)
        .alias("eps_revision_3m"),
    )


def consensus_drift(
    estimates: pl.DataFrame,
    *,
    metric: str = "revenue",
    period: str = "+1y",
    lookback_days: int = 90,
    alias: str = "revenue_revision_3m",
) -> pl.DataFrame:
    """Change in the consensus level between our own stored snapshots.

    Returns null for every company until at least two snapshots separated by
    enough time exist. That is the correct behaviour on a cold start: the metric
    is genuinely unknown, and the coverage machinery will treat it as missing
    rather than as neutral evidence.
    """
    empty = pl.DataFrame(schema={"company_id": pl.Utf8, alias: pl.Float64})
    if estimates.is_empty():
        return empty

    sub = estimates.filter(
        (pl.col("metric") == metric) & (pl.col("period") == period)
    ).select(["company_id", "as_of", "consensus"])
    if sub.is_empty():
        return empty

    latest_date = sub["as_of"].max()
    cutoff = latest_date - timedelta(days=lookback_days)
    prior = (
        sub.filter(pl.col("as_of") <= cutoff)
        .sort("as_of")
        .group_by("company_id")
        .agg(pl.col("consensus").last().alias("_prior"))
    )
    if prior.is_empty():
        log.info(
            "%s: no snapshot older than %d days yet, metric reported as unavailable",
            alias, lookback_days,
        )
        return empty

    current = sub.filter(pl.col("as_of") == latest_date).select(
        ["company_id", pl.col("consensus").alias("_now")]
    )
    return (
        current.join(prior, on="company_id", how="inner")
        .select(
            "company_id",
            pl.when(pl.col("_prior").abs() > EPS)
            .then((pl.col("_now") - pl.col("_prior")) / pl.col("_prior").abs())
            .otherwise(None)
            .alias(alias),
        )
    )


def forward_growth(estimates: pl.DataFrame, *, min_analysts: int = 3) -> pl.DataFrame:
    """Consensus next-year earnings growth, where coverage is thick enough.

    A consensus of two analysts is one analyst's opinion and a rounding error,
    so thin coverage produces null rather than a number the confidence system
    would have to discount afterwards.
    """
    empty = pl.DataFrame(
        schema={
            "company_id": pl.Utf8,
            "forward_eps_growth": pl.Float64,
            "forward_eps": pl.Float64,
            "n_analysts": pl.Int32,
        }
    )
    if estimates.is_empty():
        return empty

    as_of = estimates["as_of"].max()
    sub = estimates.filter((pl.col("as_of") == as_of) & (pl.col("metric") == "eps"))
    if sub.is_empty():
        return empty

    next_year = sub.filter(pl.col("period") == "+1y").select(
        ["company_id", "consensus", "growth", "n_analysts"]
    ).rename({"consensus": "forward_eps", "growth": "_growth"})
    this_year = sub.filter(pl.col("period") == "0y").select(
        ["company_id", pl.col("consensus").alias("_this_year")]
    )

    return (
        next_year.join(this_year, on="company_id", how="left")
        .with_columns(
            pl.coalesce(
                pl.col("_growth"),
                pl.when(pl.col("_this_year").abs() > EPS)
                .then(pl.col("forward_eps") / pl.col("_this_year") - 1.0)
                .otherwise(None),
            ).alias("forward_eps_growth")
        )
        .with_columns(
            pl.when(pl.col("n_analysts").fill_null(0) >= min_analysts)
            .then(pl.col("forward_eps_growth"))
            .otherwise(None)
            .alias("forward_eps_growth"),
            pl.when(pl.col("n_analysts").fill_null(0) >= min_analysts)
            .then(pl.col("forward_eps"))
            .otherwise(None)
            .alias("forward_eps"),
        )
        .select(["company_id", "forward_eps_growth", "forward_eps", "n_analysts"])
    )


def earnings_surprise(
    events: pl.DataFrame, *, as_of: date | None = None, quarters: int = 4
) -> pl.DataFrame:
    """Average percentage earnings surprise over the last few reported quarters."""
    empty = pl.DataFrame(
        schema={
            "company_id": pl.Utf8,
            "earnings_surprise": pl.Float64,
            "surprise_hit_rate": pl.Float64,
        }
    )
    if events.is_empty():
        return empty

    as_of = as_of or date.today()
    reported = events.filter(
        (~pl.col("is_future")) & pl.col("surprise").is_not_null() & (pl.col("event_date") <= as_of)
    )
    if reported.is_empty():
        return empty

    return (
        reported.sort(["company_id", "event_date"])
        .group_by("company_id")
        .agg(
            pl.col("surprise").tail(quarters).mean().alias("earnings_surprise"),
            (pl.col("surprise").tail(quarters) > 0).mean().cast(pl.Float64).alias(
                "surprise_hit_rate"
            ),
        )
    )


def next_earnings(events: pl.DataFrame, *, as_of: date | None = None) -> pl.DataFrame:
    """Days until the next scheduled earnings release, for event risk and catalysts."""
    empty = pl.DataFrame(
        schema={"company_id": pl.Utf8, "next_earnings_date": pl.Date, "days_to_earnings": pl.Int32}
    )
    if events.is_empty():
        return empty

    as_of = as_of or date.today()
    future = events.filter(pl.col("event_date") > as_of)
    if future.is_empty():
        return empty

    return (
        future.sort(["company_id", "event_date"])
        .group_by("company_id")
        .agg(pl.col("event_date").first().alias("next_earnings_date"))
        .with_columns(
            (pl.col("next_earnings_date") - pl.lit(as_of)).dt.total_days().cast(pl.Int32)
            .alias("days_to_earnings")
        )
    )


def post_earnings_drift(
    events: pl.DataFrame,
    prices: pl.DataFrame,
    securities: pl.DataFrame,
    *,
    as_of: date | None = None,
    window: int = 3,
    benchmark: str = "SPY",
) -> pl.DataFrame:
    """Market-relative return over the sessions following the last earnings report.

    How the market actually received the most recent news, which is a more
    direct reading than the surprise percentage: a company can beat consensus
    and still fall on guidance.
    """
    empty = pl.DataFrame(
        schema={
            "company_id": pl.Utf8,
            "post_earnings_drift": pl.Float64,
            "last_earnings_date": pl.Date,
        }
    )
    if events.is_empty() or prices.is_empty():
        return empty

    as_of = as_of or prices["date"].max()
    last_event = (
        events.filter((~pl.col("is_future")) & (pl.col("event_date") <= as_of))
        .sort(["company_id", "event_date"])
        .group_by("company_id")
        .agg(pl.col("event_date").last().alias("last_earnings_date"))
    )
    if last_event.is_empty():
        return empty

    primary = securities.filter(pl.col("is_primary")).select(["company_id", "ticker"])
    market = (
        prices.filter(pl.col("ticker") == benchmark)
        .select(["date", pl.col("close_adj").alias("mkt_close")])
        .sort("date")
    )

    px = (
        prices.join(primary, on="ticker", how="inner")
        .select(["company_id", "date", "close_adj"])
        .sort(["company_id", "date"])
        .join(market, on="date", how="left")
        .with_columns(
            pl.int_range(pl.len()).over("company_id").alias("_i"),
        )
    )

    anchored = px.join(last_event, on="company_id", how="inner")
    at_event = (
        anchored.filter(pl.col("date") <= pl.col("last_earnings_date"))
        .group_by("company_id")
        .agg(
            pl.col("_i").last().alias("_event_i"),
            pl.col("close_adj").last().alias("_px0"),
            pl.col("mkt_close").last().alias("_mkt0"),
        )
    )
    after = (
        anchored.join(at_event, on="company_id", how="inner")
        .filter(
            (pl.col("_i") > pl.col("_event_i"))
            & (pl.col("_i") <= pl.col("_event_i") + window)
        )
        .group_by("company_id")
        .agg(
            pl.col("close_adj").last().alias("_px1"),
            pl.col("mkt_close").last().alias("_mkt1"),
            pl.col("_px0").first(),
            pl.col("_mkt0").first(),
            pl.col("last_earnings_date").first(),
        )
    )
    if after.is_empty():
        return empty

    return after.select(
        "company_id",
        (
            (pl.col("_px1") / pl.col("_px0") - 1.0)
            - (pl.col("_mkt1") / pl.col("_mkt0") - 1.0)
        ).alias("post_earnings_drift"),
        pl.col("last_earnings_date"),
    )


def build_estimate_metrics(
    estimates: pl.DataFrame,
    events: pl.DataFrame,
    prices: pl.DataFrame,
    securities: pl.DataFrame,
    *,
    as_of: date | None = None,
) -> pl.DataFrame:
    """All estimate-derived metrics, one row per company."""
    companies = securities.filter(pl.col("is_primary")).select(["company_id"]).unique()
    out = companies

    for piece in (
        revision_balance(estimates, as_of=as_of),
        consensus_drift(estimates),
        forward_growth(estimates),
        earnings_surprise(events, as_of=as_of),
        next_earnings(events, as_of=as_of),
        post_earnings_drift(events, prices, securities, as_of=as_of),
    ):
        if piece.is_empty():
            continue
        out = out.join(piece, on="company_id", how="left")

    expected = [
        "eps_revision_3m", "revenue_revision_3m", "forward_eps_growth", "forward_eps",
        "n_analysts", "earnings_surprise", "surprise_hit_rate", "next_earnings_date",
        "days_to_earnings", "post_earnings_drift", "last_earnings_date",
    ]
    for col in expected:
        if col not in out.columns:
            dtype = pl.Date if col.endswith("_date") else (
                pl.Int32 if col in ("n_analysts", "days_to_earnings") else pl.Float64
            )
            out = out.with_columns(pl.lit(None, dtype=dtype).alias(col))

    log.info(
        "estimate metrics: %d companies; revisions %d, forward growth %d, surprise %d",
        out.height,
        int(out["eps_revision_3m"].is_not_null().sum()),
        int(out["forward_eps_growth"].is_not_null().sum()),
        int(out["earnings_surprise"].is_not_null().sum()),
    )
    return out
