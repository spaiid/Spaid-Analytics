"""The point-in-time panel: what Spaid Analytics could have computed, and when.

Everything in this module exists to answer one question honestly — on the last
session of some month in 2018, what would this application have ranked? — and
the answer has to be built from the evidence that existed on that morning and
nothing else.

Three specific pieces of foresight are removed here, each of which the live
pipeline legitimately uses and none of which a backtest may:

* **The current provider snapshot.** `security_snapshot` holds today's share
  count and today's forward consensus, joined on company identity with no date
  at all. On a historical row it is pure hindsight, so it is not read. The cost
  is real and is measured rather than waved away: dual-class filers whose share
  counts the SEC's aggregate interface does not carry lose their market
  capitalisation, and with it their valuation metrics.
* **Analyst estimates.** Seven metrics depend on them and there is no vintage
  history, so their columns are present and entirely null. That matters: a
  metric whose *column* is missing would quietly shrink the denominator and the
  category would report full coverage over the metrics that happen to remain. A
  null column is scored as `missing`, which is what it is.
* **Today's universe.** The grid is the historical membership reconstruction,
  so a company is ranked on a date only if it was in the index on that date and
  we hold prices covering it.

What is left is the same metric and scoring code the live pipeline runs. That
is deliberate: a backtest that reimplements the scoring is testing the
reimplementation.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from spaid.backtest.strategy import StrategyVersion
from spaid.config.settings import SETTINGS
from spaid.pipeline import scoring
from spaid.pipeline.metrics import financial, price
from spaid.storage import store

log = logging.getLogger(__name__)

# Trading days per calendar month, used to turn a forward-return horizon stated
# in months into a number of sessions.
SESSIONS_PER_MONTH = 21

# The longest backward lag any metric takes. The panel is extended this far
# before the first decision date so that the five-year growth metrics have
# something to compare against.
WARMUP_YEARS = 5


def trading_sessions(prices: pl.DataFrame, benchmark: str | None = None) -> list[date]:
    """The benchmark's own sessions -- the calendar everything else aligns to."""
    benchmark = benchmark or SETTINGS.universe.benchmark
    sessions = (
        prices.filter(pl.col("ticker") == benchmark)["date"].unique().sort().to_list()
    )
    if not sessions:
        sessions = prices["date"].unique().sort().to_list()
    return sessions


def month_end_sessions(sessions: list[date], start: date, end: date) -> list[date]:
    """The last trading session of each month in the range.

    Walking backwards keeps the *last* session of the month, which is the one a
    month-end decision would actually be made on. Keeping the first would
    quietly move every decision forward by a month.
    """
    out: list[date] = []
    seen: set[tuple[int, int]] = set()
    for session in reversed(sessions):
        if session > end:
            continue
        if session < start:
            break
        key = (session.year, session.month)
        if key not in seen:
            seen.add(key)
            out.append(session)
    return sorted(out)


def historical_securities(*, require_prices: bool = True) -> pl.DataFrame:
    """Every security the historical universe contains, with its classification.

    Read from the security master rather than the `securities` table, because
    that table holds today's constituents and the whole point of this panel is
    the companies that are no longer in it.
    """
    master = store.read("security_master", required=True)
    out = master.filter(pl.col("company_id").is_not_null())
    if require_prices:
        out = out.filter(pl.col("price_coverage") != "none")
    return out.select(
        pl.col("security_id"),
        pl.col("company_id"),
        pl.col("current_ticker").alias("ticker"),
        pl.col("name"),
        pl.col("sector"),
        pl.col("industry"),
        pl.col("business_model"),
        pl.lit(True).alias("is_primary"),
    )


def membership_windows(index_name: str | None = None) -> pl.DataFrame:
    """Entry and exit per membership spell."""
    index_name = index_name or SETTINGS.universe.name
    membership = store.read("universe_membership", required=True)
    return (
        membership.filter(pl.col("index_name") == index_name)
        .select(
            ["security_id", "entry_date", "exit_date", "entry_precision", "exit_precision"]
        )
    )


def membership_on_dates(
    decision_dates: list[date], *, index_name: str | None = None
) -> pl.DataFrame:
    """One row per security per date saying whether it was a member then.

    Flattened deliberately. A company can leave the index and rejoin -- EQT and
    several others did -- so it has more than one spell, and joining spells
    straight onto the panel would give that company two rows on every date and
    let it occupy two portfolio slots. Collapsing to a single membership flag
    per date is what keeps one company one row.
    """
    spells = membership_windows(index_name)
    grid = pl.DataFrame({"date": decision_dates}, schema={"date": pl.Date})
    return (
        spells.join(grid, how="cross")
        .filter(
            (pl.col("entry_date") <= pl.col("date"))
            & (pl.col("exit_date").is_null() | (pl.col("exit_date") > pl.col("date")))
        )
        .group_by(["security_id", "date"])
        .agg(
            pl.col("entry_date").max(),
            pl.col("exit_date").max(),
            pl.col("entry_precision").last(),
            pl.col("exit_precision").last(),
        )
        .with_columns(pl.lit(True).alias("is_member"))
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def build_metric_panel(
    strategy: StrategyVersion,
    *,
    decision_dates: list[date],
    warmup_dates: list[date] | None = None,
    prices: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Every metric, for every historical member, on every decision date.

    `warmup_dates` are computed but not returned. The multi-year growth metrics
    are built by as-of joining the panel to itself at a one, three and five year
    offset, so a panel that begins on the first decision date has nothing to
    look back at and every growth metric is null for years. The warm-up rows
    give the lags something to find; they are dropped before scoring so that no
    decision date outside the test window can influence a percentile rank.
    """
    prices = prices if prices is not None else store.read("prices", required=True)
    fundamentals = store.read("fundamentals", required=True)
    securities = historical_securities()

    grid_dates = sorted(set(decision_dates) | set(warmup_dates or []))
    log.info(
        "point-in-time panel: %d securities, %d decision dates (%s to %s), "
        "%d warm-up dates back to %s",
        securities.height, len(decision_dates), decision_dates[0], decision_dates[-1],
        len(grid_dates) - len(decision_dates), grid_dates[0],
    )

    sectors = securities.select(["ticker", "sector"])
    price_metrics = price.build_price_metrics(prices, sectors=sectors)
    price_metrics = price_metrics.filter(pl.col("date").is_in(decision_dates))

    grid = financial.build_grid(prices, securities, dates=grid_dates)
    panel = financial.build_financial_metrics(grid, fundamentals, prices)
    panel = panel.filter(pl.col("date").is_in(decision_dates))

    panel = panel.join(
        price_metrics.drop(["close_raw", "close_adj", "split_factor"]),
        on=["ticker", "date"],
        how="left",
    ).join(
        securities.select(
            ["company_id", "security_id", "name", "sector", "industry", "business_model"]
        ),
        on="company_id",
        how="left",
    )

    # Same state-based override the live pipeline applies: two years of losses
    # makes a company unprofitable growth whatever its sector label says.
    panel = panel.with_columns(
        pl.when(
            (pl.col("net_income") < 0)
            & (pl.col("net_income__lag1y") < 0)
            & (pl.col("business_model") == "operating")
        )
        .then(pl.lit("unprofitable_growth"))
        .otherwise(pl.col("business_model"))
        .alias("business_model")
    )

    # The estimate-derived metrics, present and null. See the module docstring:
    # an absent column would be silently excluded from its category's coverage
    # denominator, which would report the gap as full coverage.
    panel = panel.with_columns(
        [
            pl.lit(None, dtype=pl.Float64).alias(key)
            for key in strategy.unavailable_point_in_time
        ]
    )

    log.info("point-in-time panel: %d rows x %d columns", panel.height, panel.width)
    return panel


def add_eligibility(
    panel: pl.DataFrame, strategy: StrategyVersion, *, index_name: str | None = None
) -> pl.DataFrame:
    """Flag which rows could actually have been bought, and say why not.

    The reason is kept rather than being collapsed into a boolean, because "not
    in the index yet" and "too illiquid" and "we have no prices for it" are
    three different statements about a backtest's coverage and the first two are
    ordinary while the third is a defect.
    """
    decision_dates = panel["date"].unique().sort().to_list()
    members = membership_on_dates(decision_dates, index_name=index_name)
    out = panel.join(members, on=["security_id", "date"], how="left").with_columns(
        pl.col("is_member").fill_null(False)
    )

    e = strategy.eligibility

    # The price screen is evaluated on the price the stock genuinely traded at,
    # recovered by multiplying the stored close by the splits that happened
    # after it. Screening on the stored close directly is look-ahead of the
    # worst kind, because it is invisible: Nvidia's 2015 close reads $0.56 on
    # today's basis, so a $5 floor would have excluded the decade's best
    # performer from the universe until 2019 -- and the exclusion would have
    # been *caused* by its later splits.
    out = out.with_columns(
        (pl.col("close_raw") * pl.col("split_factor").fill_null(1.0)).alias("price_as_traded")
    ).with_columns(
        (pl.col("price_as_traded") >= e.min_price).alias("_price_ok"),
        (
            pl.col("dollar_volume_median").is_null()
            | (pl.col("dollar_volume_median") >= e.min_dollar_volume)
        ).alias("_liquidity_ok"),
        (pl.col("mom_12_1").is_not_null()).alias("_history_ok"),
    )

    out = out.with_columns(
        pl.when(~pl.col("is_member"))
        .then(pl.lit("not_in_index"))
        .when(~pl.col("_price_ok"))
        .then(pl.lit("below_min_price"))
        .when(~pl.col("_liquidity_ok"))
        .then(pl.lit("below_min_liquidity"))
        .when(~pl.col("_history_ok"))
        .then(pl.lit("insufficient_price_history"))
        .otherwise(pl.lit("eligible"))
        .alias("eligibility")
    ).with_columns((pl.col("eligibility") == "eligible").alias("is_eligible"))

    counts = dict(
        out.group_by("eligibility").agg(pl.len().alias("n")).sort("n", descending=True).iter_rows()
    )
    log.info("eligibility across all decision dates: %s", counts)
    return out.drop("_price_ok", "_liquidity_ok", "_history_ok")


def score_panel(panel: pl.DataFrame, strategy: StrategyVersion) -> pl.DataFrame:
    """Rank the eligible universe on each date with the frozen scoring spec.

    Only eligible rows are scored, and that is a modelling decision rather than
    an optimisation: percentile ranks are relative, so including companies that
    could not have been bought would change the score of every company that
    could.
    """
    spec = strategy.scoring
    eligible = panel.filter(pl.col("is_eligible"))
    if eligible.is_empty():
        raise RuntimeError("no eligible rows to score; check the universe and price coverage")

    metric_scores = scoring.build_metric_scores(eligible, spec)
    category_scores = scoring.build_category_scores(metric_scores, spec)
    opportunity = scoring.build_opportunity_scores(category_scores, spec)

    wide = opportunity.select(
        ["company_id", "date", "score", "rank", "percentile", "band", "coverage",
         "quality", "growth", "momentum", "valuation", "universe_size"]
    )
    return panel.join(wide, on=["company_id", "date"], how="left"), metric_scores, category_scores


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------


def forward_returns(
    prices: pl.DataFrame,
    decision_dates: list[date],
    horizons_months: tuple[int, ...],
    *,
    benchmark: str | None = None,
) -> pl.DataFrame:
    """Total return over each horizon from each decision date, and its excess.

    Computed on `close_adj`, which is the total-return series, so dividends are
    in it. A security that stops trading inside the horizon gets a null rather
    than a return measured to its last price: the panel's job is to report what
    is known, and the engine handles the liquidation separately.
    """
    benchmark = benchmark or SETTINGS.universe.benchmark
    sessions = trading_sessions(prices, benchmark)
    index_of = {d: i for i, d in enumerate(sessions)}

    market = (
        prices.filter(pl.col("ticker") == benchmark)
        .select(["date", pl.col("close_adj").alias("mkt")])
        .sort("date")
    )

    px = (
        prices.filter(pl.col("date").is_in(sessions))
        .select(["ticker", "date", "close_adj"])
        .sort(["ticker", "date"])
    )

    out = px.filter(pl.col("date").is_in(decision_dates)).select(["ticker", "date"])
    for months in horizons_months:
        steps = months * SESSIONS_PER_MONTH
        target = {
            d: sessions[i + steps] for d, i in index_of.items() if i + steps < len(sessions)
        }
        mapping = pl.DataFrame(
            {"date": list(target), "target_date": list(target.values())},
            schema={"date": pl.Date, "target_date": pl.Date},
        )
        later = px.rename({"date": "target_date", "close_adj": "close_then"})
        mkt_later = market.rename({"date": "target_date", "mkt": "mkt_then"})

        step = (
            out.join(mapping, on="date", how="left")
            .join(px, on=["ticker", "date"], how="left")
            .join(later, on=["ticker", "target_date"], how="left")
            .join(market, on="date", how="left")
            .join(mkt_later, on="target_date", how="left")
            .with_columns(
                (pl.col("close_then") / pl.col("close_adj") - 1.0).alias(f"fwd_{months}m"),
                (pl.col("mkt_then") / pl.col("mkt") - 1.0).alias(f"_mkt_{months}m"),
            )
            .with_columns(
                (pl.col(f"fwd_{months}m") - pl.col(f"_mkt_{months}m")).alias(
                    f"fwd_excess_{months}m"
                )
            )
            .select(["ticker", "date", f"fwd_{months}m", f"fwd_excess_{months}m"])
        )
        out = out.join(step, on=["ticker", "date"], how="left")

    return out


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------


def build(
    strategy: StrategyVersion,
    *,
    start: date,
    end: date,
    horizons_months: tuple[int, ...] = (1, 3, 6, 12),
    index_name: str | None = None,
    extra_dates: list[date] | None = None,
) -> dict:
    """Assemble the scored, eligibility-flagged, forward-return-labelled panel.

    `extra_dates` are scored alongside the month-end grid but are not returned
    as decision dates. The rebalance-shift robustness test needs scores on days
    the primary schedule never visits, and the only honest way to provide them
    is to run the same scoring on those days -- not to reuse a nearby month-end
    score, which would be the very look-ahead the test is checking for.
    """
    prices = store.read("prices", required=True)
    sessions = trading_sessions(prices)
    decision_dates = month_end_sessions(sessions, start, end)
    scoring_dates = sorted(set(decision_dates) | set(extra_dates or []))
    if len(decision_dates) < 12:
        raise ValueError(
            f"only {len(decision_dates)} monthly decision dates between {start} and {end}; "
            "a backtest needs at least a year of them"
        )

    # Five years of monthly warm-up, because the longest lag any metric takes is
    # five years. Where the price history does not reach that far back the
    # affected metrics stay missing and the spec renormalises, which is why the
    # test period starts where it does.
    warmup_dates = month_end_sessions(
        sessions, date(start.year - WARMUP_YEARS, start.month, 1), start
    )
    panel = build_metric_panel(
        strategy,
        decision_dates=scoring_dates,
        warmup_dates=warmup_dates,
        prices=prices,
    )
    panel = add_eligibility(panel, strategy, index_name=index_name)
    scored, metric_scores, category_scores = score_panel(panel, strategy)

    labels = forward_returns(prices, scoring_dates, horizons_months)
    scored = scored.join(labels, on=["ticker", "date"], how="left")

    return {
        "panel": scored,
        "metric_scores": metric_scores,
        "category_scores": category_scores,
        "decision_dates": decision_dates,
        "scoring_dates": scoring_dates,
        "prices": prices,
    }
