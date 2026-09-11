"""Data ingestion: fetch, normalise and persist the canonical tables.

Each step is independently runnable and independently skippable, because they
have very different costs and cadences. Prices change daily and take a minute;
the SEC archive changes when companies file and takes several minutes; the
constituents list changes a few times a year.

Every step ends by writing through `store.write`, which validates the frame
against its declared schema. A step that produces something the schema does not
recognise fails here, next to the code that produced it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import polars as pl

from spaid.config.settings import SETTINGS
from spaid.pipeline.fundamentals import build_fundamentals
from spaid.providers.sec.client import CompanyFacts
from spaid.providers.sec.extract import TradingCalendar, build_observations
from spaid.providers.wikipedia.universe import build_identity, fetch_constituents
from spaid.providers.yahoo.prices import fetch_prices, median_dollar_volume
from spaid.storage import store

log = logging.getLogger(__name__)


def refresh_universe(*, force: bool = False) -> dict:
    """Constituents -> companies, securities, index membership.

    Runs twice on a cold start: once to learn the ticker list, and once more
    after prices exist so that the primary share class of a dual-class filer can
    be chosen by traded volume rather than alphabetically.
    """
    fresh = store.is_fresh("companies", SETTINGS.freshness.universe_hours)
    if fresh and not force:
        log.info("universe: cached copy is fresh, skipping")
        companies = store.read("companies")
        return {"skipped": True, "companies": companies.height if companies is not None else 0}

    constituents = fetch_constituents()
    prices = store.read("prices")
    liquidity = median_dollar_volume(prices) if prices is not None else None

    companies, securities, membership = build_identity(constituents, liquidity=liquidity)

    store.write(companies, "companies", source="wikipedia")
    store.write(securities, "securities", source="wikipedia")
    store.write(
        membership,
        "universe_membership",
        source="wikipedia",
        note=(
            "Current membership with entry dates. Historical deletions are not available from "
            "this source, so backtests before the present day understate the drag from names "
            "the index dropped."
        ),
    )
    return {
        "companies": companies.height,
        "securities": securities.height,
        "multi_class": securities.height - companies.height,
    }


def refresh_prices(*, force: bool = False) -> dict:
    fresh = store.is_fresh("prices", SETTINGS.freshness.prices_hours)
    if fresh and not force:
        log.info("prices: cached copy is fresh, skipping")
        prices = store.read("prices")
        return {"skipped": True, "rows": prices.height if prices is not None else 0}

    securities = store.read("securities", required=True)
    tickers = securities["ticker"].unique().sort().to_list()
    prices = fetch_prices(tickers)
    store.write(prices, "prices", source="yahoo")
    return {
        "rows": prices.height,
        "tickers": prices["ticker"].n_unique(),
        "last_date": str(prices["date"].max()),
    }


def refresh_observations(*, force: bool = False, cache_days: float = 7.0, limit: int | None = None) -> dict:
    """SEC company facts -> the raw observation table."""
    fresh = store.is_fresh("observations", SETTINGS.freshness.fundamentals_hours)
    if fresh and not force:
        log.info("observations: cached copy is fresh, skipping")
        obs = store.read("observations")
        return {"skipped": True, "rows": obs.height if obs is not None else 0}

    companies = store.read("companies", required=True)
    prices = store.read("prices", required=True)
    securities = store.read("securities", required=True)

    calendar = TradingCalendar.from_prices(prices, SETTINGS.universe.benchmark)
    primary_ticker = dict(
        securities.filter(pl.col("is_primary"))
        .select(["company_id", "ticker"])
        .iter_rows()
    )

    rows = companies.select(["cik", "company_id"]).sort("cik")
    if limit:
        rows = rows.head(limit)
    pairs = [(int(cik), primary_ticker.get(cid)) for cik, cid in rows.iter_rows()]

    with CompanyFacts(cache_days=cache_days) as facts:
        observations = build_observations(
            pairs, facts_loader=facts.get, calendar=calendar
        )
        stats = facts.stats

    store.write(observations, "observations", source="sec_companyfacts")
    return {
        "rows": observations.height,
        "companies": observations["company_id"].n_unique(),
        "tags": observations["tag"].n_unique(),
        "restatements": int(observations["is_restatement"].sum() or 0),
        "sec_requests": stats.get("requests"),
        "sec_mb": stats.get("mb"),
    }


def refresh_fundamentals(*, force: bool = False) -> dict:
    """Observations -> point-in-time trailing and instant financials."""
    obs_age = store.age_hours("observations")
    fund_age = store.age_hours("fundamentals")
    if (
        not force
        and fund_age is not None
        and obs_age is not None
        and fund_age < obs_age
    ):
        log.info("fundamentals: newer than observations, skipping")
        fundamentals = store.read("fundamentals")
        return {"skipped": True, "rows": fundamentals.height if fundamentals is not None else 0}

    observations = store.read("observations", required=True)
    securities = store.read("securities", required=True)
    tickers = dict(
        securities.filter(pl.col("is_primary")).select(["company_id", "ticker"]).iter_rows()
    )

    fundamentals = build_fundamentals(observations, tickers=tickers)
    store.write(fundamentals, "fundamentals", source="sec_companyfacts")

    coverage = (
        fundamentals.group_by("concept")
        .agg(pl.col("company_id").n_unique().alias("companies"))
        .sort("companies", descending=True)
    )
    return {
        "rows": fundamentals.height,
        "companies": fundamentals["company_id"].n_unique(),
        "concepts": fundamentals["concept"].n_unique(),
        "coverage": dict(coverage.iter_rows()),
    }


def refresh_snapshot(*, force: bool = False, limit: int | None = None) -> dict:
    """Current provider facts: share counts, forward consensus, quotes.

    Carries the share count for the handful of dual-class filers whose counts
    are reported per class and therefore absent from the SEC's aggregate data.
    Without this step those companies have no market capitalisation and so no
    valuation at all.
    """
    from spaid.providers.yahoo.profile import fetch_snapshot

    fresh = store.is_fresh("security_snapshot", SETTINGS.freshness.prices_hours)
    if fresh and not force:
        log.info("snapshot: cached copy is fresh, skipping")
        snap = store.read("security_snapshot")
        return {"skipped": True, "rows": snap.height if snap is not None else 0}

    securities = store.read("securities", required=True)
    snapshot = fetch_snapshot(securities, limit=limit)
    store.write(snapshot, "security_snapshot", source="yahoo")
    return {
        "rows": snapshot.height,
        "with_shares": int(snapshot["shares_outstanding"].is_not_null().sum()),
        "with_forward_eps": int(snapshot["forward_eps"].is_not_null().sum()),
    }


def refresh_estimates(*, force: bool = False, limit: int | None = None) -> dict:
    """Analyst consensus, revisions and earnings dates.

    Merged rather than replaced: this source has no history, so every collection
    is a new snapshot that must be kept alongside the old ones if the app is ever
    to measure revisions over time itself.
    """
    from spaid.providers.yahoo.estimates import fetch_estimates

    fresh = store.is_fresh("estimates", SETTINGS.freshness.estimates_hours)
    if fresh and not force:
        log.info("estimates: cached copy is fresh, skipping")
        est = store.read("estimates")
        return {"skipped": True, "rows": est.height if est is not None else 0}

    securities = store.read("securities", required=True)
    estimates, events = fetch_estimates(securities, limit=limit)

    if estimates.height:
        store.upsert(estimates, "estimates", source="yahoo")
    if events.height:
        store.upsert(events, "earnings_events", source="yahoo")

    return {
        "estimate_rows": estimates.height,
        "event_rows": events.height,
        "companies": estimates["company_id"].n_unique() if estimates.height else 0,
    }


def ingest_all(*, force: bool = False, limit: int | None = None) -> dict:
    """Run every ingestion step in dependency order."""
    started = datetime.now(UTC)
    out: dict[str, dict] = {}

    out["universe"] = refresh_universe(force=force)
    out["prices"] = refresh_prices(force=force)
    # Re-derive identity now that traded volume is available, so the primary
    # share class of a dual-class filer is the one that actually trades.
    out["universe_refined"] = refresh_universe(force=True)
    out["observations"] = refresh_observations(force=force, limit=limit)
    out["fundamentals"] = refresh_fundamentals(force=force)
    out["snapshot"] = refresh_snapshot(force=force, limit=limit)
    out["estimates"] = refresh_estimates(force=force, limit=limit)

    out["elapsed_seconds"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    return out
