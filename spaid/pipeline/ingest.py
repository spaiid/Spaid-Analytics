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

    # The current list must never overwrite a historical reconstruction. This
    # step rebuilds membership from today's published constituents, which is the
    # right thing on a fresh store and a silent catastrophe on a populated one:
    # it would replace 300 observed index exits with 500 companies that have
    # never left, and every subsequent backtest would quietly go back to ranking
    # today's survivors.
    wrote_membership = False
    existing = store.read("universe_membership")
    reconstructed = (
        existing is not None
        and not existing.is_empty()
        and existing.filter(pl.col("universe_version") != "current-list").height > 0
    )
    if reconstructed:
        log.info(
            "universe: keeping the reconstructed membership history; the current list is "
            "already contained in it"
        )
    else:
        store.write(
            membership,
            "universe_membership",
            source="wikipedia",
            note=(
                "Current membership with entry dates only. Historical deletions are not in this "
                "source; run `spaid universe` to reconstruct them from its revision history."
            ),
        )
        wrote_membership = True

    return {
        "companies": companies.height,
        "securities": securities.height,
        "multi_class": securities.height - companies.height,
        "membership_written": wrote_membership,
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


def _former_members_with_prices() -> pl.DataFrame:
    """Securities that have left the index but whose price history we hold.

    Returns an empty frame when the historical universe has not been built, so
    that a store containing only the current constituents behaves exactly as it
    did before this step existed.
    """
    master = store.read("security_master")
    if master is None or master.is_empty():
        return pl.DataFrame(schema={"cik": pl.Int64, "company_id": pl.Utf8, "ticker": pl.Utf8})
    return (
        master.filter(
            (pl.col("status") != "listed")
            & (pl.col("price_coverage") != "none")
            & pl.col("cik").is_not_null()
            & pl.col("company_id").is_not_null()
        )
        .select(["cik", "company_id", pl.col("current_ticker").alias("ticker")])
        .unique(subset=["cik"], keep="first")
    )


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

    # Companies the index has dropped but whose shares we can still price file
    # with the SEC exactly like the survivors do, and a backtest that ranks the
    # historical universe needs their financials or they sit in the universe
    # unscored -- which removes them from the ranking just as effectively as
    # leaving them out altogether.
    extra = _former_members_with_prices()
    if not extra.is_empty():
        known = set(rows["cik"].to_list())
        extra = extra.filter(~pl.col("cik").is_in(list(known)))
        if not extra.is_empty():
            log.info(
                "observations: including %d former index members that remain priced",
                extra.height,
            )
            primary_ticker.update(
                dict(extra.select(["company_id", "ticker"]).iter_rows())
            )
            rows = pl.concat(
                [rows, extra.select(["cik", "company_id"])], how="vertical_relaxed"
            ).unique(subset=["cik"], keep="first").sort("cik")

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
    former = _former_members_with_prices()
    if not former.is_empty():
        for company_id, ticker in former.select(["company_id", "ticker"]).iter_rows():
            tickers.setdefault(company_id, ticker)

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


def refresh_universe_history(*, force: bool = False) -> dict:
    """Reconstruct index membership through time and rebuild the security master.

    Skipped when a recent reconstruction already exists, because it costs a
    Wikipedia revision walk and the index changes a few times a quarter.
    """
    from spaid.pipeline import security_master

    if not force and store.is_fresh("security_master", SETTINGS.freshness.universe_hours):
        log.info("universe history: cached reconstruction is fresh, skipping")
        return {"skipped": True}
    return security_master.build()


def refresh_delisted_prices(*, force: bool = False, limit: int | None = None) -> dict:
    """Fetch prices for companies the index dropped, where the provider still has them.

    Roughly a third of the companies removed from the index are still listed --
    demoted for size or liquidity rather than acquired -- and the provider
    serves their history normally. Those are recoverable and they matter: a
    company dropped for shrinking is exactly the kind of name a survivor-only
    universe silently omits.

    The other two thirds were acquired, merged or failed. The provider returns
    nothing for them, and no amount of retrying changes that; the gap is
    reported by `spaid.pipeline.security_master.coverage_report` instead.

    Two guards, because a delisted ticker is not an inert string:

    * **Reassignment.** Tickers get handed to unrelated companies. A series that
      begins after the company left the index is a different business wearing
      its clothes, and accepting it would splice a stranger's returns into the
      history. Rejected.
    * **Gaps.** A series with a long hole inside the membership window is two
      listings stitched together by the provider. Rejected.
    """
    from spaid.pipeline import security_master
    from spaid.providers.yahoo.prices import fetch_prices

    membership = store.read("universe_membership")
    prices = store.read("prices", required=True)
    if membership is None or membership.is_empty():
        return {"skipped": True, "reason": "no membership table"}

    known = set(prices["ticker"].unique().to_list())
    data_start, data_end = prices["date"].min(), prices["date"].max()

    candidates = (
        membership.filter(
            pl.col("exit_date").is_not_null() & ~pl.col("ticker").is_in(list(known))
        )
        .sort("exit_date")
        .unique(subset=["ticker"], keep="last")
    )
    if limit:
        candidates = candidates.head(limit)
    if candidates.is_empty():
        log.info("delisted prices: nothing to fetch")
        return {"candidates": 0, "accepted": 0}

    log.info("delisted prices: trying %d removed tickers", candidates.height)
    fetched = fetch_prices(candidates["ticker"].to_list(), start=data_start)

    accepted: list[str] = []
    rejected: dict[str, str] = {}
    windows = dict(
        candidates.select(["ticker", "entry_date"]).iter_rows()
    )
    exits = dict(candidates.select(["ticker", "exit_date"]).iter_rows())

    keep_rows = []
    for ticker, group in fetched.group_by("ticker"):
        name = ticker[0] if isinstance(ticker, tuple) else ticker
        if name not in windows:
            continue  # a benchmark or sector proxy the fetch adds by default
        series = group.sort("date")
        first, last = series["date"].min(), series["date"].max()
        entry = max(windows[name], data_start)
        exit_by = exits[name]

        if first > exit_by:
            rejected[name] = (
                f"series starts {first}, after the company left the index on {exit_by}; "
                "the ticker has been reassigned to a different company"
            )
            continue
        if last < entry:
            rejected[name] = f"series ends {last}, before the membership window opens {entry}"
            continue
        in_window = series.filter(
            (pl.col("date") >= entry) & (pl.col("date") <= min(exit_by, data_end))
        )
        if in_window.height < 20:
            rejected[name] = f"only {in_window.height} sessions inside the membership window"
            continue
        gaps = (
            in_window.select(
                (pl.col("date").diff().dt.total_days() > 45).sum().alias("n")
            )["n"][0]
            or 0
        )
        if gaps:
            rejected[name] = (
                f"{gaps} gap(s) longer than 45 days inside the membership window; the series "
                "is probably two different listings joined together"
            )
            continue
        accepted.append(name)
        keep_rows.append(series)

    for name, reason in sorted(rejected.items()):
        log.info("delisted prices: rejected %s -- %s", name, reason)

    if not keep_rows:
        log.warning("delisted prices: no candidate passed the checks")
        return {
            "candidates": candidates.height,
            "accepted": 0,
            "rejected": len(rejected),
            "no_data": candidates.height - len(rejected),
        }

    addition = pl.concat(keep_rows, how="vertical_relaxed")
    merged = (
        pl.concat([prices, addition], how="diagonal_relaxed")
        .unique(subset=["ticker", "date"], keep="last")
        .sort(["ticker", "date"])
    )
    store.write(
        merged,
        "prices",
        source="yahoo",
        note=(
            "Includes companies removed from the index that remain listed. Companies that "
            "were acquired, merged or failed are still missing and cannot be recovered from "
            "this provider."
        ),
    )
    no_data = candidates.height - len(accepted) - len(rejected)
    log.info(
        "delisted prices: accepted %d of %d candidates (%d rejected, %d returned nothing)",
        len(accepted), candidates.height, len(rejected), no_data,
    )
    security_master.build()
    return {
        "candidates": candidates.height,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "no_data": no_data,
        "rows_added": addition.height,
        "tickers": sorted(accepted)[:20],
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
    # Reconstruct membership history before the filings step, so that former
    # index members are included in the companies whose filings are fetched.
    out["universe_history"] = refresh_universe_history(force=force)
    out["delisted_prices"] = refresh_delisted_prices(force=force)
    out["observations"] = refresh_observations(force=force, limit=limit)
    out["fundamentals"] = refresh_fundamentals(force=force)
    out["snapshot"] = refresh_snapshot(force=force, limit=limit)
    out["estimates"] = refresh_estimates(force=force, limit=limit)

    out["elapsed_seconds"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    return out
