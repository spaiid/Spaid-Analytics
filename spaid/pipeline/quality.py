"""Data-quality checks that run after every pipeline and feed the health view.

The product brief puts data quality above visual polish, and this module is how
that shows up in the interface. Each check answers a question a user would
reasonably ask before trusting a recommendation, and each returns a warning the
dashboard displays verbatim rather than a boolean the interface has to
interpret.

The checks that matter most are the ones for problems that look like *data*
rather than errors: a company whose market capitalisation is wrong by a factor
of forty still renders as a number, and only a cross-check against a second
source reveals it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import polars as pl

from spaid.storage import store

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Warning_:
    severity: str  # info | warning | critical
    message: str
    table: str | None = None

    def as_dict(self) -> dict:
        return {"severity": self.severity, "message": self.message, "table": self.table}


def check_all() -> list[dict]:
    """Every check, as a flat list of warnings ready for the interface."""
    warnings: list[Warning_] = []
    for check in (
        _check_missing_tables,
        _check_thin_filing_history,
        _check_market_cap_against_second_source,
        _check_share_count_plausibility,
        _check_universe_membership_gap,
        _check_score_coverage,
        _check_estimate_history,
        _check_stale_filings,
    ):
        try:
            warnings.extend(check())
        except Exception as exc:  # noqa: BLE001 - a broken check must not break the page
            log.warning("data-quality check %s failed: %s", check.__name__, exc)
    order = {"critical": 0, "warning": 1, "info": 2}
    warnings.sort(key=lambda w: order.get(w.severity, 3))
    return [w.as_dict() for w in warnings]


def _check_missing_tables() -> list[Warning_]:
    required = ("companies", "securities", "prices", "observations", "fundamentals")
    out = []
    for name in required:
        if not store.exists(name):
            out.append(
                Warning_(
                    "critical",
                    f"The {name} table has not been built yet, so nothing downstream can run.",
                    name,
                )
            )
    return out


def _check_thin_filing_history() -> list[Warning_]:
    """Companies whose current registrant has very little history.

    Usually a spin-off or a reorganisation. A reorganisation can be repaired by
    linking the predecessor registrant; a spin-off genuinely has no history, and
    saying so is more useful than borrowing the parent's.
    """
    obs = store.scan("observations")
    if obs is None:
        return []
    per = (
        obs.group_by("company_id")
        .agg(
            pl.len().alias("n"),
            pl.col("ticker").first().alias("ticker"),
            pl.col("filed").min().alias("first_filed"),
        )
        .collect()
    )
    cutoff = datetime.now(UTC).date() - timedelta(days=730)
    thin = per.filter((pl.col("n") < 1500) | (pl.col("first_filed") > cutoff)).sort("n")
    if thin.is_empty():
        return []

    tickers = [t for t in thin["ticker"].to_list() if t]
    return [
        Warning_(
            "warning",
            (
                f"{thin.height} companies have less than two years of filing history under their "
                f"current SEC registrant ({', '.join(tickers[:10])}"
                f"{', and others' if len(tickers) > 10 else ''}). These are recent spin-offs or "
                "reorganisations. Multi-year growth and valuation-history metrics cannot be "
                "computed for them, and their confidence scores reflect that."
            ),
            "observations",
        )
    ]


def _check_market_cap_against_second_source() -> list[Warning_]:
    """Compare the filing-derived market capitalisation with the provider's.

    This is the check that would have caught Berkshire reading $477 million
    because a 2011 Class A share count had been frozen in place for fifteen
    years. A single source cannot detect its own errors.
    """
    snapshot = store.read("security_snapshot")
    scores = store.read("opportunity_scores")
    if snapshot is None or scores is None or snapshot.is_empty():
        return []

    metrics = store.read("metric_scores")
    if metrics is None:
        return []
    # Market capitalisation is not itself a scored metric, so read it from the
    # valuations table, which records the price and share basis actually used.
    valuations = store.read("valuations")
    if valuations is None or valuations.is_empty():
        return []

    merged = (
        valuations.select(["company_id", "ticker", "price"])
        .join(
            snapshot.select(
                ["company_id", pl.col("market_cap").alias("provider_market_cap"),
                 pl.col("price").alias("provider_price")]
            ),
            on="company_id",
            how="inner",
        )
        .filter(
            pl.col("provider_price").is_not_null() & (pl.col("provider_price") > 0)
        )
        .with_columns(
            (pl.col("price") / pl.col("provider_price") - 1.0).abs().alias("price_gap")
        )
    )
    bad = merged.filter(pl.col("price_gap") > 0.15)
    if bad.is_empty():
        return []
    tickers = bad.sort("price_gap", descending=True)["ticker"].to_list()
    return [
        Warning_(
            "warning",
            (
                f"{bad.height} companies show a price more than 15% away from the market-data "
                f"provider's quote ({', '.join(tickers[:8])}). This usually means a stale price "
                "or a corporate action that has not been reflected on both sides."
            ),
            "prices",
        )
    ]


def _check_share_count_plausibility() -> list[Warning_]:
    """Cross-check the filing share count against net income divided by earnings per share.

    Two independent routes to the same number. When they disagree by more than a
    few percent, one of them is wrong, and every per-share figure built on it is
    wrong too.
    """
    fundamentals = store.scan("fundamentals")
    if fundamentals is None:
        return []

    latest = (
        fundamentals.filter(
            pl.col("concept").is_in(
                ["shares_diluted", "eps_diluted", "net_income", "net_income_to_common"]
            )
        )
        .sort("available_at")
        .group_by(["company_id", "concept"])
        .agg(pl.col("value").last(), pl.col("ticker").last())
        .collect()
    )
    wide = latest.pivot(on="concept", index=["company_id"], values="value")
    tickers = (
        latest.group_by("company_id").agg(pl.col("ticker").last()).rename({"ticker": "tk"})
    )
    wide = wide.join(tickers, on="company_id", how="left")

    for col in ("shares_diluted", "eps_diluted", "net_income"):
        if col not in wide.columns:
            return []

    # Earnings per share is struck on income *available to common*, after
    # preferred dividends, while `net_income` is the total. Either can be the
    # right numerator depending on the filer, and the two concepts have different
    # availability dates, so the check takes whichever agrees: the purpose is to
    # catch gross errors, and one route agreeing rules a gross error out.
    has_common = "net_income_to_common" in wide.columns
    checked = wide.filter(
        pl.col("shares_diluted").is_not_null()
        & pl.col("eps_diluted").is_not_null()
        & pl.col("net_income").is_not_null()
        & (pl.col("eps_diluted").abs() > 0.05)
    ).with_columns(
        (pl.col("net_income") / pl.col("eps_diluted")).alias("_implied_total"),
        (
            pl.col("net_income_to_common") / pl.col("eps_diluted")
            if has_common
            else pl.lit(None, dtype=pl.Float64)
        ).alias("_implied_common"),
    ).with_columns(
        pl.min_horizontal(
            (pl.col("shares_diluted") / pl.col("_implied_total") - 1.0).abs(),
            (pl.col("shares_diluted") / pl.col("_implied_common") - 1.0).abs(),
        ).alias("gap")
    )
    bad = checked.filter(pl.col("gap") > 0.15).sort("gap", descending=True)
    if bad.is_empty():
        return []
    names = [t for t in bad["tk"].to_list() if t][:8]
    return [
        Warning_(
            "warning",
            (
                f"{bad.height} companies show a reported diluted share count more than 15% away "
                f"from the count implied by net income divided by the filer's reported earnings "
                f"per share ({', '.join(names)}). Either figure can be the faulty one -- a stock "
                "split restates per-share history mid-year, which corrupts a trailing total. The "
                "app derives its own per-share figures from net income and the share count rather "
                "than using the filer's trailing total, so this is a flag to investigate rather "
                "than a defect in the numbers shown."
            ),
            "fundamentals",
        )
    ]


def _check_universe_membership_gap() -> list[Warning_]:
    """The survivorship bias we know about and cannot yet remove."""
    membership = store.read("universe_membership")
    if membership is None or membership.is_empty():
        return []
    removed = membership.filter(pl.col("date_removed").is_not_null()).height
    if removed > 0:
        return []
    return [
        Warning_(
            "info",
            (
                "Index membership is current-only: the constituents source lists who is in the "
                "index today and when they joined, but not who was removed. Entry dates prevent "
                "trading a company before the index picked it, which removes half the "
                "survivorship bias. The other half -- companies dropped after poor performance -- "
                "cannot be recovered from this source, so historical results are optimistic by an "
                "amount that cannot be measured here."
            ),
            "universe_membership",
        )
    ]


def _check_score_coverage() -> list[Warning_]:
    scores = store.read("opportunity_scores")
    if scores is None or scores.is_empty():
        return []
    latest = scores.filter(pl.col("date") == scores["date"].max())
    unscored = latest.filter(pl.col("score").is_null())
    out = []
    if unscored.height:
        tickers = [t for t in unscored["ticker"].to_list() if t][:8]
        out.append(
            Warning_(
                "info",
                (
                    f"{unscored.height} companies could not be scored because too little of the "
                    f"evidence was available ({', '.join(tickers)}). They are excluded from the "
                    "ranking rather than scored on partial data."
                ),
                "opportunity_scores",
            )
        )
    thin = latest.filter(pl.col("coverage") < 0.7)
    if thin.height > latest.height * 0.15:
        out.append(
            Warning_(
                "warning",
                (
                    f"{thin.height} of {latest.height} companies scored on less than 70% of the "
                    "available evidence. Their confidence scores are reduced accordingly."
                ),
                "opportunity_scores",
            )
        )
    return out


def _check_estimate_history() -> list[Warning_]:
    """Analyst estimates have no vintage, so the history has to accumulate."""
    estimates = store.read("estimates")
    if estimates is None or estimates.is_empty():
        return [
            Warning_(
                "warning",
                (
                    "No analyst estimates have been collected, so estimate revisions, forward "
                    "growth and earnings surprises are unavailable and the growth category is "
                    "scoring on trailing data alone."
                ),
                "estimates",
            )
        ]
    snapshots = estimates["as_of"].n_unique()
    if snapshots < 2:
        return [
            Warning_(
                "info",
                (
                    "Analyst estimates have been collected only once so far. The provider "
                    "publishes a current snapshot with no history, so revision trends measured "
                    "from our own stored snapshots will become available after several more "
                    "collections. Until then, revision direction comes from the provider's own "
                    "counts and estimate-change metrics are reported as unavailable rather than "
                    "as zero."
                ),
                "estimates",
            )
        ]
    return []


def _check_stale_filings() -> list[Warning_]:
    fundamentals = store.scan("fundamentals")
    if fundamentals is None:
        return []
    latest = (
        fundamentals.filter(pl.col("concept") == "revenue")
        .group_by("company_id")
        .agg(pl.col("available_at").max().alias("last"), pl.col("ticker").last())
        .collect()
    )
    if latest.is_empty():
        return []
    cutoff = datetime.now(UTC).date() - timedelta(days=150)
    stale = latest.filter(pl.col("last") < cutoff).sort("last")
    if stale.is_empty():
        return []
    tickers = [t for t in stale["ticker"].to_list() if t][:8]
    return [
        Warning_(
            "warning",
            (
                f"{stale.height} companies have not filed revenue figures in over 150 days "
                f"({', '.join(tickers)}). A quarter may be missing from their trailing figures."
            ),
            "fundamentals",
        )
    ]
