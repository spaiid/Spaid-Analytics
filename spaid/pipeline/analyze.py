"""The analysis run: curated data in, scores and valuations out.

One function produces every derived table, in dependency order, on one explicit
date grid. Keeping it in a single place means the numbers on the dashboard, in a
backtest, and in the research journal can never come from three slightly
different code paths.

The grid deserves a note. Metrics are computed at weekly cadence over the full
history plus daily over the recent past. Weekly is enough for a product that
deliberately discourages turnover, and it turns a 1.7-million-row panel into a
400-thousand-row one, which is the difference between a run that takes minutes
and one that exhausts memory. The recent daily rows exist so that today's
recommendation reflects today's price rather than last Friday's.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from spaid.config.scoring import ACTIVE_SCORING_SPEC, Category, ScoringSpec
from spaid.config.settings import SETTINGS
from spaid.config.valuation import ACTIVE_VALUATION_SPEC, ValuationSpec
from spaid.pipeline import confidence as confidence_mod
from spaid.pipeline import risk as risk_mod
from spaid.pipeline import scoring
from spaid.pipeline.metrics import estimates as estimate_metrics
from spaid.pipeline.metrics import financial, price
from spaid.storage import store
from spaid.storage.schema import (
    OPPORTUNITY_SCORES,
    RISK,
    VALUATIONS,
    VALUE_TRAP,
    coerce,
)
from spaid.valuation.engine import CompanyValuationInputs, value_company

log = logging.getLogger(__name__)

EPS = 1e-9


def build_date_grid(
    prices: pl.DataFrame,
    *,
    weekly_years: int = 6,
    daily_days: int = 30,
    benchmark: str | None = None,
) -> list[date]:
    """Weekly history plus recent daily dates.

    Weekly resolution is ample for medium-term investing and keeps the panel a
    manageable size; the recent daily tail keeps today's view current.
    """
    benchmark = benchmark or SETTINGS.universe.benchmark
    sessions = (
        prices.filter(pl.col("ticker") == benchmark)["date"].unique().sort().to_list()
    )
    if not sessions:
        sessions = prices["date"].unique().sort().to_list()
    if not sessions:
        return []

    last = sessions[-1]
    weekly_cutoff = last - timedelta(days=365 * weekly_years)
    daily_cutoff = last - timedelta(days=daily_days)

    weekly: list[date] = []
    seen_weeks: set[tuple[int, int]] = set()
    # Walk backwards so the *last* session of each week is kept, which is the
    # one a weekly rebalance would actually trade on.
    for session in reversed(sessions):
        if session < weekly_cutoff:
            break
        key = session.isocalendar()[:2]
        if key not in seen_weeks:
            seen_weeks.add(key)
            weekly.append(session)

    daily = [s for s in sessions if s >= daily_cutoff]
    return sorted(set(weekly) | set(daily))


def _risk_free_rate() -> float:
    """Today's ten-year Treasury yield, or the configured fallback.

    Reads the market's own quote rather than a hard-coded number, because the
    discount rate is the single most influential assumption in the model and it
    should move when rates move.
    """
    from spaid.config.valuation import DcfSpec

    fallback = DcfSpec().discount.risk_free_fallback
    prices = store.read("prices")
    if prices is None:
        return fallback
    tnx = prices.filter(pl.col("ticker") == "^TNX").sort("date")
    if tnx.is_empty():
        return fallback
    value = tnx["close_raw"][-1]
    if value is None or not np.isfinite(value):
        return fallback
    # The index quotes the yield times ten.
    rate = float(value) / 100.0
    return rate if 0.001 < rate < 0.20 else fallback


def build_metric_panel(
    *,
    dates: list[date] | None = None,
    weekly_years: int = 6,
    daily_days: int = 30,
) -> pl.DataFrame:
    """Every metric for every company on the grid, one wide frame."""
    prices = store.read("prices", required=True)
    securities = store.read("securities", required=True)
    companies = store.read("companies", required=True)
    fundamentals = store.read("fundamentals", required=True)

    grid_dates = dates or build_date_grid(
        prices, weekly_years=weekly_years, daily_days=daily_days
    )
    log.info(
        "analysis grid: %d dates from %s to %s",
        len(grid_dates), grid_dates[0] if grid_dates else "-", grid_dates[-1] if grid_dates else "-",
    )

    sectors = (
        securities.filter(pl.col("is_primary"))
        .join(companies.select(["company_id", "sector"]), on="company_id", how="left")
        .select(["ticker", "sector"])
    )

    log.info("computing price metrics over the full history")
    price_metrics = price.build_price_metrics(prices, sectors=sectors)
    price_metrics = price_metrics.filter(pl.col("date").is_in(grid_dates))

    log.info("computing financial metrics")
    grid = financial.build_grid(prices, securities, dates=grid_dates)
    fin = financial.build_financial_metrics(grid, fundamentals, prices)

    panel = fin.join(
        price_metrics.drop(["close_raw", "close_adj", "split_factor"]),
        on=["ticker", "date"],
        how="left",
    ).join(
        companies.select(
            ["company_id", "name", "sector", "industry", "business_model"]
        ),
        on="company_id",
        how="left",
    )

    # A company with no profit for two years is analysed as unprofitable growth
    # regardless of what its sector label says. This is a state a business enters
    # and leaves, not a permanent classification.
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

    # Estimate-derived metrics have no history, so they attach to every date
    # with the same value and the confidence layer discounts accordingly.
    est = store.read("estimates")
    events = store.read("earnings_events")
    if est is not None and events is not None:
        em = estimate_metrics.build_estimate_metrics(est, events, prices, securities)
        panel = panel.join(em, on="company_id", how="left")
    else:
        for col in ("eps_revision_3m", "revenue_revision_3m", "forward_eps_growth",
                    "forward_eps", "earnings_surprise", "post_earnings_drift"):
            panel = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
        panel = panel.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("n_analysts"),
            pl.lit(None, dtype=pl.Int32).alias("days_to_earnings"),
            pl.lit(None, dtype=pl.Date).alias("next_earnings_date"),
        )

    # Share count and forward consensus from the second provider, used where the
    # filings have nothing. Dual-class issuers report share counts per class, so
    # for those companies this is the only source there is.
    snapshot = store.read("security_snapshot")
    if snapshot is not None and not snapshot.is_empty():
        snap = snapshot.select(
            ["company_id", "shares_outstanding", "market_cap", "forward_pe", "spread_bps"]
        ).rename(
            {
                "shares_outstanding": "_snap_shares",
                "market_cap": "_snap_market_cap",
                "forward_pe": "_snap_forward_pe",
                "spread_bps": "spread_bps",
            }
        )
        panel = panel.join(snap, on="company_id", how="left")
        panel = panel.with_columns(
            pl.when(pl.col("market_cap").is_null() & pl.col("_snap_market_cap").is_not_null())
            .then(pl.lit("provider_snapshot"))
            .when(pl.col("market_cap").is_not_null())
            .then(pl.lit("filings"))
            .otherwise(pl.lit("unavailable"))
            .alias("market_cap_source"),
            pl.coalesce(pl.col("market_cap"), pl.col("_snap_market_cap")).alias("market_cap"),
            pl.coalesce(pl.col("shares_current_basis"), pl.col("_snap_shares")).alias(
                "shares_current_basis"
            ),
            pl.coalesce(pl.col("forward_pe"), pl.col("_snap_forward_pe")).alias("forward_pe")
            if "forward_pe" in panel.columns
            else pl.col("_snap_forward_pe").alias("forward_pe"),
        ).drop("_snap_shares", "_snap_market_cap", "_snap_forward_pe")
    else:
        panel = panel.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("spread_bps"),
            pl.lit("filings").alias("market_cap_source"),
        )

    # Metrics that depend on the resolved market cap must be recomputed after
    # the provider fallback filled the gaps.
    panel = _recompute_market_cap_metrics(panel)
    panel = risk_mod.build_risk(panel)

    log.info("metric panel: %d rows x %d columns", panel.height, panel.width)
    return panel


def _recompute_market_cap_metrics(panel: pl.DataFrame) -> pl.DataFrame:
    """Redo the valuation ratios once the share-count fallback has been applied."""
    c = pl.col
    sd = financial.safe_div
    return panel.with_columns(
        (
            c("market_cap")
            + c("total_debt")
            + c("minority_interest").fill_null(0.0)
            + c("preferred_equity").fill_null(0.0)
            + c("operating_lease_liability").fill_null(0.0)
            - c("cash_and_equivalents")
        ).alias("enterprise_value")
    ).with_columns(
        sd(c("enterprise_value"), c("ebit"), positive_only=True).alias("ev_ebit"),
        sd(c("enterprise_value"), c("ebitda"), positive_only=True).alias("ev_ebitda"),
        sd(c("enterprise_value"), c("revenue"), positive_only=True).alias("ev_revenue"),
        sd(c("free_cash_flow"), c("market_cap"), positive_only=True).alias("fcf_yield"),
        sd(c("net_income"), c("market_cap"), positive_only=True).alias("earnings_yield"),
        sd(c("market_cap"), pl.coalesce(c("equity_incl_nci"), c("equity")), positive_only=True)
        .alias("price_to_book"),
        sd(
            c("dividends_paid").fill_null(0.0)
            + c("buyback").fill_null(0.0)
            - c("stock_issued").fill_null(0.0),
            c("market_cap"),
            positive_only=True,
        ).alias("shareholder_yield"),
    ).with_columns(
        # Growth-adjusted valuation. Defined only where growth is positive:
        # dividing a multiple by a negative growth rate produces a negative
        # figure that would sort as "cheap".
        pl.when(
            (c("forward_pe") > EPS) & (c("forward_eps_growth") > 0.01)
        )
        .then(c("forward_pe") / (c("forward_eps_growth") * 100.0))
        .otherwise(None)
        .alias("peg")
    )


# ---------------------------------------------------------------------------
# Valuation across the whole universe
# ---------------------------------------------------------------------------


def _peer_arrays(panel_date: pl.DataFrame, row: dict, keys: tuple[str, ...]) -> dict:
    """Multiples of the subject's industry peers, excluding the subject itself."""
    peers = panel_date.filter(
        (pl.col("industry") == row["industry"])
        & (pl.col("company_id") != row["company_id"])
    )
    if peers.height < 6:
        peers = panel_date.filter(
            (pl.col("sector") == row["sector"])
            & (pl.col("company_id") != row["company_id"])
        )
    return {
        key: peers[key].drop_nulls().to_numpy()
        for key in keys
        if key in peers.columns
    }, peers


def build_valuations(
    panel: pl.DataFrame,
    *,
    as_of: date,
    spec: ValuationSpec | None = None,
    history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Fair value for every company on one date."""
    spec = spec or ACTIVE_VALUATION_SPEC
    today = panel.filter(pl.col("date") == as_of)
    if today.is_empty():
        raise ValueError(f"no panel rows for {as_of}")

    rf = _risk_free_rate()
    log.info("valuing %d companies as of %s (risk-free %.2f%%)", today.height, as_of, rf * 100)

    multiple_keys = ("ev_ebit", "ev_ebitda", "fcf_yield", "trailing_pe", "price_to_book", "forward_pe")
    hist = history if history is not None else panel

    records: list[dict] = []
    failures = 0

    for row in today.iter_rows(named=True):
        peer_multiples, peers = _peer_arrays(today, row, multiple_keys)
        own_history: dict[str, np.ndarray] = {}
        margin_history = None
        historical_growth = None
        historical_margin = None

        company_history = hist.filter(pl.col("company_id") == row["company_id"])
        if not company_history.is_empty():
            for key in ("ev_ebit", "ev_ebitda", "trailing_pe", "price_to_book"):
                if key in company_history.columns:
                    vals = company_history[key].drop_nulls().to_numpy()
                    if vals.size:
                        own_history[key] = vals
            if "operating_margin" in company_history.columns:
                margins = company_history["operating_margin"].drop_nulls().to_numpy()
                if margins.size:
                    margin_history = margins
                    historical_margin = float(np.median(margins))
            if "revenue_growth_1y" in company_history.columns:
                growths = company_history["revenue_growth_1y"].drop_nulls().to_numpy()
                if growths.size:
                    historical_growth = float(np.median(growths))

        data_age = None
        if row.get("revenue__asof"):
            data_age = (as_of - row["revenue__asof"]).days

        inputs = CompanyValuationInputs(
            ticker=row["ticker"],
            price=row["close_raw"],
            business_model=row.get("business_model") or "operating",
            shares=row.get("shares_current_basis"),
            revenue=row.get("revenue"),
            operating_income=row.get("operating_income"),
            ebitda=row.get("ebitda"),
            net_income=row.get("net_income"),
            free_cash_flow=row.get("free_cash_flow"),
            share_based_comp=row.get("share_based_comp"),
            depreciation=row.get("depreciation_amortization"),
            capex=row.get("capex"),
            equity=row.get("equity"),
            net_debt=row.get("net_debt"),
            minority_interest=row.get("minority_interest"),
            preferred_equity=row.get("preferred_equity"),
            intangibles=row.get("intangibles"),
            operating_margin=row.get("operating_margin"),
            tax_rate=row.get("effective_tax_rate"),
            roic=row.get("roic"),
            roe=row.get("roe"),
            revenue_growth=row.get("revenue_growth_1y"),
            forward_eps_growth=row.get("forward_eps_growth"),
            forward_eps=row.get("forward_eps"),
            sales_to_capital=_sales_to_capital(row),
            beta=row.get("beta"),
            market_cap=row.get("market_cap"),
            risk_free_rate=rf,
            premiums_earned=row.get("premiums_earned"),
            policy_benefits=row.get("policy_benefits"),
            peer_multiples=peer_multiples,
            peer_growth=peers["revenue_growth_1y"].drop_nulls().to_numpy()
            if "revenue_growth_1y" in peers.columns
            else None,
            peer_margin=peers["operating_margin"].drop_nulls().to_numpy()
            if "operating_margin" in peers.columns
            else None,
            own_multiple_history=own_history,
            margin_history=margin_history,
            historical_growth=historical_growth,
            historical_margin=historical_margin,
            data_age_days=data_age,
            metric_coverage=row.get("_coverage"),
            n_analysts=row.get("n_analysts"),
        )

        try:
            result = value_company(inputs, spec)
        except Exception as exc:
            failures += 1
            log.debug("%s: valuation failed (%s)", row["ticker"], exc)
            continue

        records.append(
            {
                "company_id": row["company_id"],
                "ticker": row["ticker"],
                "date": as_of,
                "price": result.price,
                "bear": result.bear,
                "base": result.base,
                "bull": result.bull,
                "range_low": result.range_low,
                "range_high": result.range_high,
                "midpoint": result.midpoint,
                "upside": result.upside,
                "classification": result.classification,
                "confidence": result.confidence,
                "confidence_label": result.confidence_label,
                "business_model": result.business_model,
                "methods_used": json.dumps(
                    [
                        {
                            "method": m.method,
                            "label": m.label,
                            "weight": m.weight,
                            "value_per_share": m.value_per_share,
                            "used": m.used,
                            "reason": m.reason,
                            "detail": m.detail,
                        }
                        for m in result.methods
                    ]
                ),
                "method_dispersion": result.method_dispersion,
                "assumptions": json.dumps(
                    {**result.assumptions, "scenarios": result.scenarios}, default=str
                ),
                "sensitivities": json.dumps(result.sensitivities, default=str),
                "caveats": json.dumps(result.caveats),
                "spec_version": spec.version,
                "computed_at": datetime.now(UTC),
            }
        )

    if failures:
        log.warning("valuation failed for %d companies", failures)
    if not records:
        raise RuntimeError("no valuations could be produced")

    df = coerce(pl.DataFrame(records, infer_schema_length=None), VALUATIONS)
    classified = df.filter(pl.col("classification") != "insufficient_confidence").height
    log.info(
        "valuations: %d companies, %d classified, median upside %.1f%%",
        df.height, classified, (df["upside"].median() or 0) * 100,
    )
    return df


def _sales_to_capital(row: dict) -> float | None:
    """Revenue per dollar of invested capital: how expensive growth is here."""
    revenue, capital = row.get("revenue"), row.get("invested_capital")
    if not revenue or not capital or capital <= 0:
        return None
    ratio = revenue / capital
    return ratio if 0.1 < ratio < 20 else None


# ---------------------------------------------------------------------------
# The full run
# ---------------------------------------------------------------------------


def run_analysis(
    *,
    scoring_spec: ScoringSpec | None = None,
    valuation_spec: ValuationSpec | None = None,
    weekly_years: int = 6,
    daily_days: int = 30,
    persist: bool = True,
) -> dict:
    """Compute every derived table from the curated data."""
    started = datetime.now(UTC)
    scoring_spec = scoring_spec or ACTIVE_SCORING_SPEC
    valuation_spec = valuation_spec or ACTIVE_VALUATION_SPEC

    panel = build_metric_panel(weekly_years=weekly_years, daily_days=daily_days)
    as_of = panel["date"].max()

    log.info("scoring %d rows", panel.height)
    metric_scores = scoring.build_metric_scores(panel, scoring_spec)
    category_scores = scoring.build_category_scores(metric_scores, scoring_spec)
    opportunity = scoring.build_opportunity_scores(category_scores, scoring_spec)

    # Coverage feeds the confidence system, so it has to be on the panel before
    # valuations run.
    coverage = opportunity.select(["company_id", "date", "coverage"]).rename(
        {"coverage": "_coverage"}
    )
    panel = panel.join(coverage, on=["company_id", "date"], how="left")

    valuations = build_valuations(
        panel, as_of=as_of, spec=valuation_spec, history=panel
    )

    # --- value traps -------------------------------------------------------
    today_panel = panel.filter(pl.col("date") == as_of)
    quality = (
        category_scores.filter(
            (pl.col("date") == as_of) & (pl.col("category") == Category.QUALITY.value)
        )
        .select(["company_id", pl.col("score").alias("_quality")])
    )
    val_class = valuations.select(["company_id", "classification"])
    trap_input = today_panel.join(quality, on="company_id", how="left").join(
        val_class, on="company_id", how="left"
    )

    trap_records = []
    for row in trap_input.iter_rows(named=True):
        assessment = risk_mod.assess_value_trap(
            row,
            valuation_class=row.get("classification"),
            quality_score=row.get("_quality"),
        )
        trap_records.append(
            {
                "company_id": row["company_id"],
                "ticker": row["ticker"],
                "date": as_of,
                "trap_score": assessment.trap_score,
                "classification": assessment.classification,
                "signals": json.dumps(assessment.signals),
                "n_signals": assessment.n_fired,
                "spec_version": scoring_spec.version,
                "computed_at": datetime.now(UTC),
            }
        )
    traps = coerce(pl.DataFrame(trap_records, infer_schema_length=None), VALUE_TRAP)

    # --- risk table --------------------------------------------------------
    risk_cols = [c for c in RISK.schema if c in today_panel.columns]
    risk_table = today_panel.select(risk_cols).with_columns(
        pl.lit(scoring_spec.version).alias("spec_version"),
        pl.lit(datetime.now(UTC)).alias("computed_at"),
    )
    risk_table = coerce(risk_table, RISK)

    # --- confidence --------------------------------------------------------
    opportunity = _attach_confidence(opportunity, panel, valuations, as_of)

    if persist:
        store.write(metric_scores, "metric_scores", source="spaid")
        store.write(category_scores, "category_scores", source="spaid")
        store.write(opportunity, "opportunity_scores", source="spaid")
        store.write(valuations, "valuations", source="spaid")
        store.write(traps, "value_trap", source="spaid")
        store.write(risk_table, "risk", source="spaid")

    elapsed = (datetime.now(UTC) - started).total_seconds()
    today_scores = opportunity.filter(pl.col("date") == as_of)
    summary = {
        "as_of": str(as_of),
        "grid_dates": panel["date"].n_unique(),
        "companies": panel["company_id"].n_unique(),
        "scored": int(today_scores["score"].is_not_null().sum()),
        "median_score": float(today_scores["score"].median() or float("nan")),
        "valuations": valuations.height,
        "classified": int(
            valuations.filter(pl.col("classification") != "insufficient_confidence").height
        ),
        "scoring_version": scoring_spec.version,
        "valuation_version": valuation_spec.version,
        "elapsed_seconds": round(elapsed, 1),
    }
    log.info("analysis complete: %s", summary)
    return summary


def _attach_confidence(
    opportunity: pl.DataFrame,
    panel: pl.DataFrame,
    valuations: pl.DataFrame,
    as_of: date,
) -> pl.DataFrame:
    """Compute the overall recommendation confidence for the current date."""
    context = (
        panel.filter(pl.col("date") == as_of)
        .select(
            [
                "company_id", "business_model", "n_analysts",
                pl.col("revenue__asof").alias("_revenue_asof"),
            ]
        )
        .join(
            valuations.select(
                [
                    "company_id",
                    pl.col("confidence").alias("_val_conf"),
                    "method_dispersion",
                ]
            ),
            on="company_id",
            how="left",
        )
    )
    today = opportunity.filter(pl.col("date") == as_of).join(
        context, on="company_id", how="left"
    )

    scores: list[float] = []
    labels: list[str] = []
    for row in today.iter_rows(named=True):
        age = (
            (as_of - row["_revenue_asof"]).days if row.get("_revenue_asof") else None
        )
        result = confidence_mod.assess(
            metric_coverage=row.get("coverage"),
            data_age_days=age,
            valuation_confidence=row.get("_val_conf"),
            method_dispersion=row.get("method_dispersion"),
            score_change_1m=row.get("score_change_1m"),
            business_model=row.get("business_model") or "operating",
            n_analysts=row.get("n_analysts"),
        )
        scores.append(result.score)
        labels.append(result.label)

    today = today.with_columns(
        pl.Series("confidence", scores), pl.Series("confidence_label", labels)
    ).select(opportunity.columns)

    other = opportunity.filter(pl.col("date") != as_of)
    return coerce(
        pl.concat([other, today], how="vertical_relaxed").sort(["date", "rank"]),
        OPPORTUNITY_SCORES,
    )
