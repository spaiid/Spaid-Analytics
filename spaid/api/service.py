"""Assembling API responses from the stored tables.

Every number the interface displays is read from a persisted table, never
recomputed here. That is what makes the dashboard and the research journal agree:
there is one calculation, it happened in the pipeline, and this layer only
arranges the result.

This module also owns the qualitative readings -- bands, assessments, labels --
because the product brief forbids the interface from deciding what a number
means. The client receives both the value and the verdict.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime

import polars as pl

from spaid.api import schemas as S
from spaid.config.scoring import (
    ACTIVE_SCORING_SPEC,
    METRIC_BY_KEY,
    Category,
    score_band,
)
from spaid.config.settings import SETTINGS
from spaid.config.valuation import ACTIVE_VALUATION_SPEC
from spaid.pipeline import confidence as confidence_mod
from spaid.pipeline import quality
from spaid.pipeline.risk import TRAP_LABELS
from spaid.storage import store

log = logging.getLogger(__name__)

CATEGORY_LABELS = {
    Category.QUALITY: "Quality",
    Category.GROWTH: "Growth",
    Category.MOMENTUM: "Momentum",
    Category.VALUATION: "Valuation",
}

VALUATION_CLASS_LABELS = {
    "significantly_undervalued": "Significantly undervalued",
    "undervalued": "Undervalued",
    "fairly_valued": "Fairly valued",
    "overvalued": "Overvalued",
    "significantly_overvalued": "Significantly overvalued",
    "insufficient_confidence": "Insufficient confidence to classify",
}

# Compact forms for dense table cells. The full label is what the detail view
# shows; this is what fits in a column.
VALUATION_CLASS_SHORT = {
    "significantly_undervalued": "Very undervalued",
    "undervalued": "Undervalued",
    "fairly_valued": "Fairly valued",
    "overvalued": "Overvalued",
    "significantly_overvalued": "Very overvalued",
    "insufficient_confidence": "Not classified",
}

STATUS_DETAIL = {
    "scored": None,
    "missing": "No data available for this metric.",
    "not_applicable": "This metric is not meaningful for this kind of business.",
    "thin_peers": "Too few comparable companies to rank against.",
}

# How to render each metric's raw value. Kept server-side so the interface never
# has to know that a margin is a fraction and a multiple is not.
UNIT_BY_METRIC: dict[str, str] = {
    "roic": "percent", "roe": "percent", "gross_margin": "percent",
    "operating_margin": "percent", "fcf_margin": "percent",
    "fcf_conversion": "ratio", "accrual_ratio": "percent",
    "interest_coverage": "times", "net_debt_to_ebitda": "times",
    "share_count_change": "percent", "margin_stability": "score",
    "revenue_growth_1y": "percent", "revenue_growth_3y": "percent",
    "revenue_growth_5y": "percent", "eps_growth_1y": "percent",
    "eps_growth_3y": "percent", "fcf_growth_3y": "percent",
    "gross_profit_growth_3y": "percent", "growth_acceleration": "percent",
    "eps_revision_3m": "ratio", "revenue_revision_3m": "percent",
    "earnings_surprise": "percent", "forward_eps_growth": "percent",
    "mom_12_1": "percent", "mom_6_1": "percent", "ret_3m": "percent",
    "rel_strength_spy": "percent", "rel_strength_sector": "percent",
    "px_over_ma200": "percent", "ma200_slope": "percent",
    "momentum_consistency": "percent", "near_52w_high": "percent",
    "post_earnings_drift": "percent",
    "ev_ebit": "times", "ev_ebitda": "times", "fcf_yield": "percent",
    "earnings_yield": "percent", "forward_pe": "times", "ev_revenue": "times",
    "peg": "ratio", "valuation_vs_history": "score",
    "shareholder_yield": "percent", "price_to_book": "times",
}


def _format_value(value: float | None, unit: str) -> str | None:
    if value is None:
        return None
    if unit == "percent":
        return f"{value * 100:.1f}%"
    if unit == "times":
        return f"{value:.1f}x"
    if unit == "currency":
        return _compact(value)
    if unit == "score":
        return f"{value:.2f}"
    return f"{value:.2f}"


def _compact(value: float | None) -> str | None:
    if value is None:
        return None
    sign = "-" if value < 0 else ""
    v = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= threshold:
            return f"{sign}${v / threshold:.1f}{suffix}"
    return f"{sign}${v:,.0f}"


def _freshness() -> S.DataFreshness:
    info = store.info("prices")
    if info is None:
        return S.DataFreshness(detail="Prices have not been fetched yet.")
    age = info.age_hours
    limit = SETTINGS.freshness.prices_hours
    stale = age > limit
    return S.DataFreshness(
        as_of=date.fromisoformat(info.date_max) if info.date_max else None,
        age_hours=round(age, 1),
        is_stale=stale,
        detail=(
            f"Prices were last refreshed {age:.0f} hours ago, beyond the {limit:.0f}-hour "
            "threshold. Run the pipeline to update."
            if stale
            else f"Prices refreshed {age:.0f} hours ago."
        ),
    )


def _latest_date(table: str = "opportunity_scores") -> date | None:
    df = store.scan(table)
    if df is None:
        return None
    out = df.select(pl.col("date").max()).collect()
    return out.item() if out.height else None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def get_status(pipeline_state: dict | None = None) -> S.StatusResponse:
    scores = store.read("opportunity_scores")
    companies = store.read("companies")
    as_of = _latest_date()

    scored = 0
    universe = companies.height if companies is not None else 0
    if scores is not None and as_of is not None:
        latest = scores.filter(pl.col("date") == as_of)
        scored = int(latest["score"].is_not_null().sum())

    state = pipeline_state or {}
    return S.StatusResponse(
        as_of=as_of,
        universe_size=universe,
        scored_count=scored,
        has_scores=scored > 0,
        scoring_version=ACTIVE_SCORING_SPEC.version,
        valuation_version=ACTIVE_VALUATION_SPEC.version,
        portfolio_version=None,
        freshness=_freshness(),
        pipeline_running=bool(state.get("running")),
        pipeline_stage=state.get("stage"),
        pipeline_progress=state.get("progress"),
        pipeline_error=state.get("error"),
        message=(
            None
            if scored
            else "No scores have been computed yet. Run the pipeline to build them."
        ),
    )


# ---------------------------------------------------------------------------
# Opportunities
# ---------------------------------------------------------------------------


def get_opportunities(limit: int | None = None) -> S.OpportunitiesResponse:
    as_of = _latest_date()
    if as_of is None:
        return S.OpportunitiesResponse(
            freshness=_freshness(),
            notes=["No scores have been computed yet."],
        )

    scores = store.read("opportunity_scores", required=True).filter(pl.col("date") == as_of)
    companies = store.read("companies", required=True)
    valuations = store.read("valuations")
    traps = store.read("value_trap")
    risk = store.read("risk")
    metric_scores = store.read("metric_scores")

    df = scores.join(
        companies.select(["company_id", "name", "sector", "industry", "business_model"]),
        on="company_id",
        how="left",
    )
    if valuations is not None:
        df = df.join(
            valuations.select(
                [
                    "company_id", "price", "classification", "upside",
                    pl.col("range_low").alias("fair_value_low"),
                    pl.col("range_high").alias("fair_value_high"),
                ]
            ),
            on="company_id",
            how="left",
        )
    if traps is not None:
        df = df.join(
            traps.select(["company_id", pl.col("classification").alias("trap_class")]),
            on="company_id",
            how="left",
        )
    if risk is not None:
        df = df.join(risk.select(["company_id", "risk_score", "days_to_earnings"]), on="company_id", how="left")

    # The single strongest category, computed here rather than in the browser.
    top_reason = _top_reasons(metric_scores, as_of) if metric_scores is not None else {}

    df = df.sort("rank", nulls_last=True)
    # Count over the whole ranked set before truncating, so "112 of 500 scored"
    # describes the universe rather than the page.
    total_scored = int(df["score"].is_not_null().sum())
    if limit:
        df = df.head(limit)

    rows: list[S.OpportunityRow] = []
    for r in df.iter_rows(named=True):
        rows.append(
            S.OpportunityRow(
                rank=r.get("rank"),
                company_id=r["company_id"],
                ticker=r["ticker"],
                name=r.get("name") or r["ticker"],
                sector=r.get("sector"),
                industry=r.get("industry"),
                business_model=r.get("business_model"),
                price=r.get("price"),
                score=r.get("score"),
                band=r.get("band"),
                quality=r.get("quality"),
                growth=r.get("growth"),
                momentum=r.get("momentum"),
                valuation=r.get("valuation"),
                coverage=r.get("coverage"),
                confidence=r.get("confidence"),
                confidence_label=r.get("confidence_label"),
                valuation_class=r.get("classification"),
                valuation_label=VALUATION_CLASS_LABELS.get(r.get("classification") or ""),
                valuation_label_short=VALUATION_CLASS_SHORT.get(
                    r.get("classification") or ""
                ),
                upside=r.get("upside"),
                fair_value_low=r.get("fair_value_low"),
                fair_value_high=r.get("fair_value_high"),
                risk_score=r.get("risk_score"),
                trap_class=TRAP_LABELS.get(r.get("trap_class") or "", r.get("trap_class")),
                score_change_1w=r.get("score_change_1w"),
                score_change_1m=r.get("score_change_1m"),
                days_to_earnings=r.get("days_to_earnings"),
                top_reason=top_reason.get(r["company_id"]),
            )
        )

    sectors = sorted({s for s in companies["sector"].unique().to_list() if s})
    bands = [label for _, label in __import__(
        "spaid.config.scoring", fromlist=["SCORE_BANDS"]
    ).SCORE_BANDS]

    notes: list[str] = []
    unscored = companies.height - total_scored
    if unscored:
        notes.append(
            f"{unscored} companies are listed without a score because too little of the "
            "evidence was available to compute one."
        )

    return S.OpportunitiesResponse(
        as_of=as_of,
        universe_size=companies.height,
        scored_count=total_scored,
        rows=rows,
        sectors=sectors,
        bands=bands,
        spec_version=ACTIVE_SCORING_SPEC.version,
        freshness=_freshness(),
        notes=notes,
    )


def _top_reasons(metric_scores: pl.DataFrame, as_of: date) -> dict[str, str]:
    """The single metric contributing most to each company's score."""
    today = metric_scores.filter(
        (pl.col("date") == as_of) & (pl.col("status") == "scored")
    ).with_columns(((pl.col("score") - 50.0) * pl.col("weight")).alias("_impact"))
    best = (
        today.sort("_impact", descending=True)
        .group_by("company_id")
        .agg(pl.col("metric").first(), pl.col("score").first())
    )
    out: dict[str, str] = {}
    for r in best.iter_rows(named=True):
        spec = METRIC_BY_KEY.get(r["metric"])
        if spec is None:
            continue
        out[r["company_id"]] = f"{spec.label} ({r['score']:.0f})"
    return out


# ---------------------------------------------------------------------------
# Stock detail
# ---------------------------------------------------------------------------


def get_stock(ticker: str) -> S.StockDetail | None:
    ticker = ticker.strip().upper()
    securities = store.read("securities", required=True)
    match = securities.filter(
        (pl.col("ticker") == ticker) & pl.col("is_primary")
    )
    if match.is_empty():
        match = securities.filter(pl.col("ticker") == ticker)
    if match.is_empty():
        return None
    company_id = match["company_id"][0]

    companies = store.read("companies", required=True).filter(
        pl.col("company_id") == company_id
    )
    if companies.is_empty():
        return None
    company = companies.to_dicts()[0]

    as_of = _latest_date()
    scores = store.read("opportunity_scores")
    row = {}
    if scores is not None and as_of is not None:
        sub = scores.filter(
            (pl.col("company_id") == company_id) & (pl.col("date") == as_of)
        )
        if not sub.is_empty():
            row = sub.to_dicts()[0]

    detail = S.StockDetail(
        company_id=company_id,
        ticker=ticker,
        name=company.get("name") or ticker,
        sector=company.get("sector"),
        industry=company.get("industry"),
        business_model=company.get("business_model"),
        as_of=as_of,
        score=row.get("score"),
        rank=row.get("rank"),
        universe_size=row.get("universe_size"),
        percentile=row.get("percentile"),
        band=row.get("band") or score_band(row.get("score")),
        coverage=row.get("coverage"),
        freshness=_freshness(),
        spec_versions={
            "scoring": ACTIVE_SCORING_SPEC.version,
            "valuation": ACTIVE_VALUATION_SPEC.version,
        },
    )

    detail.categories = _categories(company_id, as_of)
    detail.score_changes = [
        S.ScoreChange(window="1 week", delta=row.get("score_change_1w")),
        S.ScoreChange(window="1 month", delta=row.get("score_change_1m")),
    ]
    detail.fair_value = _fair_value(company_id)
    detail.risk = _risk(company_id)
    detail.value_trap = _value_trap(company_id)
    detail.confidence = _confidence(company_id, row, detail)
    detail.price_history, detail.price, detail.price_change_1d = _price_history(ticker)
    detail.financial_trends = _financial_trends(company_id)
    detail.peers = _peers(company_id, company, as_of)
    detail.sources = _sources(company_id, ticker)

    strengths, weaknesses = _strengths_and_weaknesses(detail.categories)
    detail.strengths = strengths
    detail.weaknesses = weaknesses
    detail.risks = _risk_narrative(detail)
    detail.catalysts = _catalysts(company_id)

    if detail.fair_value is not None:
        detail.market_cap = None
    return detail


def _categories(company_id: str, as_of: date | None) -> list[S.CategoryDetail]:
    if as_of is None:
        return []
    metric_scores = store.read("metric_scores")
    category_scores = store.read("category_scores")
    if metric_scores is None or category_scores is None:
        return []

    ms = metric_scores.filter(
        (pl.col("company_id") == company_id) & (pl.col("date") == as_of)
    )
    cs = category_scores.filter(
        (pl.col("company_id") == company_id) & (pl.col("date") == as_of)
    )
    out: list[S.CategoryDetail] = []
    for cat in Category:
        crow = cs.filter(pl.col("category") == cat.value)
        if crow.is_empty():
            continue
        c = crow.to_dicts()[0]
        metrics = []
        sub = ms.filter(pl.col("category") == cat.value)
        observed_weight = float(
            sub.filter(pl.col("status") == "scored")["weight"].sum() or 0.0
        )
        for m in sub.sort(["status", "weight"], descending=[False, True]).iter_rows(named=True):
            spec = METRIC_BY_KEY.get(m["metric"])
            unit = UNIT_BY_METRIC.get(m["metric"], "ratio")
            contribution = (
                (m["score"] * m["weight"] / observed_weight)
                if m["status"] == "scored" and observed_weight > 0
                else None
            )
            metrics.append(
                S.MetricDetail(
                    key=m["metric"],
                    label=spec.label if spec else m["metric"],
                    category=cat.value,
                    description=spec.description if spec else "",
                    raw_value=m.get("raw_value"),
                    display_value=_format_value(m.get("raw_value"), unit),
                    unit=unit,
                    score=m.get("score"),
                    percentile=m.get("percentile"),
                    peer_basis=m.get("peer_basis"),
                    peer_group=m.get("peer_group"),
                    peer_count=m.get("peer_count"),
                    peer_median=m.get("peer_median"),
                    peer_median_display=_format_value(m.get("peer_median"), unit),
                    weight=m.get("weight") or 0.0,
                    contribution=contribution,
                    status=m.get("status") or "scored",
                    status_detail=STATUS_DETAIL.get(m.get("status") or "scored"),
                    direction=spec.direction if spec else 1,
                )
            )
        out.append(
            S.CategoryDetail(
                key=cat.value,
                label=CATEGORY_LABELS[cat],
                score=c.get("score"),
                weight=c.get("weight") or 0.0,
                contribution=c.get("contribution"),
                coverage=c.get("coverage") or 0.0,
                n_metrics=c.get("n_metrics") or 0,
                n_missing=c.get("n_missing") or 0,
                assessment=_category_assessment(c.get("score"), c.get("coverage")),
                metrics=metrics,
            )
        )
    return out


def _category_assessment(score: float | None, coverage: float | None) -> S.Assessment:
    if score is None:
        return S.Assessment(
            level="neutral",
            text=(
                "Not enough data to score this category. It has been excluded and the "
                "remaining categories reweighted."
            ),
        )
    if score >= 70:
        level, text = "good", "Ranks well ahead of its peers on this dimension."
    elif score >= 55:
        level, text = "good", "Ranks above its peers on this dimension."
    elif score >= 45:
        level, text = "neutral", "Roughly in line with its peers."
    elif score >= 30:
        level, text = "warning", "Ranks below its peers on this dimension."
    else:
        level, text = "critical", "Ranks near the bottom of its peer group."
    if coverage is not None and coverage < 0.7:
        text += f" Based on {coverage:.0%} of the metrics in this category."
    return S.Assessment(level=level, text=text)


def _fair_value(company_id: str) -> S.FairValue | None:
    valuations = store.read("valuations")
    if valuations is None:
        return None
    sub = valuations.filter(pl.col("company_id") == company_id)
    if sub.is_empty():
        return None
    v = sub.to_dicts()[0]

    methods = [
        S.MethodValue(
            method=m["method"],
            label=m["label"],
            value_per_share=m.get("value_per_share"),
            weight=m.get("weight") or 0.0,
            used=bool(m.get("used")),
            reason=m.get("reason") or (m.get("detail") if m.get("used") else None),
        )
        for m in json.loads(v.get("methods_used") or "[]")
    ]
    assumptions = json.loads(v.get("assumptions") or "{}")
    scenarios = [
        S.ScenarioValue(
            label=s["label"],
            value_per_share=s["value_per_share"],
            probability=s.get("probability") or 0.0,
            discount_rate=s.get("discount_rate"),
            terminal_growth=s.get("terminal_growth"),
            summary=s.get("summary"),
        )
        for s in assumptions.pop("scenarios", [])
        if s.get("value_per_share") is not None
    ]
    sensitivities = [
        S.Sensitivity(**{k: s[k] for k in
                         ("assumption", "description", "low_value", "high_value",
                          "base_value", "swing_pct") if k in s})
        for s in json.loads(v.get("sensitivities") or "[]")
    ]

    return S.FairValue(
        price=v["price"],
        bear=v.get("bear"),
        base=v.get("base"),
        bull=v.get("bull"),
        range_low=v.get("range_low"),
        range_high=v.get("range_high"),
        midpoint=v.get("midpoint"),
        range_midpoint=v.get("range_midpoint"),
        upside=v.get("upside"),
        classification=v.get("classification") or "insufficient_confidence",
        classification_label=VALUATION_CLASS_LABELS.get(
            v.get("classification") or "", "Not classified"
        ),
        confidence=v.get("confidence"),
        confidence_label=v.get("confidence_label"),
        business_model=v.get("business_model") or "operating",
        methods=methods,
        scenarios=scenarios,
        sensitivities=sensitivities,
        assumptions=assumptions,
        caveats=json.loads(v.get("caveats") or "[]"),
        method_dispersion=v.get("method_dispersion"),
        spec_version=v.get("spec_version") or ACTIVE_VALUATION_SPEC.version,
    )


def _risk(company_id: str) -> S.RiskDetail | None:
    risk = store.read("risk")
    if risk is None:
        return None
    sub = risk.filter(pl.col("company_id") == company_id)
    if sub.is_empty():
        return None
    r = sub.to_dicts()[0]
    score = r.get("risk_score")
    if score is None:
        level, text = "neutral", "Not enough data to assess risk."
        label = "Not assessed"
    elif score >= 75:
        level, label = "critical", "High"
        text = "Among the riskiest names in the index on volatility, drawdown and leverage."
    elif score >= 50:
        level, label = "warning", "Above average"
        text = "Carries more risk than the typical company in the index."
    elif score >= 25:
        level, label = "good", "Below average"
        text = "Carries less risk than the typical company in the index."
    else:
        level, label = "good", "Low"
        text = "Among the lowest-risk names in the index."

    return S.RiskDetail(
        volatility=r.get("volatility"),
        beta=r.get("beta"),
        downside_beta=r.get("downside_beta"),
        downside_deviation=r.get("downside_deviation"),
        max_drawdown_1y=r.get("max_drawdown_1y"),
        max_drawdown_5y=r.get("max_drawdown_5y"),
        dollar_volume_median=r.get("dollar_volume_median"),
        spread_bps=r.get("spread_bps_est"),
        days_to_earnings=r.get("days_to_earnings"),
        distress_score=r.get("distress_score"),
        distress_label=r.get("distress_label"),
        piotroski=r.get("piotroski"),
        risk_score=score,
        risk_label=label,
        assessment=S.Assessment(level=level, text=text),
    )


def _value_trap(company_id: str) -> S.ValueTrap | None:
    traps = store.read("value_trap")
    if traps is None:
        return None
    sub = traps.filter(pl.col("company_id") == company_id)
    if sub.is_empty():
        return None
    t = sub.to_dicts()[0]
    signals = [S.TrapSignal(**s) for s in json.loads(t.get("signals") or "[]")]
    return S.ValueTrap(
        trap_score=t.get("trap_score"),
        classification=t.get("classification") or "neutral",
        classification_label=TRAP_LABELS.get(t.get("classification") or "", "Not assessed"),
        signals=signals,
        n_fired=t.get("n_signals") or 0,
    )


def _confidence(company_id: str, row: dict, detail: S.StockDetail) -> S.Confidence | None:
    if not row:
        return None
    valuations = store.read("valuations")
    val_conf = None
    dispersion = None
    if valuations is not None:
        sub = valuations.filter(pl.col("company_id") == company_id)
        if not sub.is_empty():
            val_conf = sub["confidence"][0]
            dispersion = sub["method_dispersion"][0]

    fundamentals = store.scan("fundamentals")
    age_days = None
    if fundamentals is not None and detail.as_of is not None:
        latest = (
            fundamentals.filter(
                (pl.col("company_id") == company_id) & (pl.col("concept") == "revenue")
            )
            .select(pl.col("available_at").max())
            .collect()
        )
        if latest.height and latest.item() is not None:
            age_days = (detail.as_of - latest.item()).days

    n_missing = sum(c.n_missing for c in detail.categories)
    result = confidence_mod.assess(
        metric_coverage=row.get("coverage"),
        data_age_days=age_days,
        valuation_confidence=val_conf,
        method_dispersion=dispersion,
        score_change_1m=row.get("score_change_1m"),
        business_model=detail.business_model or "operating",
        n_analysts=None,
        n_missing_metrics=n_missing,
    )
    return S.Confidence(
        score=result.score,
        label=result.label,
        components=[S.ConfidenceComponent(**c) for c in result.components],
        caveats=result.caveats,
    )


def _price_history(ticker: str, days: int = 756) -> tuple[dict, float | None, float | None]:
    prices = store.scan("prices")
    if prices is None:
        return {}, None, None
    sub = (
        prices.filter(pl.col("ticker") == ticker)
        .select(["date", "close_raw", "close_adj"])
        .sort("date")
        .tail(days)
        .collect()
    )
    if sub.is_empty():
        return {}, None, None
    closes = sub["close_raw"].to_list()
    change = None
    if len(closes) >= 2 and closes[-2]:
        change = closes[-1] / closes[-2] - 1.0
    return (
        {"dates": [str(d) for d in sub["date"].to_list()], "close": closes},
        closes[-1],
        change,
    )


TREND_CONCEPTS = (
    ("revenue", "Revenue"),
    ("net_income", "Net income"),
    ("free_cash_flow", "Free cash flow"),
    ("operating_income", "Operating income"),
)


def _financial_trends(company_id: str) -> list[S.FinancialTrend]:
    fundamentals = store.scan("fundamentals")
    if fundamentals is None:
        return []
    out: list[S.FinancialTrend] = []
    for concept, label in TREND_CONCEPTS:
        sub = (
            fundamentals.filter(
                (pl.col("company_id") == company_id) & (pl.col("concept") == concept)
            )
            .select(["period_end", "available_at", "value"])
            .sort("period_end")
            .collect()
        )
        if sub.is_empty():
            continue
        # One point per period end, taking the most recently knowable value.
        sub = sub.unique(subset=["period_end"], keep="last").sort("period_end").tail(24)
        out.append(
            S.FinancialTrend(
                concept=concept,
                label=label,
                unit="currency",
                points=[
                    S.FinancialTrendPoint(
                        period_end=r["period_end"],
                        available_at=r["available_at"],
                        value=r["value"],
                    )
                    for r in sub.iter_rows(named=True)
                ],
            )
        )
    return out


def _peers(company_id: str, company: dict, as_of: date | None) -> list[S.PeerRow]:
    if as_of is None:
        return []
    scores = store.read("opportunity_scores")
    companies = store.read("companies")
    if scores is None or companies is None:
        return []

    same = companies.filter(pl.col("industry") == company.get("industry"))
    if same.height < 4:
        same = companies.filter(pl.col("sector") == company.get("sector"))

    merged = (
        scores.filter(pl.col("date") == as_of)
        .join(same.select(["company_id", "name"]), on="company_id", how="inner")
        .sort("score", descending=True, nulls_last=True)
        .head(12)
    )
    return [
        S.PeerRow(
            company_id=r["company_id"],
            ticker=r["ticker"],
            name=r.get("name") or r["ticker"],
            score=r.get("score"),
            quality=r.get("quality"),
            growth=r.get("growth"),
            momentum=r.get("momentum"),
            valuation=r.get("valuation"),
            is_self=r["company_id"] == company_id,
        )
        for r in merged.iter_rows(named=True)
    ]


def _sources(company_id: str, ticker: str) -> list[S.SourceRecord]:
    out: list[S.SourceRecord] = []
    fundamentals = store.scan("fundamentals")
    if fundamentals is not None:
        latest = (
            fundamentals.filter(pl.col("company_id") == company_id)
            .group_by("concept")
            .agg(pl.col("available_at").max().alias("as_of"))
            .collect()
        )
        for concept, label in TREND_CONCEPTS:
            row = latest.filter(pl.col("concept") == concept)
            if row.is_empty():
                continue
            out.append(
                S.SourceRecord(
                    field=label,
                    source="SEC XBRL company facts",
                    as_of=row["as_of"][0],
                    detail="Point-in-time: usable from the date the filing became public.",
                )
            )
    info = store.info("prices")
    if info is not None:
        out.append(
            S.SourceRecord(
                field="Price",
                source="Yahoo Finance",
                as_of=date.fromisoformat(info.date_max) if info.date_max else None,
                detail="Daily bars, split and dividend adjusted series kept separately.",
            )
        )
    snapshot = store.read("security_snapshot")
    if snapshot is not None:
        sub = snapshot.filter(pl.col("company_id") == company_id)
        if not sub.is_empty():
            out.append(
                S.SourceRecord(
                    field="Share count and forward consensus",
                    source="Yahoo Finance",
                    as_of=sub["as_of"][0],
                    detail="Current snapshot only; no history, so backtests do not read it.",
                )
            )
    return out


def _strengths_and_weaknesses(
    categories: list[S.CategoryDetail],
) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []
    for cat in categories:
        for m in cat.metrics:
            if m.status != "scored" or m.score is None:
                continue
            if m.score >= 85:
                strengths.append(
                    f"{m.label}: {m.display_value} ranks in the top "
                    f"{100 - m.score:.0f}% of {m.peer_count} {m.peer_basis} peers."
                )
            elif m.score <= 15:
                weaknesses.append(
                    f"{m.label}: {m.display_value} ranks in the bottom "
                    f"{m.score:.0f}% of {m.peer_count} {m.peer_basis} peers."
                )
    return strengths[:8], weaknesses[:8]


def _risk_narrative(detail: S.StockDetail) -> list[str]:
    out: list[str] = []
    if detail.risk is not None:
        r = detail.risk
        if r.distress_label == "Distress zone":
            out.append(
                "The Altman Z-score places this company in the financial distress zone."
            )
        if r.max_drawdown_1y is not None and r.max_drawdown_1y < -0.35:
            out.append(
                f"The shares fell {abs(r.max_drawdown_1y):.0%} peak to trough in the past year."
            )
        if r.volatility is not None and r.volatility > 0.45:
            out.append(f"Annualised volatility of {r.volatility:.0%} is high.")
        if r.days_to_earnings is not None and 0 <= r.days_to_earnings <= 14:
            out.append(
                f"Earnings are due in {r.days_to_earnings} days, which is an event risk."
            )
    if detail.value_trap is not None:
        for s in detail.value_trap.signals:
            if s.fired:
                out.append(s.detail or s.label)
    if detail.fair_value is not None:
        out.extend(detail.fair_value.caveats[:3])
    return out[:10]


def _catalysts(company_id: str) -> list[str]:
    events = store.read("earnings_events")
    out: list[str] = []
    if events is not None:
        future = events.filter(
            (pl.col("company_id") == company_id) & pl.col("is_future")
        ).sort("event_date")
        if not future.is_empty():
            d = future["event_date"][0]
            out.append(f"Next scheduled earnings release: {d}.")
    return out


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def get_health() -> S.HealthResponse:
    rows = [
        S.HealthRow(
            name=r["name"],
            layer=r["layer"],
            status=r["status"],
            rows=r.get("rows"),
            entities=r.get("entities"),
            updated_at=r.get("updated_at"),
            age_hours=r.get("age_hours"),
            max_age_hours=r.get("max_age_hours"),
            span=r.get("span"),
            source=r.get("source"),
            detail=r.get("detail"),
        )
        for r in store.health()
    ]
    warnings = [S.HealthWarning(**w) for w in quality.check_all()]
    return S.HealthResponse(
        tables=rows, warnings=warnings, checked_at=datetime.now(UTC)
    )
