"""Risk, financial distress, and the value-trap assessment.

Three related questions, deliberately kept apart because they answer different
things and feed different decisions:

* **Risk** is how much the position could hurt. It sizes positions rather than
  eliminating candidates: a volatile company with an excellent score gets bought
  smaller, not skipped.
* **Distress** is whether the company might not survive. It is a hard constraint,
  not a sizing input.
* **Value-trap risk** is whether a cheap price is cheap for a reason. This is the
  single most useful check the product runs, because the valuation engine's
  natural failure mode is to fall in love with a declining business.

The value-trap logic deserves a note on design. It is a *signal count* rather
than a fitted model, and deliberately so: each signal is independently
meaningful, independently explainable, and the user can see exactly which ones
fired. A logistic regression over the same inputs might score marginally better
and would be impossible to argue with.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import polars as pl

log = logging.getLogger(__name__)

EPS = 1e-9


# ---------------------------------------------------------------------------
# Financial strength scores
# ---------------------------------------------------------------------------


def altman_z(
    *,
    working_capital: float | None,
    retained_earnings: float | None,
    ebit: float | None,
    market_cap: float | None,
    total_liabilities: float | None,
    revenue: float | None,
    assets: float | None,
) -> float | None:
    """Altman's Z-score for public manufacturers.

    Above 3 is safe, below 1.8 is the distress zone. It is not meaningful for
    banks, insurers or real-estate trusts, whose balance sheets are structurally
    different; the caller is responsible for not asking.
    """
    if not assets or assets <= 0 or not total_liabilities or total_liabilities <= 0:
        return None
    parts = [working_capital, retained_earnings, ebit, market_cap, revenue]
    if any(p is None or not math.isfinite(p) for p in parts):
        return None

    return (
        1.2 * (working_capital / assets)
        + 1.4 * (retained_earnings / assets)
        + 3.3 * (ebit / assets)
        + 0.6 * (market_cap / total_liabilities)
        + 1.0 * (revenue / assets)
    )


def distress_label(z: float | None) -> str:
    if z is None:
        return "Not assessed"
    if z >= 3.0:
        return "Safe"
    if z >= 1.8:
        return "Grey zone"
    return "Distress zone"


def piotroski_f_score(row: dict) -> int | None:
    """Nine binary tests of whether the financial position improved this year.

    A high score means profitability, leverage and efficiency all moved the right
    way. It is most useful exactly where this product needs help: separating a
    cheap company that is getting better from a cheap company that is not.
    """
    needed = ("net_income", "assets", "operating_cash_flow")
    if any(row.get(k) is None for k in needed):
        return None

    score = 0
    ni, assets, ocf = row["net_income"], row["assets"], row["operating_cash_flow"]
    if not assets or assets <= 0:
        return None

    roa = ni / assets
    # Profitability
    if ni > 0:
        score += 1
    if ocf > 0:
        score += 1
    prior_assets = row.get("assets__lag1y")
    prior_ni = row.get("net_income__lag1y")
    if prior_ni is not None and prior_assets and prior_assets > 0:
        if roa > prior_ni / prior_assets:
            score += 1
    if ocf > ni:  # cash exceeds accounting profit
        score += 1

    # Leverage, liquidity and dilution
    debt, prior_debt = row.get("total_debt"), row.get("total_debt__lag1y")
    if debt is not None and prior_debt is not None and assets > 0 and prior_assets:
        if (debt / assets) < (prior_debt / prior_assets):
            score += 1
    cr, prior_cr = row.get("current_ratio"), row.get("current_ratio__lag1y")
    if cr is not None and prior_cr is not None and cr > prior_cr:
        score += 1
    share_change = row.get("share_count_change")
    if share_change is not None and share_change >= 0:  # no net dilution
        score += 1

    # Operating efficiency
    gm, prior_gm = row.get("gross_margin"), row.get("gross_margin__lag1y")
    if gm is not None and prior_gm is not None and gm > prior_gm:
        score += 1
    rev, prior_rev = row.get("revenue"), row.get("revenue__lag1y")
    if rev is not None and prior_rev and prior_assets and prior_assets > 0 and assets > 0:
        if (rev / assets) > (prior_rev / prior_assets):
            score += 1

    return score


# ---------------------------------------------------------------------------
# Value-trap signals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrapSignalSpec:
    key: str
    label: str
    weight: float
    description: str


TRAP_SIGNALS: tuple[TrapSignalSpec, ...] = (
    TrapSignalSpec("revenue_declining", "Revenue is shrinking", 1.5,
                   "Trailing revenue is below the same figure a year ago."),
    TrapSignalSpec("margin_eroding", "Operating margin is eroding", 1.3,
                   "Operating margin has fallen by more than two points over the year."),
    TrapSignalSpec("fcf_deteriorating", "Free cash flow is deteriorating", 1.3,
                   "Free cash flow is negative or has fallen sharply."),
    TrapSignalSpec("negative_revisions", "Analysts are cutting estimates", 1.4,
                   "Earnings estimate revisions have been net negative over the past month."),
    TrapSignalSpec("excessive_debt", "Debt is high relative to earnings", 1.2,
                   "Net debt exceeds four times EBITDA."),
    TrapSignalSpec("weak_interest_cover", "Interest cover is thin", 1.2,
                   "Operating profit covers interest less than three times over."),
    TrapSignalSpec("persistent_dilution", "Shareholders are being diluted", 1.0,
                   "The share count has grown by more than two percent over the year."),
    TrapSignalSpec("weak_relative_strength", "The market disagrees", 1.2,
                   "The stock has underperformed the index by more than twenty points over six months."),
    TrapSignalSpec("below_long_average", "Price is below its long-term average", 0.8,
                   "The price sits below its two-hundred-day moving average and that average is falling."),
    TrapSignalSpec("accounting_quality", "Earnings are not backed by cash", 1.3,
                   "Accruals are high: reported profit substantially exceeds operating cash flow."),
    TrapSignalSpec("distress_risk", "Financial distress risk", 1.5,
                   "The Altman Z-score sits in the distress zone."),
    TrapSignalSpec("falling_returns", "Returns on capital are falling", 1.1,
                   "Return on invested capital has declined materially over the year."),
)

TRAP_BY_KEY = {s.key: s for s in TRAP_SIGNALS}
MAX_TRAP_WEIGHT = sum(s.weight for s in TRAP_SIGNALS)


@dataclass
class TrapAssessment:
    trap_score: float
    classification: str
    classification_label: str
    signals: list[dict] = field(default_factory=list)
    n_fired: int = 0


TRAP_LABELS = {
    "attractive_value": "Attractively undervalued",
    "speculative_value": "Speculatively undervalued",
    "possible_trap": "Possible value trap",
    "quality_compounder_expensive": "Overvalued high-quality compounder",
    "deteriorating_expensive": "Overvalued and deteriorating",
    "neutral": "No strong value signal",
}


def assess_value_trap(
    row: dict,
    *,
    valuation_class: str | None = None,
    quality_score: float | None = None,
) -> TrapAssessment:
    """Score how trap-like a company looks, and classify the value situation.

    The classification is the product's answer to "undervalued does not mean
    buy". A cheap company with deteriorating fundamentals is a different object
    from a cheap company with improving ones, and they deserve different words.
    """
    fired: list[dict] = []

    def check(key: str, condition: bool | None, value=None, detail: str | None = None):
        spec = TRAP_BY_KEY[key]
        is_fired = bool(condition) if condition is not None else False
        fired.append(
            {
                "key": key,
                "label": spec.label,
                "fired": is_fired,
                "value": float(value) if value is not None and math.isfinite(value) else None,
                "detail": detail or spec.description,
            }
        )

    rev_growth = row.get("revenue_growth_1y")
    check("revenue_declining", rev_growth is not None and rev_growth < -0.01, rev_growth)

    margin_trend = row.get("margin_trend")
    check("margin_eroding", margin_trend is not None and margin_trend < -0.02, margin_trend)

    fcf = row.get("free_cash_flow")
    fcf_growth = row.get("fcf_growth_3y")
    check(
        "fcf_deteriorating",
        (fcf is not None and fcf < 0) or (fcf_growth is not None and fcf_growth < -0.15),
        fcf_growth,
    )

    revisions = row.get("eps_revision_3m")
    check("negative_revisions", revisions is not None and revisions < -0.2, revisions)

    nd_ebitda = row.get("net_debt_to_ebitda")
    check("excessive_debt", nd_ebitda is not None and nd_ebitda > 4.0, nd_ebitda)

    coverage = row.get("interest_coverage")
    check("weak_interest_cover", coverage is not None and coverage < 3.0, coverage)

    share_change = row.get("share_count_change")
    check("persistent_dilution", share_change is not None and share_change < -0.02, share_change)

    rel = row.get("rel_strength_spy")
    check("weak_relative_strength", rel is not None and rel < -0.20, rel)

    px_ma = row.get("px_over_ma200")
    ma_slope = row.get("ma200_slope")
    check(
        "below_long_average",
        (px_ma is not None and px_ma < 0) and (ma_slope is not None and ma_slope < 0),
        px_ma,
    )

    accruals = row.get("accrual_ratio")
    check("accounting_quality", accruals is not None and accruals > 0.08, accruals)

    z = row.get("distress_score")
    check("distress_risk", z is not None and z < 1.8, z)

    roic = row.get("roic")
    roic_prior = row.get("roic__lag1y")
    falling_returns = (
        roic is not None and roic_prior is not None and (roic - roic_prior) < -0.03
    )
    check("falling_returns", falling_returns, roic)

    weight_fired = sum(TRAP_BY_KEY[s["key"]].weight for s in fired if s["fired"])
    trap_score = 100.0 * weight_fired / MAX_TRAP_WEIGHT
    n_fired = sum(1 for s in fired if s["fired"])

    classification = _classify_trap(trap_score, valuation_class, quality_score)
    return TrapAssessment(
        trap_score=trap_score,
        classification=classification,
        classification_label=TRAP_LABELS[classification],
        signals=fired,
        n_fired=n_fired,
    )


def _classify_trap(
    trap_score: float, valuation_class: str | None, quality_score: float | None
) -> str:
    """Turn the trap score and the valuation verdict into one of five situations."""
    cheap = valuation_class in ("significantly_undervalued", "undervalued")
    expensive = valuation_class in ("significantly_overvalued", "overvalued")
    high_quality = quality_score is not None and quality_score >= 65.0

    if cheap:
        if trap_score >= 35.0:
            return "possible_trap"
        if trap_score >= 18.0 or not high_quality:
            return "speculative_value"
        return "attractive_value"

    if expensive:
        if trap_score >= 30.0:
            return "deteriorating_expensive"
        if high_quality:
            return "quality_compounder_expensive"
        return "deteriorating_expensive" if trap_score >= 20.0 else "neutral"

    return "neutral"


# ---------------------------------------------------------------------------
# Composite risk score
# ---------------------------------------------------------------------------

RISK_COMPONENTS: tuple[tuple[str, str, float, int], ...] = (
    # (metric, label, weight, direction: +1 higher is riskier)
    ("volatility", "Volatility", 0.22, 1),
    ("downside_deviation", "Downside deviation", 0.16, 1),
    ("beta", "Market beta", 0.12, 1),
    ("downside_beta", "Downside beta", 0.10, 1),
    ("max_drawdown_1y", "One-year drawdown", 0.14, -1),
    ("amihud", "Illiquidity", 0.10, 1),
    ("net_debt_to_ebitda", "Leverage", 0.10, 1),
    ("distress_score", "Financial strength", 0.06, -1),
)


def build_risk(panel: pl.DataFrame) -> pl.DataFrame:
    """Risk metrics plus a 0-100 composite where higher means riskier.

    The composite is a percentile blend within the date, for the same reason the
    opportunity score is: an absolute volatility number means nothing without
    knowing what the rest of the universe is doing.
    """
    out = panel

    # Altman Z, for the business models where it means anything.
    out = out.with_columns(
        (pl.col("current_assets") - pl.col("current_liabilities")).alias("_working_capital"),
        pl.coalesce(
            pl.col("liabilities"),
            pl.col("assets") - pl.coalesce(pl.col("equity_incl_nci"), pl.col("equity")),
        ).alias("_total_liabilities"),
    )
    out = out.with_columns(
        pl.when(pl.col("business_model").is_in(["operating", "cyclical", "unprofitable_growth"]))
        .then(
            1.2 * (pl.col("_working_capital") / pl.col("assets"))
            + 1.4 * (pl.col("retained_earnings") / pl.col("assets"))
            + 3.3 * (pl.col("ebit") / pl.col("assets"))
            + 0.6 * (pl.col("market_cap") / pl.col("_total_liabilities"))
            + 1.0 * (pl.col("revenue") / pl.col("assets"))
        )
        .otherwise(None)
        .alias("distress_score")
    )

    # Piotroski's nine tests of whether the financial position improved. Useful
    # exactly where this product needs help: separating a cheap company that is
    # getting better from a cheap company that is not.
    scores: list[int | None] = []
    for row in out.iter_rows(named=True):
        scores.append(piotroski_f_score(row))
    out = out.with_columns(pl.Series("piotroski", scores, dtype=pl.Int32))

    parts: list[pl.Expr] = []
    weights: list[float] = []
    for metric, _label, weight, direction in RISK_COMPONENTS:
        if metric not in out.columns:
            continue
        rank = pl.col(metric).rank("average").over("date")
        n = pl.col(metric).is_not_null().sum().over("date")
        pct = (rank - 0.5) / n
        if direction < 0:
            pct = 1.0 - pct
        parts.append(
            pl.when(pl.col(metric).is_not_null()).then(pct * weight).otherwise(0.0)
        )
        weights.append(weight)
        out = out.with_columns(
            pl.when(pl.col(metric).is_not_null()).then(weight).otherwise(0.0).alias(
                f"_w_{metric}"
            )
        )

    if parts:
        observed = pl.sum_horizontal(
            [pl.col(f"_w_{m}") for m, _, _, _ in RISK_COMPONENTS if f"_w_{m}" in out.columns]
        )
        out = out.with_columns(
            pl.when(observed > 0)
            .then(pl.sum_horizontal(parts) / observed * 100.0)
            .otherwise(None)
            .alias("risk_score")
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("risk_score"))

    out = out.with_columns(
        pl.when(pl.col("risk_score") >= 75)
        .then(pl.lit("High"))
        .when(pl.col("risk_score") >= 50)
        .then(pl.lit("Above average"))
        .when(pl.col("risk_score") >= 25)
        .then(pl.lit("Below average"))
        .when(pl.col("risk_score").is_not_null())
        .then(pl.lit("Low"))
        .otherwise(pl.lit("Not assessed"))
        .alias("risk_label"),
        pl.when(pl.col("distress_score") >= 3.0)
        .then(pl.lit("Safe"))
        .when(pl.col("distress_score") >= 1.8)
        .then(pl.lit("Grey zone"))
        .when(pl.col("distress_score").is_not_null())
        .then(pl.lit("Distress zone"))
        .otherwise(pl.lit("Not assessed"))
        .alias("distress_label"),
    )

    drop = [c for c in out.columns if c.startswith("_w_") or c.startswith("_working") or c.startswith("_total_liab")]
    return out.drop([c for c in drop if c in out.columns])
