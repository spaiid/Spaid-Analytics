"""The fair-value engine: run the applicable methods, blend them, classify.

The output is deliberately a *range* with a confidence attached, never a single
number. A fair value of $184.37 implies a precision nobody has; a range of $150
to $230 with moderate confidence is both more honest and more useful, because it
tells you when the market price is inside the range and the answer is "this is
roughly fairly valued, do something else".

How the blend works:

1. The business model selects which methods are appropriate and what weight each
   carries. A bank does not get a free-cash-flow model at all.
2. Each method runs. A method that cannot run on the available data returns
   nothing and says why, and its weight redistributes across the survivors.
3. The base case is the weighted average of the methods. Bear and bull come from
   the discounted cash-flow scenarios where available, widened by how much the
   methods disagree with each other.
4. Classification compares price with the range, scaled by the range's width, so
   a wide uncertain range demands more upside before it will call something
   undervalued.
5. Confidence falls when methods disagree, when data is thin or stale, when the
   estimate is sensitive to assumptions, and when most of the value sits in the
   terminal period.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from spaid.config.scoring import BusinessModel
from spaid.config.valuation import (
    ClassificationSpec,
    ValuationClass,
    ValuationMethod,
    ValuationSpec,
)
from spaid.valuation import industry, relative
from spaid.valuation.dcf import DcfInputs, cost_of_equity, run_scenarios, sensitivity

log = logging.getLogger(__name__)

EPS = 1e-9

CLASS_LABELS: dict[str, str] = {
    ValuationClass.SIGNIFICANTLY_UNDERVALUED: "Significantly undervalued",
    ValuationClass.UNDERVALUED: "Undervalued",
    ValuationClass.FAIRLY_VALUED: "Fairly valued",
    ValuationClass.OVERVALUED: "Overvalued",
    ValuationClass.SIGNIFICANTLY_OVERVALUED: "Significantly overvalued",
    ValuationClass.INSUFFICIENT_CONFIDENCE: "Insufficient confidence to classify",
}

METHOD_LABELS: dict[str, str] = {
    ValuationMethod.DCF: "Discounted cash flow",
    ValuationMethod.COMPARABLES: "Comparable companies",
    ValuationMethod.HISTORICAL: "Own valuation history",
    ValuationMethod.RESIDUAL_INCOME: "Residual income",
    ValuationMethod.NAV_AFFO: "AFFO and net asset value",
    ValuationMethod.NORMALIZED_EARNINGS: "Normalised mid-cycle earnings",
}


@dataclass
class MethodOutcome:
    method: str
    label: str
    weight: float
    value_per_share: float | None = None
    used: bool = False
    reason: str | None = None
    detail: str = ""
    assumptions: dict = field(default_factory=dict)
    # Where a method can express its own uncertainty -- the spread of the peer
    # group, for instance -- it reports it here, so a lone method still has a
    # defensible range instead of a decorative one.
    value_low: float | None = None
    value_high: float | None = None


@dataclass
class ValuationResult:
    price: float
    bear: float | None = None
    base: float | None = None
    bull: float | None = None
    range_low: float | None = None
    range_high: float | None = None
    # The headline estimate: the weight-blended base case. Named `midpoint` for
    # the schema's sake, but it is no longer the centre of the range -- see
    # `range_midpoint` for that, and the note where both are assigned.
    midpoint: float | None = None
    range_midpoint: float | None = None
    upside: float | None = None
    classification: str = ValuationClass.INSUFFICIENT_CONFIDENCE
    classification_label: str = CLASS_LABELS[ValuationClass.INSUFFICIENT_CONFIDENCE]
    confidence: float = 0.0
    confidence_label: str = "Very low"
    business_model: str = BusinessModel.OPERATING
    methods: list[MethodOutcome] = field(default_factory=list)
    scenarios: list[dict] = field(default_factory=list)
    sensitivities: list[dict] = field(default_factory=list)
    assumptions: dict = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    method_dispersion: float | None = None
    spec_version: str = ""


def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Moderate"
    if score >= 0.35:
        return "Low"
    return "Very low"


@dataclass(frozen=True)
class CompanyValuationInputs:
    """Everything the engine needs about one company at one date."""

    ticker: str
    price: float
    business_model: str
    shares: float | None = None
    # Income statement and cash flow (trailing twelve months)
    revenue: float | None = None
    operating_income: float | None = None
    ebitda: float | None = None
    net_income: float | None = None
    free_cash_flow: float | None = None
    share_based_comp: float | None = None
    depreciation: float | None = None
    capex: float | None = None
    # Balance sheet
    equity: float | None = None
    net_debt: float | None = None
    minority_interest: float | None = None
    preferred_equity: float | None = None
    intangibles: float | None = None
    # Ratios and drivers
    operating_margin: float | None = None
    tax_rate: float | None = None
    roic: float | None = None
    roe: float | None = None
    revenue_growth: float | None = None
    forward_eps_growth: float | None = None
    forward_eps: float | None = None
    sales_to_capital: float | None = None
    consensus_eps_this_year: float | None = None
    consensus_revenue_this_year: float | None = None
    consensus_revenue_next_year: float | None = None
    consensus_revenue_next_low: float | None = None
    consensus_revenue_next_high: float | None = None
    n_revenue_analysts: int | None = None
    # Needed to restate consensus (non-GAAP) earnings onto the GAAP basis the
    # projection runs on -- see `_consensus_path`.
    amortization_intangibles: float | None = None
    beta: float | None = None
    market_cap: float | None = None
    risk_free_rate: float | None = None
    # Insurance-specific
    premiums_earned: float | None = None
    policy_benefits: float | None = None
    # Peer and history context
    peer_multiples: dict[str, np.ndarray] = field(default_factory=dict)
    peer_growth: np.ndarray | None = None
    peer_margin: np.ndarray | None = None
    own_multiple_history: dict[str, np.ndarray] = field(default_factory=dict)
    margin_history: np.ndarray | None = None
    historical_growth: float | None = None
    historical_margin: float | None = None
    # Data quality context, for the confidence score
    data_age_days: float | None = None
    metric_coverage: float | None = None
    n_analysts: int | None = None


def _growth_estimate(inputs: CompanyValuationInputs) -> float:
    """The starting growth rate: analyst consensus blended with recent history.

    Neither alone is trustworthy. History does not know about a product cycle;
    consensus is systematically optimistic. A blend is less wrong than either,
    and where one is missing the other stands alone.
    """
    trailing = inputs.revenue_growth
    forward = inputs.forward_eps_growth
    values = [v for v in (trailing, forward) if v is not None and math.isfinite(v)]
    if not values:
        return 0.03
    if len(values) == 1:
        return max(min(values[0], 0.40), -0.25)
    # Weight the forward estimate slightly higher; it is the only input that
    # knows anything about the future at all.
    blended = 0.45 * trailing + 0.55 * forward
    return max(min(blended, 0.40), -0.25)


def _consensus_path(
    inputs: CompanyValuationInputs,
) -> tuple[tuple[float, ...], tuple[float | None, ...], float, float]:
    """The consensus revenue path, the margin it implies each year, and its spread.

    Returns empty when coverage is too thin or the estimates do not describe
    growth, which leaves the projection on its own trailing assumptions.
    """
    pairs = [
        (inputs.consensus_revenue_this_year, inputs.consensus_eps_this_year),
        (inputs.consensus_revenue_next_year, inputs.forward_eps),
    ]
    levels: list[float] = []
    margins: list[float | None] = []

    tax = inputs.tax_rate if inputs.tax_rate is not None else 0.21
    tax = min(max(tax, 0.0), 0.60)

    # Consensus earnings are quoted non-GAAP, and the projection is GAAP.
    #
    # Analysts publish adjusted earnings: the two items almost every company
    # excludes are amortisation of acquired intangibles and share-based
    # compensation. The projection starts from `OperatingIncomeLoss`, which
    # charges both. Reading a non-GAAP margin onto a GAAP projection therefore
    # grants a margin expansion that is purely a change of accounting basis --
    # for AMD it lifted 15.71% to 34.42% and held it for eight years, worth over
    # half the fair value, and it silently undid the model's own decision to
    # treat stock compensation as a cost.
    #
    # Restating the implied margin back onto a GAAP basis is the correction. It
    # is partial by construction, since a given company's adjustments may
    # include restructuring or litigation the filings do not separate, but it
    # removes the two that are large, universal and separately reported.
    addback_ratio = 0.0
    if inputs.revenue and inputs.revenue > 0:
        addbacks = (inputs.amortization_intangibles or 0.0) + (
            inputs.share_based_comp or 0.0
        )
        if math.isfinite(addbacks) and addbacks > 0:
            addback_ratio = addbacks / inputs.revenue

    for revenue, eps in pairs:
        if revenue is None or not math.isfinite(revenue) or revenue <= 0:
            break
        levels.append(revenue)
        # The margin consensus earnings imply on that revenue. Earnings are
        # after interest and tax, so they are grossed back up to an operating
        # figure; net interest is left out because its sign differs by company
        # and the error is small next to the margin change being measured.
        margin = None
        if eps and inputs.shares and math.isfinite(eps):
            candidate = (eps * inputs.shares) / (1.0 - tax) / revenue
            candidate -= addback_ratio
            if math.isfinite(candidate) and 0.0 < candidate < 0.90:
                margin = candidate
        margins.append(margin)

    if not levels or not inputs.revenue or inputs.revenue <= 0:
        return (), (), 1.0, 1.0

    far = levels[-1]
    low_ratio = high_ratio = 1.0
    if inputs.consensus_revenue_next_low and far > EPS:
        low_ratio = min(max(inputs.consensus_revenue_next_low / far, 0.4), 1.0)
    if inputs.consensus_revenue_next_high and far > EPS:
        high_ratio = min(max(inputs.consensus_revenue_next_high / far, 1.0), 2.5)

    return tuple(levels), tuple(margins), low_ratio, high_ratio


def _run_dcf(inputs: CompanyValuationInputs, spec: ValuationSpec) -> tuple[dict, list, list]:
    """Returns (scenario results, scenario dicts, sensitivity rows)."""
    if not inputs.shares or not inputs.revenue or inputs.operating_margin is None:
        return {}, [], []

    consensus_levels, consensus_margins, low_ratio, high_ratio = _consensus_path(inputs)
    dcf_inputs = DcfInputs(
        revenue=inputs.revenue,
        operating_margin=inputs.operating_margin,
        tax_rate=inputs.tax_rate or spec.dcf.tax_rate_default,
        shares=inputs.shares,
        net_debt=inputs.net_debt or 0.0,
        minority_interest=inputs.minority_interest or 0.0,
        preferred_equity=inputs.preferred_equity or 0.0,
        share_based_comp=inputs.share_based_comp or 0.0,
        growth_rate=_growth_estimate(inputs),
        sales_to_capital=inputs.sales_to_capital or spec.dcf.sales_to_capital_default,
        roic=inputs.roic,
        beta=inputs.beta or 1.0,
        market_cap=inputs.market_cap,
        risk_free_rate=inputs.risk_free_rate,
        reported_free_cash_flow=inputs.free_cash_flow,
        consensus_revenue=consensus_levels,
        consensus_low_ratio=low_ratio,
        consensus_high_ratio=high_ratio,
        consensus_margins=consensus_margins,
        n_revenue_analysts=inputs.n_revenue_analysts,
    )
    results = run_scenarios(dcf_inputs, spec.dcf, risk_free=inputs.risk_free_rate)
    probabilities = {s.label: s.probability for s in spec.dcf.scenarios}
    scenarios = [
        {
            "label": label,
            "value_per_share": r.value_per_share,
            "probability": probabilities.get(label, 0.0),
            "discount_rate": r.discount_rate,
            "terminal_growth": r.terminal_growth,
            "terminal_share": r.terminal_share,
            "summary": (
                f"{label.title()} case discounts at {r.discount_rate:.1%} with "
                f"{r.terminal_growth:.1%} terminal growth"
            ),
        }
        for label, r in results.items()
    ]
    sens = sensitivity(dcf_inputs, spec.dcf, risk_free=inputs.risk_free_rate)
    return results, scenarios, sens


def _preferred_multiples(model: BusinessModel) -> tuple[str, ...] | None:
    """Which multiples make sense for this kind of business.

    For a cyclical company, earnings-based multiples are the trap the normalised
    model exists to avoid: applying a peer multiple to peak earnings values a
    fertiliser producer at four times the market at the top of its cycle. Revenue
    is far less cyclical than margin, so it leads for these names.
    """
    if model in (BusinessModel.BANK, BusinessModel.INSURER):
        return ("price_to_book", "trailing_pe", "forward_pe")
    if model is BusinessModel.CYCLICAL:
        return ("ev_revenue", "price_to_book", "ev_ebitda", "ev_ebit")
    if model is BusinessModel.UNPROFITABLE_GROWTH:
        return ("ev_revenue", "fcf_yield")
    return None


def _relative_inputs(inputs: CompanyValuationInputs) -> relative.RelativeInputs:
    forward_earnings = None
    if inputs.forward_eps and inputs.shares:
        forward_earnings = inputs.forward_eps * inputs.shares
    return relative.RelativeInputs(
        shares=inputs.shares or 0.0,
        net_debt=inputs.net_debt or 0.0,
        minority_interest=inputs.minority_interest or 0.0,
        preferred_equity=inputs.preferred_equity or 0.0,
        revenue=inputs.revenue,
        ebit=inputs.operating_income,
        ebitda=inputs.ebitda,
        net_income=inputs.net_income,
        free_cash_flow=inputs.free_cash_flow,
        forward_earnings=forward_earnings,
        equity=inputs.equity,
        growth=inputs.revenue_growth,
        margin=inputs.operating_margin,
        roe=inputs.roe,
    )


def value_company(
    inputs: CompanyValuationInputs, spec: ValuationSpec
) -> ValuationResult:
    """Run every applicable method and blend them into a range."""
    model = BusinessModel(inputs.business_model)
    weights = spec.weights_for(model)
    result = ValuationResult(
        price=inputs.price, business_model=model.value, spec_version=spec.version
    )
    caveats: list[str] = []

    rf = inputs.risk_free_rate or spec.dcf.discount.risk_free_fallback
    coe = cost_of_equity(
        inputs.beta or 1.0,
        risk_free=rf,
        spec=spec.dcf,
        market_cap=inputs.market_cap,
        extra_premium=(
            spec.dcf.discount.uncertainty_premium
            if model is BusinessModel.UNPROFITABLE_GROWTH
            else 0.0
        ),
    )

    dcf_results: dict = {}
    outcomes: list[MethodOutcome] = []

    for method, weight in weights.items():
        label = METHOD_LABELS.get(method, method.value)
        outcome = MethodOutcome(method=method.value, label=label, weight=weight)

        if method is ValuationMethod.DCF:
            dcf_results, scenarios, sens = _run_dcf(inputs, spec)
            base = dcf_results.get("base")
            if base is not None and math.isfinite(base.value_per_share):
                outcome.value_per_share = base.value_per_share
                outcome.used = True
                outcome.detail = (
                    f"{spec.dcf.forecast_years}-year projection discounted at "
                    f"{base.discount_rate:.1%}"
                )
                outcome.assumptions = base.assumptions
                caveats.extend(base.warnings)
                # Two ways the projection can be arithmetically fine and still
                # not be about this company. Both are assumption failures, not
                # findings, so the method is withdrawn rather than blended in
                # at half weight -- and withdrawn here, with a reason that says
                # what went wrong, rather than later by the price-ratio guard,
                # which can only say the answer looked odd.
                if base.contradicts_reported_fcf:
                    outcome.used = False
                    outcome.reason = (
                        "the projection contradicts the free cash flow the company actually "
                        "reported, so its reinvestment assumption cannot be trusted"
                    )
                elif base.enterprise_value_negative:
                    outcome.used = False
                    outcome.reason = (
                        "the projection values the operating business at or below zero, "
                        "leaving only the balance sheet"
                    )
            else:
                outcome.reason = (
                    "not enough of revenue, operating margin and share count to project cash flows"
                )

            # The scenario and sensitivity tables belong to a model that ran.
            # Publishing them for one the engine threw away put per-share values
            # like -$69,244,078.63 on McDonald's page, under the same Bear/Base/
            # Bull headings the range bar uses, and let a discarded model's
            # sensitivity cut the valuation's confidence by 40%.
            if outcome.used:
                result.scenarios = scenarios
                result.sensitivities = sens

        elif method is ValuationMethod.COMPARABLES:
            rel = relative.comparables_value(
                _relative_inputs(inputs),
                inputs.peer_multiples,
                spec=spec.comparables,
                peer_growth=inputs.peer_growth,
                peer_margin=inputs.peer_margin,
                preferred=_preferred_multiples(model),
            )
            if rel.value_per_share is not None and math.isfinite(rel.value_per_share):
                outcome.value_per_share = rel.value_per_share
                outcome.used = True
                outcome.detail = rel.detail
                outcome.value_low = rel.value_low
                outcome.value_high = rel.value_high
                outcome.assumptions = {
                    "multiple": rel.multiple_used,
                    "peer_multiple": rel.peer_multiple,
                    "peer_count": rel.peer_count,
                    "quality_adjustment": rel.adjustment,
                    "multiples_blended": [
                        {
                            "multiple": c["multiple"],
                            "label": c["label"],
                            "peer_multiple": c["peer_multiple"],
                            "peer_count": c["peer_count"],
                            "value_per_share": c["value_per_share"],
                        }
                        for c in rel.components
                    ],
                }
            else:
                outcome.reason = rel.detail or "no usable peer multiple"
                caveats.extend(rel.warnings)

        elif method is ValuationMethod.HISTORICAL:
            hist = relative.historical_value(
                _relative_inputs(inputs),
                inputs.own_multiple_history,
                spec=spec.historical,
                current_growth=inputs.revenue_growth,
                historical_growth=inputs.historical_growth,
                current_margin=inputs.operating_margin,
                historical_margin=inputs.historical_margin,
            )
            if hist.value_per_share is not None and math.isfinite(hist.value_per_share):
                outcome.value_per_share = hist.value_per_share
                outcome.used = True
                outcome.detail = hist.detail
                outcome.assumptions = {
                    "multiple": hist.multiple_used,
                    "historical_multiple": hist.historical_multiple,
                    "observations": hist.observations,
                    "regime_change": hist.regime_change,
                }
                if hist.regime_change:
                    # The method still contributes, at reduced weight, because a
                    # changed business is not a valueless comparison -- just a
                    # weaker one.
                    outcome.weight *= 0.5
                    outcome.reason = "down-weighted: the business has changed materially"
            else:
                outcome.reason = hist.detail or "not enough valuation history"
            caveats.extend(hist.warnings)

        elif method is ValuationMethod.RESIDUAL_INCOME:
            model_result = (
                industry.insurer_value(
                    book_equity=inputs.equity,
                    shares=inputs.shares,
                    roe=inputs.roe,
                    cost_of_equity=coe,
                    premiums_earned=inputs.premiums_earned,
                    policy_benefits=inputs.policy_benefits,
                )
                if model is BusinessModel.INSURER
                else industry.residual_income_value(
                    book_equity=inputs.equity,
                    shares=inputs.shares,
                    roe=inputs.roe,
                    cost_of_equity=coe,
                    intangibles=inputs.intangibles,
                    use_tangible_book=model is BusinessModel.BANK,
                )
            )
            outcome.value_per_share = model_result.value_per_share
            outcome.used = model_result.value_per_share is not None
            outcome.detail = model_result.detail
            outcome.assumptions = model_result.assumptions
            if not outcome.used:
                outcome.reason = model_result.detail
            caveats.extend(model_result.warnings)

        elif method is ValuationMethod.NAV_AFFO:
            model_result = industry.reit_value(
                net_income=inputs.net_income,
                depreciation=inputs.depreciation,
                capex=inputs.capex,
                shares=inputs.shares,
                net_debt=inputs.net_debt,
            )
            outcome.value_per_share = model_result.value_per_share
            outcome.used = model_result.value_per_share is not None
            outcome.detail = model_result.detail
            outcome.assumptions = model_result.assumptions
            if not outcome.used:
                outcome.reason = model_result.detail
            caveats.extend(model_result.warnings)

        elif method is ValuationMethod.NORMALIZED_EARNINGS:
            model_result = industry.normalized_earnings_value(
                revenue=inputs.revenue,
                margin_history=inputs.margin_history,
                current_margin=inputs.operating_margin,
                tax_rate=inputs.tax_rate or spec.dcf.tax_rate_default,
                shares=inputs.shares,
                net_debt=inputs.net_debt,
            )
            outcome.value_per_share = model_result.value_per_share
            outcome.used = model_result.value_per_share is not None
            outcome.detail = model_result.detail
            outcome.assumptions = model_result.assumptions
            if not outcome.used:
                outcome.reason = model_result.detail
            caveats.extend(model_result.warnings)

        # A method producing a value an order of magnitude away from the price is
        # almost certainly reading bad data rather than finding a spectacular
        # mispricing. The bounds are asymmetric: a genuinely overvalued stock can
        # trade at many times a defensible value, so the low side is looser than
        # the high side, which has no innocent explanation.
        if outcome.used and outcome.value_per_share is not None and inputs.price > EPS:
            ratio = outcome.value_per_share / inputs.price
            if not math.isfinite(ratio) or ratio <= 0 or ratio > 10.0 or ratio < 0.05:
                outcome.used = False
                outcome.reason = (
                    f"produced {outcome.value_per_share:,.0f} against a price of "
                    f"{inputs.price:,.2f}, which is implausible and suggests a data problem"
                )

        outcomes.append(outcome)

    result.methods = outcomes
    used = [o for o in outcomes if o.used and o.value_per_share is not None]

    # The price-ratio guard above can withdraw the DCF *after* its scenario and
    # sensitivity tables were attached, so the check has to be repeated once
    # every guard has had its say. Publishing them anyway is how a rejected
    # projection still put a -$69,244,078.63 base case on a stock page.
    if not any(o.method == ValuationMethod.DCF.value and o.used for o in outcomes):
        result.scenarios = []
        result.sensitivities = []

    if not used:
        result.caveats = _dedupe(
            caveats + ["no valuation method could run on the available data"]
        )
        result.confidence = 0.0
        result.confidence_label = confidence_label(0.0)
        return result

    total_weight = sum(o.weight for o in used)
    base_value = sum(o.value_per_share * o.weight for o in used) / total_weight

    # --- disagreement between methods --------------------------------------
    # With one method there is no disagreement to measure, and recording zero
    # would be indistinguishable from several methods agreeing perfectly --
    # which is the opposite situation. None means "unknown", and every consumer
    # has to decide what to do about that rather than reading a confident zero.
    values = np.array([o.value_per_share for o in used], dtype=float)
    dispersion = (
        float(np.std(values) / abs(base_value))
        if len(values) > 1 and abs(base_value) > EPS
        else None
    )
    result.method_dispersion = dispersion

    # --- bear and bull ------------------------------------------------------
    # The scenarios are only usable when the discounted cash-flow model actually
    # contributed to the blend. When it was rejected as implausible, its
    # scenarios are equally implausible, and scaling the blend by the ratio
    # between them turns a rejected $1.80 base case into a $1,900 bull case.
    dcf_used = any(o.method == ValuationMethod.DCF.value and o.used for o in used)
    bear_dcf = dcf_results.get("bear") if dcf_used else None
    bull_dcf = dcf_results.get("bull") if dcf_used else None
    dcf_base = dcf_results.get("base") if dcf_used else None

    bear = bull = None
    if bear_dcf is not None and bull_dcf is not None and dcf_base is not None:
        anchor = dcf_base.value_per_share
        if abs(anchor) > EPS:
            low_ratio = bear_dcf.value_per_share / anchor
            high_ratio = bull_dcf.value_per_share / anchor
            # A scenario set spanning more than a factor of a few is describing
            # model instability, not a genuine range of outcomes.
            if 0.2 <= low_ratio <= 1.0 <= high_ratio <= 3.0:
                bear = base_value * low_ratio
                bull = base_value * high_ratio
            else:
                caveats.append(
                    "the scenario range was too wide to be meaningful and has been replaced "
                    "by a range derived from how much the valuation methods disagree"
                )

    # A single method that can express its own uncertainty should be believed
    # over any fixed percentage. For comparables that uncertainty is the spread
    # of the peer group, which is a real measurement of how much defensible
    # answers differ, rather than a band chosen to look like a range.
    if (bear is None or bull is None) and len(used) == 1:
        lone = used[0]
        if (
            lone.value_low is not None
            and lone.value_high is not None
            and math.isfinite(lone.value_low)
            and math.isfinite(lone.value_high)
            and lone.value_low < lone.value_high
        ):
            bear, bull = lone.value_low, lone.value_high

    if bear is None or bull is None:
        # Nothing measured the uncertainty, so it has to be asserted. The floor
        # is wider when only one method ran: a lone estimate is less certain
        # than a blend, and the range is the only place that can show.
        floor = 0.20 if len(used) > 1 else 0.35
        spread = min(max(dispersion or 0.0, floor), 0.60)
        bear = base_value * (1.0 - spread)
        bull = base_value * (1.0 + spread)
        caveats.append(
            "the fair-value range is a generic band, not a measured one: no method could "
            "express its own uncertainty"
        )

    # Methods disagreeing widens the range: their spread is real uncertainty, so
    # the range must at least span every method that ran. It should not extend
    # *beyond* them, though -- padding an already-extreme method by a further
    # fifth compounds one aggressive peer multiple into a fair value nobody
    # proposed.
    if len(values) > 1:
        bear = min(bear, float(values.min()))
        bull = max(bull, float(values.max()))

    # The base case must sit inside its own range. If widening for method
    # disagreement has pushed a bound past it, the bound moves, not the base:
    # the base is the weighted estimate and the bounds express uncertainty
    # around it.
    bear, bull = min(bear, bull), max(bear, bull)
    bear = min(bear, base_value)
    bull = max(bull, base_value)

    result.bear, result.base, result.bull = bear, base_value, bull
    result.range_low, result.range_high = bear, bull

    # The headline estimate is the weighted blend, not the centre of the range.
    #
    # These are not the same number and the difference is not noise. Value
    # compounds, so a bull case sits further above the base than the bear case
    # sits below it, and the arithmetic midpoint of an asymmetric range is
    # therefore biased upward: it exceeded the blend for 402 of 495 companies,
    # by a mean of 10.3%. Reporting it flipped the sign of the upside for 42
    # names and made 89 verdicts one bucket more bullish than the methods
    # supported. `midpoint` is kept as the geometric centre of the range, which
    # is what it always was, but nothing is decided on it.
    result.midpoint = base_value
    result.range_midpoint = (bear + bull) / 2.0
    result.upside = (base_value / inputs.price - 1.0) if inputs.price > EPS else None

    # --- confidence ---------------------------------------------------------
    confidence, confidence_notes = _valuation_confidence(
        inputs, used, dispersion, result, spec, dcf_results
    )
    result.confidence = confidence
    result.confidence_label = confidence_label(confidence)
    caveats.extend(confidence_notes)

    # --- classification -----------------------------------------------------
    result.classification = classify(
        price=inputs.price,
        midpoint=result.midpoint,
        low=bear,
        high=bull,
        confidence=confidence,
        spec=spec.classification,
    )
    result.classification_label = CLASS_LABELS[result.classification]

    result.assumptions = {
        "cost_of_equity": coe,
        "risk_free_rate": rf,
        "growth_estimate": _growth_estimate(inputs),
        "method_weights": {o.method: o.weight for o in used},
        "total_weight_used": total_weight,
        "business_model": model.value,
    }
    result.caveats = _dedupe(caveats)
    return result


def classify(
    *,
    price: float,
    midpoint: float | None,
    low: float | None,
    high: float | None,
    confidence: float,
    spec: ClassificationSpec,
) -> str:
    """Where the price sits relative to the range, scaled by the range's width.

    A stock 15% below a tight range is a stronger statement than one 15% below a
    range spanning a factor of three, so the thresholds widen with the range. And
    below a confidence floor we decline to classify at all rather than issue a
    verdict we do not believe.
    """
    if midpoint is None or price <= EPS or not math.isfinite(midpoint):
        return ValuationClass.INSUFFICIENT_CONFIDENCE
    if confidence < spec.min_confidence:
        return ValuationClass.INSUFFICIENT_CONFIDENCE

    upside = midpoint / price - 1.0

    width = 1.0
    if low is not None and high is not None and abs(midpoint) > EPS:
        observed = (high - low) / abs(midpoint)
        width = min(max(observed / spec.reference_width, 0.5), spec.max_width_multiplier)

    if upside >= spec.significantly_undervalued * width:
        return ValuationClass.SIGNIFICANTLY_UNDERVALUED
    if upside >= spec.undervalued * width:
        return ValuationClass.UNDERVALUED
    if upside <= spec.significantly_overvalued * width:
        return ValuationClass.SIGNIFICANTLY_OVERVALUED
    if upside <= spec.overvalued * width:
        return ValuationClass.OVERVALUED
    return ValuationClass.FAIRLY_VALUED


def _valuation_confidence(
    inputs: CompanyValuationInputs,
    used: list[MethodOutcome],
    dispersion: float | None,
    result: ValuationResult,
    spec: ValuationSpec,
    dcf_results: dict,
) -> tuple[float, list[str]]:
    """How much to believe this estimate, in [0, 1].

    Deliberately pessimistic. A valuation that rests on one method, stale data
    and a terminal value carrying most of the weight should not present itself
    as equivalent to one where three methods agree on fresh numbers.
    """
    notes: list[str] = []
    score = 1.0

    # Agreement between methods.
    if len(used) == 1:
        score *= 0.70
        notes.append(
            f"only one valuation method could run ({used[0].label}), so there is no "
            "cross-check on the estimate"
        )
    elif dispersion is None:
        # Several methods ran but the blend came out at nothing, so their
        # disagreement cannot be expressed as a fraction of it.
        score *= 0.70
        notes.append("the methods could not be compared against a meaningful base case")
    elif dispersion > 0.50:
        score *= 0.55
        notes.append(
            f"the methods disagree sharply (spread of {dispersion:.0%} around the base case)"
        )
    elif dispersion > 0.25:
        score *= 0.80
        notes.append(f"the methods disagree moderately (spread of {dispersion:.0%})")

    # The width of the range is itself evidence. An estimate spanning a factor
    # of three is not a fair value with wide error bars; it is an admission that
    # the methods cannot pin the company down, and it should not be able to
    # present itself as a confident call just because the methods that ran had
    # complete data.
    if (
        result.range_low is not None
        and result.range_high is not None
        and result.midpoint
        and abs(result.midpoint) > EPS
    ):
        width = (result.range_high - result.range_low) / abs(result.midpoint)
        if width > 1.5:
            score *= 0.50
            notes.append(
                f"the fair-value range spans {width:.1f} times its own midpoint, which is too "
                "wide to support a precise view"
            )
        elif width > 1.0:
            score *= 0.70
            notes.append(
                f"the fair-value range is wide (spanning {width:.0%} of the midpoint)"
            )
        elif width > 0.7:
            score *= 0.88

    # Sensitivity to assumptions.
    if result.sensitivities:
        worst = max((s.get("swing_pct") or 0.0) for s in result.sensitivities)
        if worst > 1.0:
            score *= 0.60
            notes.append(
                f"a single assumption moves fair value by {worst:.0%}, so the estimate is "
                "highly assumption-dependent"
            )
        elif worst > 0.50:
            score *= 0.80

    # Terminal-value dependence. An undefined terminal share is not a missing
    # reading to be skipped over -- it is the signature of a projection that
    # discounted to nothing, and the threshold tests below would all pass it
    # silently if it were allowed through as a number.
    base_dcf = dcf_results.get("base")
    if base_dcf is not None:
        if not math.isfinite(base_dcf.terminal_share):
            score *= 0.60
            notes.append(
                "the cash-flow projection was degenerate -- it valued the operating business "
                "at or below zero -- so it could not be used"
            )
        elif base_dcf.terminal_share > 0.85:
            score *= 0.70
            notes.append(
                f"{base_dcf.terminal_share:.0%} of the discounted value sits beyond the "
                "forecast horizon"
            )
        elif base_dcf.terminal_share > spec.dcf.terminal_share_warning:
            score *= 0.88

    # Data completeness and freshness.
    if inputs.metric_coverage is not None:
        if inputs.metric_coverage < 0.5:
            score *= 0.65
            notes.append(
                f"only {inputs.metric_coverage:.0%} of the financial inputs were available"
            )
        elif inputs.metric_coverage < 0.75:
            score *= 0.85

    if inputs.data_age_days is not None and inputs.data_age_days > 130:
        score *= 0.80
        notes.append(
            f"the most recent financial filing is {inputs.data_age_days:.0f} days old"
        )

    # Analyst coverage: a forward estimate from two analysts is one opinion.
    if inputs.n_analysts is not None and inputs.n_analysts < 5:
        score *= 0.90

    # Business predictability.
    if inputs.business_model == BusinessModel.UNPROFITABLE_GROWTH:
        score *= 0.65
        notes.append(
            "the company is not yet profitable, so its value depends almost entirely on "
            "assumptions about a future that has not happened"
        )
    elif inputs.business_model == BusinessModel.CYCLICAL:
        score *= 0.85
        notes.append(
            "earnings are cyclical, so any point-in-time valuation depends on where in the "
            "cycle the business currently sits"
        )

    return max(min(score, 1.0), 0.0), notes


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))
