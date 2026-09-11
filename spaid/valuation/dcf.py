"""Discounted cash-flow valuation.

The model values the whole firm and then subtracts what is owed, which is the
only way to compare a debt-funded company with an equity-funded one on the same
terms:

    revenue grows and fades -> operating margin converges -> tax -> NOPAT
    -> subtract the reinvestment needed to fund that growth -> free cash flow
    -> discount at the cost of capital -> add a terminal value
    -> subtract net debt, minority interest and preferred -> divide by shares

Choices worth stating because they move the answer:

* **Share-based compensation is a cost, and is already charged.** The cash-flow
  statement adds it back because no cash left the building, but something of
  value did leave the existing owners, so it should not be added back here.
  Note *already*: the projection starts from GAAP operating income
  (`OperatingIncomeLoss`), which under ASC 718 has expensed stock compensation
  before the model sees it. Subtracting it a second time -- which this model did
  until it was caught by comparing its first projected year against reported
  cash flow -- charges the companies that use it most for it twice, and turned
  CrowdStrike's 20.6% pre-compensation margin into an effective -25.0%.
* **Growth must be paid for.** Reinvestment is tied to revenue growth through a
  sales-to-capital ratio, so a model cannot assume a company grows at 15% a year
  while returning all its cash. That assumption is the most common way a
  discounted cash-flow model produces a number twice the market price.
* **Terminal growth is capped below the long-run growth rate of the economy.**
  A company growing faster than the economy forever eventually becomes the
  economy.
* **The terminal value's share of the total is reported.** When four fifths of
  the value sits beyond the forecast horizon, the estimate is an opinion about
  the terminal multiple wearing a spreadsheet, and the confidence system needs
  to know that.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from spaid.config.valuation import DcfSpec, ScenarioAssumptions

log = logging.getLogger(__name__)

EPS = 1e-9


@dataclass(frozen=True)
class DcfInputs:
    """Everything the model needs about one company, in absolute currency units."""

    revenue: float
    operating_margin: float
    tax_rate: float
    shares: float
    net_debt: float
    # Claims that rank ahead of, or alongside, the common equity.
    minority_interest: float = 0.0
    preferred_equity: float = 0.0
    share_based_comp: float = 0.0
    # Starting revenue growth: the blend of what the company has done and what
    # analysts expect, decided by the caller.
    growth_rate: float = 0.05
    # How much revenue a dollar of invested capital supports. Governs how
    # expensive growth is for this particular business.
    sales_to_capital: float = 2.0
    # Return on invested capital, used to set the terminal reinvestment rate.
    roic: float | None = None
    beta: float = 1.0
    market_cap: float | None = None
    risk_free_rate: float | None = None
    # What the company actually reported as free cash flow over the trailing
    # year. Not used to value anything -- used to check that the projection's
    # first year resembles the company it claims to be projecting.
    reported_free_cash_flow: float | None = None
    # Consensus revenue *levels* for the next years, nearest first, and the
    # ratio of the analyst low and high to the consensus at the far end of that
    # path. Levels rather than growth rates because the trailing twelve months
    # already overlaps the current fiscal year, so a growth rate computed
    # against it is not the growth rate anyone is forecasting.
    consensus_revenue: tuple[float, ...] = ()
    consensus_low_ratio: float = 1.0
    consensus_high_ratio: float = 1.0
    # The operating margin implied by consensus earnings in each of those
    # years, aligned with `consensus_revenue`. Consensus earnings cannot anchor
    # the model directly -- they are a margin assumption and a revenue
    # assumption bundled together -- but once revenue is known the margin falls
    # out of them, year by year.
    consensus_margins: tuple[float | None, ...] = ()
    n_revenue_analysts: int | None = None

    def validate(self) -> list[str]:
        """Reasons this company cannot be valued this way, for the caller to act on."""
        problems: list[str] = []
        if self.revenue is None or self.revenue <= 0:
            problems.append("no positive revenue")
        if self.shares is None or self.shares <= 0:
            problems.append("no share count")
        if self.operating_margin is None or not math.isfinite(self.operating_margin):
            problems.append("no operating margin")
        return problems


@dataclass
class YearProjection:
    year: int
    revenue: float
    growth: float
    margin: float
    ebit: float
    nopat: float
    reinvestment: float
    free_cash_flow: float
    discount_factor: float
    present_value: float


@dataclass
class DcfResult:
    value_per_share: float
    enterprise_value: float
    equity_value: float
    discount_rate: float
    terminal_growth: float
    terminal_value: float
    terminal_value_pv: float
    terminal_share: float
    pv_explicit: float
    years: list[YearProjection] = field(default_factory=list)
    assumptions: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # The projection's first year disagrees in sign, and materially in size,
    # with the cash flow the company actually reported. Set when the
    # reinvestment assumption -- not the business -- is driving the result.
    contradicts_reported_fcf: bool = False
    # The operating business values at or below zero, so whatever equity value
    # survives is the balance sheet rather than the company.
    enterprise_value_negative: bool = False


def cost_of_equity(
    beta: float,
    *,
    risk_free: float,
    spec: DcfSpec,
    market_cap: float | None = None,
    extra_premium: float = 0.0,
) -> float:
    """Capital-asset-pricing cost of equity, with the beta estimate kept sane.

    A rolling regression beta on a volatile small company can print 2.8 or 0.1,
    neither of which is a defensible forward-looking estimate, so it is bounded.
    A size premium is added for smaller companies, where the historical evidence
    for a higher required return is strongest.
    """
    d = spec.discount
    b = min(max(beta if beta and math.isfinite(beta) else 1.0, d.beta_floor), d.beta_cap)
    rate = risk_free + b * d.equity_risk_premium + extra_premium
    if market_cap is not None and market_cap < d.size_premium_threshold:
        rate += d.size_premium_small
    return min(max(rate, d.min_rate), d.max_rate)


def _fade(start: float, end: float, t: int, n: int, strength: float) -> float:
    """Move from `start` to `end` over `n` years, front- or back-loaded.

    `strength` above one fades *faster* -- the pessimistic reading, in which the
    good years end sooner. Below one the starting rate persists longer.

    The exponent is the reciprocal of the strength, which is the part that is
    easy to get backwards: raising progress to a power below one makes it
    *larger* and therefore moves the value toward the endpoint sooner. Inverting
    it once here means a bear case with strength 1.35 genuinely fades faster
    than a bull case with 0.75, rather than the reverse -- a sign error that
    produced bull cases worth less than their base case for fast-growing
    companies.
    """
    if n <= 0:
        return end
    exponent = 1.0 / max(strength, EPS)
    progress = min(max(t / n, 0.0), 1.0) ** exponent
    return start + (end - start) * progress


def project(
    inputs: DcfInputs,
    scenario: ScenarioAssumptions,
    spec: DcfSpec,
    *,
    risk_free: float | None = None,
) -> DcfResult:
    """Run one scenario and return the full year-by-year projection."""
    problems = inputs.validate()
    if problems:
        raise ValueError(f"cannot run a discounted cash-flow model: {'; '.join(problems)}")

    warnings: list[str] = []
    rf = (
        risk_free
        if risk_free is not None
        else (inputs.risk_free_rate if inputs.risk_free_rate is not None else spec.discount.risk_free_fallback)
    )

    rate = cost_of_equity(
        inputs.beta,
        risk_free=rf,
        spec=spec,
        market_cap=inputs.market_cap,
        extra_premium=scenario.discount_rate_adjustment,
    )

    terminal_growth = min(
        spec.terminal_growth + scenario.terminal_growth_adjustment,
        spec.terminal_growth_cap,
        # Terminal growth above the discount rate makes the perpetuity formula
        # produce a negative or infinite value, which is a mathematical artefact
        # rather than a valuation.
        rate - 0.01,
    )

    start_growth = (inputs.growth_rate or 0.0) * scenario.growth_multiplier
    # Nobody grows at 60% a year for a decade; cap the starting rate so one
    # spectacular year cannot drive the whole projection.
    if start_growth > 0.40:
        warnings.append(
            f"starting growth of {start_growth:.0%} capped at 40% for the projection"
        )
        start_growth = 0.40
    start_growth = max(start_growth, -0.25)

    # --- consensus anchoring ------------------------------------------------
    # Where enough analysts cover the company, the first years of the
    # projection are theirs rather than the model's. Fading a trailing growth
    # rate from a trailing margin cannot produce a company whose earnings are
    # about to triple, and for AMD it projected $6.04 of earnings per share
    # against a consensus of $15.61 -- a projection of a different company,
    # which then set the fair value.
    consensus_path: list[float] = []
    use_consensus = (
        bool(inputs.consensus_revenue)
        and inputs.revenue > 0
        and (inputs.n_revenue_analysts or 0) >= spec.consensus_min_analysts
        and all(v and v > 0 for v in inputs.consensus_revenue)
    )
    if use_consensus:
        side = getattr(scenario, "consensus_side", "mid")
        far_ratio = (
            inputs.consensus_low_ratio
            if side == "low"
            else inputs.consensus_high_ratio
            if side == "high"
            else 1.0
        )
        # The analyst spread widens with distance, so the near year is barely
        # scaled and the far year takes the full low or high.
        horizon = len(inputs.consensus_revenue)
        for i, level in enumerate(inputs.consensus_revenue, start=1):
            ratio = 1.0 + (far_ratio - 1.0) * (i / horizon)
            consensus_path.append(level * ratio)
        # A path that implies the company shrinks below where it already is has
        # nothing to contribute over the model's own fade.
        if consensus_path[-1] <= inputs.revenue:
            consensus_path = []

    # Consensus margins, capped: a margin far above today's is an expectation
    # worth taking seriously, but one implied by a broken earnings estimate is
    # not, and from a single number the two look alike.
    margin_path: list[float] = []
    if consensus_path:
        ceiling = inputs.operating_margin + spec.consensus_margin_cap_pp
        for value in inputs.consensus_margins[: len(consensus_path)]:
            if value is None or not math.isfinite(value):
                margin_path = []
                break
            capped = min(value, ceiling)
            if capped < value:
                warnings.append(
                    f"consensus earnings imply a {value:.0%} operating margin against "
                    f"{inputs.operating_margin:.0%} today, which is too far to take at face "
                    f"value; capped at {capped:.0%}"
                )
            margin_path.append(capped)

    target_margin = inputs.operating_margin + scenario.margin_adjustment
    if margin_path:
        # Beyond the consensus years the margin holds where consensus left it,
        # nudged by the scenario. Reverting to the trailing margin instead would
        # undo in year three what years one and two were anchored to.
        target_margin = margin_path[-1] + scenario.margin_adjustment
    sales_to_capital = min(
        max(inputs.sales_to_capital or spec.sales_to_capital_default, spec.sales_to_capital_floor),
        spec.sales_to_capital_cap,
    )
    tax = min(max(inputs.tax_rate or spec.tax_rate_default, spec.tax_rate_floor), spec.tax_rate_cap)

    years: list[YearProjection] = []
    revenue = inputs.revenue
    pv_explicit = 0.0
    # Share-based compensation is scaled as a share of revenue so it tracks the
    # business rather than staying frozen at today's dollar amount. It is only
    # ever *added back*: the starting margin is GAAP, so the cost is already in
    # the projection, and treating it as an expense means leaving it alone.
    sbc_ratio = (
        (inputs.share_based_comp / inputs.revenue)
        if not spec.treat_sbc_as_expense and inputs.revenue > 0
        else 0.0
    )

    n = spec.forecast_years
    horizon = len(consensus_path)
    # After the consensus runs out, growth fades from where consensus left it
    # rather than from the trailing rate, so the two halves of the projection
    # join up. The cap still applies: a consensus year-on-year rate above it is
    # a forecast, not a decade-long trend.
    post_growth = start_growth
    if horizon:
        previous = consensus_path[-2] if horizon > 1 else inputs.revenue
        if previous > 0:
            post_growth = min(consensus_path[-1] / previous - 1.0, 0.40)

    for t in range(1, n + 1):
        if t <= horizon:
            growth = consensus_path[t - 1] / revenue - 1.0
        else:
            growth = _fade(
                post_growth, terminal_growth, t - horizon, n - horizon, scenario.fade_strength
            )

        if margin_path and t <= len(margin_path):
            margin = margin_path[t - 1]
        elif margin_path:
            # Already at the consensus margin; only the scenario nudge remains.
            margin = _fade(
                margin_path[-1],
                target_margin,
                t - len(margin_path),
                min(5, n - len(margin_path)),
                1.0,
            )
        else:
            margin = _fade(inputs.operating_margin, target_margin, t, min(5, n), 1.0)

        previous_revenue = revenue
        revenue = revenue * (1.0 + growth)
        ebit = revenue * margin
        nopat = ebit * (1.0 - tax)
        # Growth has to be funded. A company adding revenue needs working
        # capital and capacity to support it.
        reinvestment = max(revenue - previous_revenue, 0.0) / sales_to_capital
        # Added back net of tax, because the expense was deductible.
        sbc_addback = revenue * sbc_ratio * (1.0 - tax)
        fcf = nopat + sbc_addback - reinvestment

        discount_factor = 1.0 / ((1.0 + rate) ** t)
        pv = fcf * discount_factor
        pv_explicit += pv

        years.append(
            YearProjection(
                year=t,
                revenue=revenue,
                growth=growth,
                margin=margin,
                ebit=ebit,
                nopat=nopat,
                reinvestment=reinvestment,
                free_cash_flow=fcf,
                discount_factor=discount_factor,
                present_value=pv,
            )
        )

    # --- terminal value ----------------------------------------------------
    # In perpetuity a company must still fund its growth, and the rate at which
    # it can do so is its return on capital. Assuming otherwise is assuming free
    # growth, which is where most inflated valuations come from.
    terminal_roic = rate + 0.01
    if inputs.roic and math.isfinite(inputs.roic) and inputs.roic > rate:
        terminal_roic = min(inputs.roic, rate + 0.05)
    terminal_reinvestment_rate = (
        min(max(terminal_growth / terminal_roic, 0.0), 0.90) if terminal_roic > EPS else 0.0
    )

    final = years[-1]
    terminal_nopat = final.nopat * (1.0 + terminal_growth)
    terminal_fcf = terminal_nopat * (1.0 - terminal_reinvestment_rate)
    spread = rate - terminal_growth
    if spread <= EPS:
        raise ValueError("discount rate does not exceed terminal growth")

    terminal_value = terminal_fcf / spread
    terminal_pv = terminal_value * final.discount_factor

    enterprise_value = pv_explicit + terminal_pv
    equity_value = (
        enterprise_value
        - inputs.net_debt
        - (inputs.minority_interest or 0.0)
        - (inputs.preferred_equity or 0.0)
    )
    value_per_share = equity_value / inputs.shares

    # The terminal value's share of the total is only a share when the total is
    # a positive number. Against a negative total the ratio stops describing
    # anything -- it printed 434%, -9753% and 322% across the three scenarios of
    # a company whose enterprise value was near zero -- and every downstream
    # threshold test on it silently passes. Undefined is the honest answer
    # there, and the caller is told why separately.
    #
    # A share above one is a different case and is left alone: a company that
    # consumes cash for a few years and earns it back later genuinely has more
    # than all of its value beyond the forecast, and saying so is useful.
    total = pv_explicit + terminal_pv
    terminal_share = terminal_pv / total if total > EPS else float("nan")
    if math.isfinite(terminal_share) and terminal_share > spec.terminal_share_warning:
        warnings.append(
            f"{terminal_share:.0%} of the value sits in the terminal period, so the estimate "
            "rests mostly on assumptions about the distant future"
        )

    enterprise_value_negative = enterprise_value <= 0
    if enterprise_value_negative:
        warnings.append(
            "the operating business discounts to zero or less, so the whole estimate is the "
            "balance sheet rather than the company"
        )
    elif pv_explicit < 0:
        warnings.append(
            "every year of the forecast consumes cash, so the entire value rests on the "
            "terminal period"
        )
    if equity_value < 0:
        warnings.append("debt exceeds the computed enterprise value, so equity values below zero")

    # A projection whose first year contradicts the cash flow the company just
    # reported is describing an investment cycle that is not happening. This is
    # the cheapest available check that the reinvestment assumption is sane,
    # and it is the one that catches an acquisition-inflated capital base being
    # used as the marginal cost of growth.
    contradicts = False
    reported = inputs.reported_free_cash_flow
    first_year_fcf = years[0].free_cash_flow
    if (
        reported is not None
        and math.isfinite(reported)
        and reported > 0
        and first_year_fcf < 0
        and inputs.revenue > 0
        and (reported - first_year_fcf) / inputs.revenue > spec.fcf_contradiction_threshold
    ):
        contradicts = True
        warnings.append(
            f"the projection turns a reported {reported / 1e9:,.1f}bn of free cash flow into "
            f"{first_year_fcf / 1e9:,.1f}bn in its first year, a swing of "
            f"{(reported - first_year_fcf) / inputs.revenue:.0%} of revenue, so the "
            "reinvestment assumption rather than the business is driving the result"
        )

    return DcfResult(
        value_per_share=value_per_share,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        discount_rate=rate,
        terminal_growth=terminal_growth,
        terminal_value=terminal_value,
        terminal_value_pv=terminal_pv,
        terminal_share=terminal_share,
        pv_explicit=pv_explicit,
        contradicts_reported_fcf=contradicts,
        enterprise_value_negative=enterprise_value_negative,
        years=years,
        assumptions={
            "scenario": scenario.label,
            "anchored_to_consensus": bool(consensus_path),
            "consensus_years": len(consensus_path),
            "consensus_margins": margin_path or None,
            "starting_revenue": inputs.revenue,
            "starting_growth": start_growth,
            "starting_margin": inputs.operating_margin,
            "target_margin": target_margin,
            "tax_rate": tax,
            "sales_to_capital": sales_to_capital,
            "discount_rate": rate,
            "risk_free_rate": rf,
            "beta": inputs.beta,
            "terminal_growth": terminal_growth,
            "terminal_roic": terminal_roic,
            "terminal_reinvestment_rate": terminal_reinvestment_rate,
            "share_based_comp_ratio": sbc_ratio,
            "forecast_years": n,
            "shares": inputs.shares,
            "net_debt": inputs.net_debt,
        },
        warnings=warnings,
    )


def run_scenarios(
    inputs: DcfInputs, spec: DcfSpec, *, risk_free: float | None = None
) -> dict[str, DcfResult]:
    """Bear, base and bull under the spec's declared assumption sets."""
    out: dict[str, DcfResult] = {}
    for scenario in spec.scenarios:
        try:
            out[scenario.label] = project(inputs, scenario, spec, risk_free=risk_free)
        except ValueError as exc:
            log.debug("scenario %s failed: %s", scenario.label, exc)
    return out


def sensitivity(
    inputs: DcfInputs,
    spec: DcfSpec,
    *,
    risk_free: float | None = None,
    base_label: str = "base",
) -> list[dict]:
    """How much the base-case value moves when one assumption changes.

    This is the honest answer to "how sure are you": if a one-point change in
    the discount rate moves fair value by 40%, the precision implied by a single
    number was never real, and the confidence score should say so.
    """
    base_scenario = next(
        (s for s in spec.scenarios if s.label == base_label), spec.scenarios[0]
    )
    try:
        base = project(inputs, base_scenario, spec, risk_free=risk_free)
    except ValueError:
        return []

    rf = risk_free if risk_free is not None else spec.discount.risk_free_fallback
    out: list[dict] = []

    def record(label: str, description: str, low_inputs, high_inputs, low_rf=None, high_rf=None):
        try:
            low = project(low_inputs, base_scenario, spec, risk_free=low_rf or rf)
            high = project(high_inputs, base_scenario, spec, risk_free=high_rf or rf)
        except ValueError:
            return
        out.append(
            {
                "assumption": label,
                "description": description,
                "low_value": low.value_per_share,
                "high_value": high.value_per_share,
                "base_value": base.value_per_share,
                "swing_pct": (
                    abs(high.value_per_share - low.value_per_share)
                    / abs(base.value_per_share)
                    if abs(base.value_per_share) > EPS
                    else None
                ),
            }
        )

    from dataclasses import replace

    # Perturb what the projection actually reads.
    #
    # Under consensus anchoring `growth_rate` and `operating_margin` are both
    # overridden -- revenue comes from the analyst path and the margin from what
    # consensus earnings imply on it -- so nudging them moved nothing and the
    # table reported a 0.0% swing on the two assumptions that drive the model.
    # That is not "insensitive", it is "not measured", and it silently inflated
    # confidence because the worst observed swing was understated.
    anchored = bool(
        inputs.consensus_revenue
        and (inputs.n_revenue_analysts or 0) >= spec.consensus_min_analysts
    )

    def _scaled_revenue(factor: float) -> DcfInputs:
        return replace(
            inputs,
            consensus_revenue=tuple(v * factor for v in inputs.consensus_revenue),
            growth_rate=(inputs.growth_rate or 0) + (factor - 1.0),
        )

    def _shifted_margin(delta: float) -> DcfInputs:
        return replace(
            inputs,
            consensus_margins=tuple(
                None if m is None else m + delta for m in inputs.consensus_margins
            ),
            operating_margin=inputs.operating_margin + delta,
        )

    if anchored:
        record(
            "Revenue growth",
            "Consensus revenue path 3% either side",
            _scaled_revenue(0.97),
            _scaled_revenue(1.03),
        )
        record(
            "Operating margin",
            "Operating margin 2 percentage points either side",
            _shifted_margin(-0.02),
            _shifted_margin(+0.02),
        )
    else:
        record(
            "Revenue growth",
            "Starting growth rate 3 percentage points either side",
            replace(inputs, growth_rate=(inputs.growth_rate or 0) - 0.03),
            replace(inputs, growth_rate=(inputs.growth_rate or 0) + 0.03),
        )
        record(
            "Operating margin",
            "Operating margin 2 percentage points either side",
            replace(inputs, operating_margin=inputs.operating_margin - 0.02),
            replace(inputs, operating_margin=inputs.operating_margin + 0.02),
        )
    record(
        "Discount rate",
        "Cost of capital 1 percentage point either side",
        inputs, inputs, low_rf=rf + 0.01, high_rf=rf - 0.01,
    )
    record(
        "Reinvestment intensity",
        "Sales-to-capital ratio 25% either side",
        replace(inputs, sales_to_capital=(inputs.sales_to_capital or 2.0) * 0.75),
        replace(inputs, sales_to_capital=(inputs.sales_to_capital or 2.0) * 1.25),
    )
    return sorted(out, key=lambda d: -(d["swing_pct"] or 0.0))
