"""Tests for the discounted cash-flow model.

Most of these are economic sanity properties rather than golden numbers: a
valuation model is right when it responds correctly to changes in its inputs,
and the absolute level is only meaningful relative to the assumptions.

The perpetuity identity test is the exception. It is the one place the model can
be checked against closed-form arithmetic, so it is checked exactly.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from spaid.config.valuation import DcfSpec, ScenarioAssumptions
from spaid.valuation.dcf import (
    DcfInputs,
    cost_of_equity,
    project,
    run_scenarios,
    sensitivity,
)

SPEC = DcfSpec()
BASE_SCENARIO = next(s for s in SPEC.scenarios if s.label == "base")


def company(**kw) -> DcfInputs:
    """A profitable mid-cap with modest growth."""
    defaults = dict(
        revenue=10_000_000_000.0,
        operating_margin=0.20,
        tax_rate=0.21,
        shares=1_000_000_000.0,
        net_debt=2_000_000_000.0,
        share_based_comp=100_000_000.0,
        growth_rate=0.06,
        sales_to_capital=2.0,
        roic=0.15,
        beta=1.0,
        market_cap=50_000_000_000.0,
    )
    defaults.update(kw)
    return DcfInputs(**defaults)


class TestCostOfEquity:
    def test_capm_arithmetic(self):
        rate = cost_of_equity(1.0, risk_free=0.04, spec=SPEC, market_cap=1e12)
        assert rate == pytest.approx(0.04 + 1.0 * SPEC.discount.equity_risk_premium)

    def test_beta_is_bounded_both_ways(self):
        """A regression beta of 4 is an artefact, not a forward-looking estimate."""
        wild = cost_of_equity(4.0, risk_free=0.04, spec=SPEC, market_cap=1e12)
        capped = cost_of_equity(SPEC.discount.beta_cap, risk_free=0.04, spec=SPEC, market_cap=1e12)
        assert wild == pytest.approx(capped)

        tiny = cost_of_equity(0.01, risk_free=0.04, spec=SPEC, market_cap=1e12)
        floored = cost_of_equity(SPEC.discount.beta_floor, risk_free=0.04, spec=SPEC, market_cap=1e12)
        assert tiny == pytest.approx(floored)

    def test_small_companies_carry_a_size_premium(self):
        small = cost_of_equity(1.0, risk_free=0.04, spec=SPEC, market_cap=1e9)
        large = cost_of_equity(1.0, risk_free=0.04, spec=SPEC, market_cap=1e12)
        assert small > large
        assert small - large == pytest.approx(SPEC.discount.size_premium_small)

    def test_rate_never_leaves_its_bounds(self):
        assert cost_of_equity(1.0, risk_free=-0.05, spec=SPEC) >= SPEC.discount.min_rate
        assert cost_of_equity(2.0, risk_free=0.30, spec=SPEC) <= SPEC.discount.max_rate

    def test_missing_beta_defaults_to_the_market(self):
        assert cost_of_equity(float("nan"), risk_free=0.04, spec=SPEC, market_cap=1e12) == pytest.approx(
            cost_of_equity(1.0, risk_free=0.04, spec=SPEC, market_cap=1e12)
        )


class TestProjection:
    def test_produces_a_positive_value_for_a_profitable_company(self):
        result = project(company(), BASE_SCENARIO, SPEC)
        assert result.value_per_share > 0
        assert len(result.years) == SPEC.forecast_years

    def test_growth_fades_toward_terminal(self):
        result = project(company(growth_rate=0.25), BASE_SCENARIO, SPEC)
        growths = [y.growth for y in result.years]
        assert growths[0] > growths[-1]
        assert growths[-1] == pytest.approx(result.terminal_growth, abs=1e-9)

    def test_growth_must_be_funded(self):
        """A faster-growing company reinvests more, so cash flow is not linear in growth."""
        slow = project(company(growth_rate=0.02), BASE_SCENARIO, SPEC)
        fast = project(company(growth_rate=0.20), BASE_SCENARIO, SPEC)
        assert fast.years[0].reinvestment > slow.years[0].reinvestment

    def test_higher_growth_still_raises_value_when_returns_exceed_the_cost_of_capital(self):
        slow = project(company(growth_rate=0.02, roic=0.20), BASE_SCENARIO, SPEC)
        fast = project(company(growth_rate=0.12, roic=0.20), BASE_SCENARIO, SPEC)
        assert fast.value_per_share > slow.value_per_share

    def test_higher_discount_rate_lowers_value(self):
        low = project(company(beta=0.7), BASE_SCENARIO, SPEC)
        high = project(company(beta=1.8), BASE_SCENARIO, SPEC)
        assert high.value_per_share < low.value_per_share

    def test_higher_margin_raises_value(self):
        thin = project(company(operating_margin=0.08), BASE_SCENARIO, SPEC)
        fat = project(company(operating_margin=0.30), BASE_SCENARIO, SPEC)
        assert fat.value_per_share > thin.value_per_share

    def test_more_debt_lowers_equity_value_one_for_one(self):
        a = project(company(net_debt=0.0), BASE_SCENARIO, SPEC)
        b = project(company(net_debt=5_000_000_000.0), BASE_SCENARIO, SPEC)
        per_share_gap = (a.equity_value - b.equity_value) / 1_000_000_000.0
        assert per_share_gap == pytest.approx(5.0)

    def test_share_based_compensation_reduces_value(self):
        """The cash statement adds it back; owners still paid for it."""
        none = project(company(share_based_comp=0.0), BASE_SCENARIO, SPEC)
        heavy = project(company(share_based_comp=1_000_000_000.0), BASE_SCENARIO, SPEC)
        assert heavy.value_per_share < none.value_per_share

    def test_minority_interest_and_preferred_rank_ahead_of_common(self):
        plain = project(company(), BASE_SCENARIO, SPEC)
        encumbered = project(
            company(minority_interest=1e9, preferred_equity=5e8), BASE_SCENARIO, SPEC
        )
        assert encumbered.equity_value == pytest.approx(plain.equity_value - 1.5e9)

    def test_capital_intensity_matters(self):
        """A business needing more capital per dollar of revenue is worth less."""
        light = project(company(sales_to_capital=5.0), BASE_SCENARIO, SPEC)
        heavy = project(company(sales_to_capital=1.0), BASE_SCENARIO, SPEC)
        assert light.value_per_share > heavy.value_per_share

    def test_terminal_growth_never_exceeds_the_cap(self):
        result = project(company(), BASE_SCENARIO, SPEC)
        assert result.terminal_growth <= SPEC.terminal_growth_cap

    def test_terminal_growth_stays_below_the_discount_rate(self):
        """Otherwise the perpetuity formula divides by a negative number."""
        result = project(company(beta=0.6), BASE_SCENARIO, SPEC)
        assert result.terminal_growth < result.discount_rate

    def test_extreme_growth_is_capped_and_flagged(self):
        result = project(company(growth_rate=1.5), BASE_SCENARIO, SPEC)
        assert result.years[0].growth <= 0.41
        assert any("capped" in w for w in result.warnings)

    def test_terminal_share_is_reported(self):
        result = project(company(), BASE_SCENARIO, SPEC)
        assert 0.0 < result.terminal_share < 1.0
        assert result.pv_explicit + result.terminal_value_pv == pytest.approx(
            result.enterprise_value
        )

    def test_warns_when_the_terminal_period_dominates(self):
        result = project(company(growth_rate=0.02, operating_margin=0.05), BASE_SCENARIO, SPEC)
        if result.terminal_share > SPEC.terminal_share_warning:
            assert any("terminal" in w for w in result.warnings)

    def test_zero_growth_perpetuity_matches_closed_form(self):
        """The one place the model can be checked against exact arithmetic.

        With no growth there is no reinvestment and no fade, so every year's
        free cash flow is identical and the value is an annuity plus a
        perpetuity, which must equal NOPAT / discount rate.
        """
        spec = DcfSpec(
            forecast_years=10,
            terminal_growth=0.0,
            treat_sbc_as_expense=False,
            scenarios=(
                ScenarioAssumptions(
                    label="flat",
                    growth_multiplier=1.0,
                    margin_adjustment=0.0,
                    fade_strength=1.0,
                    terminal_growth_adjustment=0.0,
                    discount_rate_adjustment=0.0,
                    probability=1.0,
                ),
            ),
        )
        inputs = DcfInputs(
            revenue=1_000_000_000.0,
            operating_margin=0.20,
            tax_rate=0.20,
            shares=1_000_000.0,
            net_debt=0.0,
            share_based_comp=0.0,
            growth_rate=0.0,
            sales_to_capital=2.0,
            roic=0.15,
            beta=1.0,
            market_cap=1e12,
            risk_free_rate=0.05,
        )
        result = project(inputs, spec.scenarios[0], spec)
        nopat = 1_000_000_000.0 * 0.20 * 0.80
        expected = nopat / result.discount_rate
        assert result.enterprise_value == pytest.approx(expected, rel=1e-9)

    def test_rejects_a_company_with_no_revenue(self):
        with pytest.raises(ValueError, match="revenue"):
            project(company(revenue=0.0), BASE_SCENARIO, SPEC)

    def test_rejects_a_company_with_no_share_count(self):
        with pytest.raises(ValueError, match="share count"):
            project(company(shares=0.0), BASE_SCENARIO, SPEC)

    def test_loss_making_company_can_still_be_valued_on_recovery(self):
        """A negative margin today does not make the model refuse; it makes it cheap."""
        result = project(
            company(operating_margin=-0.05, growth_rate=0.30, roic=None),
            next(s for s in SPEC.scenarios if s.label == "bull"),
            SPEC,
        )
        assert math.isfinite(result.value_per_share)


class TestScenarios:
    def test_bull_exceeds_base_exceeds_bear(self):
        results = run_scenarios(company(), SPEC)
        assert set(results) == {"bear", "base", "bull"}
        assert (
            results["bull"].value_per_share
            > results["base"].value_per_share
            > results["bear"].value_per_share
        )

    def test_scenarios_differ_in_every_lever(self):
        results = run_scenarios(company(), SPEC)
        assert results["bear"].discount_rate > results["bull"].discount_rate
        assert results["bear"].terminal_growth < results["bull"].terminal_growth

    def test_probabilities_sum_to_one(self):
        assert sum(s.probability for s in SPEC.scenarios) == pytest.approx(1.0)


class TestSensitivity:
    def test_reports_every_assumption_and_ranks_by_impact(self):
        rows = sensitivity(company(), SPEC)
        assert {r["assumption"] for r in rows} >= {
            "Revenue growth", "Operating margin", "Discount rate", "Reinvestment intensity",
        }
        swings = [r["swing_pct"] for r in rows if r["swing_pct"] is not None]
        assert swings == sorted(swings, reverse=True)

    def test_discount_rate_sensitivity_has_the_right_sign(self):
        rows = sensitivity(company(), SPEC)
        rate_row = next(r for r in rows if r["assumption"] == "Discount rate")
        # `high_value` is the low-discount-rate case, which must be worth more.
        assert rate_row["high_value"] > rate_row["low_value"]

    def test_a_high_growth_company_carries_more_absolute_uncertainty(self):
        """The dollar swing from a growth error scales with how much growth is assumed.

        The *relative* swing turns out to be roughly scale-invariant here, which
        is itself worth knowing: a three-point growth error costs about a fifth
        of fair value whether the company grows at 2% or 25%.
        """
        stable = sensitivity(company(growth_rate=0.02), SPEC)
        growthy = sensitivity(company(growth_rate=0.25), SPEC)

        def absolute_swing(rows, name):
            row = next(r for r in rows if r["assumption"] == name)
            return abs(row["high_value"] - row["low_value"])

        assert absolute_swing(growthy, "Revenue growth") > absolute_swing(
            stable, "Revenue growth"
        )


class TestScenarioOrdering:
    """Bull must beat base must beat bear, for every kind of company.

    This is the regression guard for an inverted fade exponent, which made a
    bull case fade growth *faster* than its base case. For a slow grower the
    discount-rate and terminal-growth differences masked it; for a company
    growing at 40% the fade dominated and the bull case came out worth less
    than the base.
    """

    @pytest.mark.parametrize("growth", [-0.10, 0.0, 0.03, 0.08, 0.15, 0.25, 0.40, 1.67])
    def test_ordering_holds_across_growth_rates(self, growth):
        results = run_scenarios(company(growth_rate=growth), SPEC)
        assert (
            results["bull"].value_per_share
            > results["base"].value_per_share
            > results["bear"].value_per_share
        ), f"scenario ordering broke at growth {growth:.0%}"

    @pytest.mark.parametrize("margin", [0.02, 0.10, 0.25, 0.50, 0.66])
    def test_ordering_holds_across_margins(self, margin):
        results = run_scenarios(company(operating_margin=margin), SPEC)
        assert (
            results["bull"].value_per_share
            > results["base"].value_per_share
            > results["bear"].value_per_share
        ), f"scenario ordering broke at margin {margin:.0%}"

    def test_a_bear_case_fades_growth_faster_than_a_bull_case(self):
        bear = project(
            company(growth_rate=0.30),
            next(s for s in SPEC.scenarios if s.label == "bear"),
            SPEC,
        )
        bull = project(
            company(growth_rate=0.30),
            next(s for s in SPEC.scenarios if s.label == "bull"),
            SPEC,
        )
        midpoint = SPEC.forecast_years // 2
        assert bull.years[midpoint].growth > bear.years[midpoint].growth
