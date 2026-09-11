"""Tests for the fair-value blending engine and its classification rules."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spaid.config.scoring import BusinessModel
from spaid.config.valuation import ClassificationSpec, ValuationClass, get_valuation_spec
from spaid.valuation.engine import (
    CompanyValuationInputs,
    classify,
    confidence_label,
    value_company,
)

SPEC = get_valuation_spec()


def operating(**kw) -> CompanyValuationInputs:
    """A profitable operating company with full peer and history context."""
    rng = np.random.default_rng(7)
    defaults = dict(
        ticker="TEST",
        price=100.0,
        business_model="operating",
        shares=1_000_000_000.0,
        revenue=10_000_000_000.0,
        operating_income=2_000_000_000.0,
        ebitda=2_600_000_000.0,
        net_income=1_500_000_000.0,
        free_cash_flow=1_600_000_000.0,
        share_based_comp=150_000_000.0,
        depreciation=600_000_000.0,
        capex=400_000_000.0,
        equity=8_000_000_000.0,
        net_debt=2_000_000_000.0,
        operating_margin=0.20,
        tax_rate=0.21,
        roic=0.16,
        roe=0.19,
        revenue_growth=0.07,
        forward_eps_growth=0.09,
        forward_eps=1.75,
        sales_to_capital=2.0,
        beta=1.05,
        market_cap=100_000_000_000.0,
        risk_free_rate=0.042,
        peer_multiples={
            "ev_ebit": rng.normal(18.0, 3.0, 25),
            "ev_ebitda": rng.normal(14.0, 2.0, 25),
            "fcf_yield": rng.normal(0.05, 0.01, 25),
        },
        peer_growth=rng.normal(0.06, 0.03, 25),
        peer_margin=rng.normal(0.18, 0.05, 25),
        own_multiple_history={"ev_ebit": rng.normal(17.0, 2.0, 1300)},
        historical_growth=0.06,
        historical_margin=0.19,
        metric_coverage=0.9,
        data_age_days=45.0,
        n_analysts=20,
    )
    defaults.update(kw)
    return CompanyValuationInputs(**defaults)


class TestClassification:
    SPEC_C = ClassificationSpec()

    def _classify(self, price, midpoint, low, high, confidence=0.8):
        return classify(
            price=price, midpoint=midpoint, low=low, high=high,
            confidence=confidence, spec=self.SPEC_C,
        )

    def test_large_discount_is_significantly_undervalued(self):
        assert self._classify(100.0, 150.0, 120.0, 180.0) == (
            ValuationClass.SIGNIFICANTLY_UNDERVALUED
        )

    def test_price_inside_a_tight_range_is_fairly_valued(self):
        assert self._classify(100.0, 102.0, 95.0, 109.0) == ValuationClass.FAIRLY_VALUED

    def test_large_premium_is_significantly_overvalued(self):
        assert self._classify(200.0, 120.0, 100.0, 140.0) == (
            ValuationClass.SIGNIFICANTLY_OVERVALUED
        )

    def test_a_wide_range_demands_more_upside_for_the_same_verdict(self):
        """The central rule: uncertainty raises the bar for a verdict."""
        tight = self._classify(100.0, 115.0, 110.0, 120.0)
        wide = self._classify(100.0, 115.0, 40.0, 190.0)
        assert tight == ValuationClass.UNDERVALUED
        assert wide == ValuationClass.FAIRLY_VALUED

    def test_low_confidence_declines_to_classify(self):
        assert self._classify(100.0, 200.0, 150.0, 250.0, confidence=0.2) == (
            ValuationClass.INSUFFICIENT_CONFIDENCE
        )

    def test_missing_midpoint_declines_to_classify(self):
        assert self._classify(100.0, None, None, None) == (
            ValuationClass.INSUFFICIENT_CONFIDENCE
        )

    def test_a_low_multiple_alone_does_not_produce_undervalued(self):
        """Classification reads the range, never a bare multiple.

        A company whose fair-value range sits below the price is overvalued no
        matter how cheap its price/earnings ratio looks.
        """
        assert self._classify(100.0, 70.0, 55.0, 85.0) == (
            ValuationClass.SIGNIFICANTLY_OVERVALUED
        )


class TestConfidenceLabel:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [(0.9, "High"), (0.6, "Moderate"), (0.4, "Low"), (0.1, "Very low")],
    )
    def test_labels(self, score, expected):
        assert confidence_label(score) == expected


class TestValueCompany:
    def test_produces_an_ordered_range_for_a_normal_company(self):
        r = value_company(operating(), SPEC)
        assert r.bear is not None and r.base is not None and r.bull is not None
        assert r.bear <= r.base <= r.bull
        assert r.range_low == r.bear
        assert r.range_high == r.bull
        assert r.midpoint == pytest.approx((r.bear + r.bull) / 2)

    def test_runs_all_three_methods_for_an_operating_company(self):
        r = value_company(operating(), SPEC)
        used = {m.method for m in r.methods if m.used}
        assert used == {"dcf", "comparables", "historical"}

    def test_upside_is_a_fraction_relative_to_price(self):
        r = value_company(operating(price=50.0), SPEC)
        assert r.upside == pytest.approx(r.midpoint / 50.0 - 1.0)

    def test_a_bank_never_runs_a_discounted_cash_flow_model(self):
        """A bank's borrowing is its raw material; free cash flow is meaningless."""
        r = value_company(
            operating(business_model="bank", roe=0.14, equity=30_000_000_000.0), SPEC
        )
        methods = {m.method for m in r.methods}
        assert "dcf" not in methods
        assert "residual_income" in methods

    def test_a_reit_uses_funds_from_operations(self):
        r = value_company(operating(business_model="reit"), SPEC)
        methods = {m.method for m in r.methods}
        assert "nav_affo" in methods
        assert "dcf" not in methods

    def test_a_cyclical_uses_normalised_earnings(self):
        rng = np.random.default_rng(3)
        r = value_company(
            operating(
                business_model="cyclical",
                margin_history=rng.normal(0.12, 0.06, 40),
            ),
            SPEC,
        )
        methods = {m.method for m in r.methods}
        assert "normalized_earnings" in methods

    def test_a_skipped_method_says_why_and_redistributes_its_weight(self):
        """Weights must renormalise over the methods that actually ran."""
        r = value_company(operating(own_multiple_history={}, peer_multiples={}), SPEC)
        skipped = [m for m in r.methods if not m.used]
        assert skipped
        assert all(m.reason for m in skipped)
        used = [m for m in r.methods if m.used]
        assert used
        # The base case is the weighted mean over the survivors alone.
        total = sum(m.weight for m in used)
        expected = sum(m.value_per_share * m.weight for m in used) / total
        assert r.base == pytest.approx(expected)

    def test_no_usable_data_returns_no_classification_rather_than_a_guess(self):
        r = value_company(
            CompanyValuationInputs(
                ticker="EMPTY", price=10.0, business_model="operating"
            ),
            SPEC,
        )
        assert r.base is None
        assert r.classification == ValuationClass.INSUFFICIENT_CONFIDENCE
        assert r.confidence == 0.0
        assert r.caveats

    def test_method_disagreement_lowers_confidence_and_widens_the_range(self):
        rng = np.random.default_rng(11)
        agree = operating(
            peer_multiples={"ev_ebit": rng.normal(18.0, 0.5, 30)},
            own_multiple_history={"ev_ebit": rng.normal(18.0, 0.5, 1300)},
        )
        disagree = operating(
            peer_multiples={"ev_ebit": rng.normal(45.0, 1.0, 30)},
            own_multiple_history={"ev_ebit": rng.normal(6.0, 0.5, 1300)},
        )
        a = value_company(agree, SPEC)
        d = value_company(disagree, SPEC)
        assert d.method_dispersion > a.method_dispersion
        assert d.confidence < a.confidence

    def test_unprofitable_growth_is_structurally_less_confident(self):
        profitable = value_company(operating(), SPEC)
        speculative = value_company(
            operating(business_model="unprofitable_growth", operating_margin=-0.15), SPEC
        )
        assert speculative.confidence < profitable.confidence

    def test_stale_data_lowers_confidence(self):
        fresh = value_company(operating(data_age_days=20.0), SPEC)
        stale = value_company(operating(data_age_days=200.0), SPEC)
        assert stale.confidence < fresh.confidence

    def test_thin_coverage_lowers_confidence(self):
        full = value_company(operating(metric_coverage=0.95), SPEC)
        thin = value_company(operating(metric_coverage=0.35), SPEC)
        assert thin.confidence < full.confidence

    def test_an_implausible_method_result_is_rejected_not_blended(self):
        """A method saying a $100 stock is worth $5,000 has read bad data."""
        r = value_company(
            operating(peer_multiples={"ev_ebit": np.full(30, 3000.0)}), SPEC
        )
        comps = next(m for m in r.methods if m.method == "comparables")
        assert not comps.used
        assert "implausible" in (comps.reason or "")

    def test_never_returns_a_bare_number_without_a_range(self):
        """The brief forbids presenting fair value as a precise figure."""
        r = value_company(operating(), SPEC)
        assert r.range_low is not None and r.range_high is not None
        assert r.range_high > r.range_low

    def test_scenarios_and_sensitivities_are_exposed(self):
        r = value_company(operating(), SPEC)
        assert {s["label"] for s in r.scenarios} == {"bear", "base", "bull"}
        assert r.sensitivities
        assert all("assumption" in s for s in r.sensitivities)

    def test_caveats_are_populated_and_deduplicated(self):
        r = value_company(operating(metric_coverage=0.3, data_age_days=300.0), SPEC)
        assert r.caveats
        assert len(r.caveats) == len(set(r.caveats))

    def test_higher_price_moves_the_classification_toward_overvalued(self):
        cheap = value_company(operating(price=40.0), SPEC)
        dear = value_company(operating(price=250.0), SPEC)
        order = [
            ValuationClass.SIGNIFICANTLY_UNDERVALUED,
            ValuationClass.UNDERVALUED,
            ValuationClass.FAIRLY_VALUED,
            ValuationClass.OVERVALUED,
            ValuationClass.SIGNIFICANTLY_OVERVALUED,
        ]
        assert order.index(cheap.classification) < order.index(dear.classification)


class TestRangeIntegrity:
    """The fair-value range must always be internally coherent.

    These cover a real defect: a discounted cash-flow model that was rejected as
    implausible still supplied the bear and bull scenarios, so scaling the blend
    by the ratio between them turned a rejected $1.80 base case into a $1,904
    bull case and a midpoint five times the market price.
    """

    def test_base_always_sits_inside_the_range(self):
        rng = np.random.default_rng(5)
        for seed in range(12):
            r = value_company(
                operating(
                    price=float(20 * (seed + 1)),
                    operating_margin=float(rng.uniform(-0.05, 0.35)),
                    revenue_growth=float(rng.uniform(-0.1, 0.35)),
                    beta=float(rng.uniform(0.5, 2.2)),
                ),
                SPEC,
            )
            if r.base is None:
                continue
            assert r.range_low <= r.base <= r.range_high, (
                f"range {r.range_low:.2f}-{r.range_high:.2f} excludes base {r.base:.2f}"
            )

    def test_a_rejected_dcf_does_not_supply_the_scenario_range(self):
        """The Advanced Micro Devices case: a rejected model must not set the bull."""
        # Margins so thin the discounted cash-flow model collapses to near zero
        # and is rejected, while comparables still produce a sane figure.
        r = value_company(
            operating(
                price=500.0,
                operating_margin=0.002,
                revenue=34_000_000_000.0,
                sales_to_capital=0.9,
            ),
            SPEC,
        )
        dcf = next(m for m in r.methods if m.method == "dcf")
        if dcf.used or r.base is None:
            pytest.skip("this fixture did not trigger a discounted cash-flow rejection")
        assert r.range_high is not None
        # The range must stay anchored to the methods that actually ran.
        assert r.range_high <= r.base * 4.0
        assert r.range_low <= r.base <= r.range_high

    def test_the_range_is_never_absurdly_wide(self):
        r = value_company(operating(), SPEC)
        assert r.range_high / max(r.range_low, 1e-9) < 10.0

    def test_midpoint_is_the_centre_of_the_range(self):
        r = value_company(operating(), SPEC)
        assert r.midpoint == pytest.approx((r.range_low + r.range_high) / 2)


class TestRangeWidthConfidence:
    def test_a_very_wide_range_cannot_be_high_confidence(self):
        """A range spanning a factor of three is not a precise view.

        Micron produced a $910 to $3,416 range on complete, fresh data and still
        reported high confidence, because every other component was healthy. The
        width of the range is itself evidence about how much is known.
        """
        rng = np.random.default_rng(21)
        wide = value_company(
            operating(
                peer_multiples={"ev_ebit": rng.normal(60.0, 2.0, 30)},
                own_multiple_history={"ev_ebit": rng.normal(9.0, 1.0, 1300)},
            ),
            SPEC,
        )
        if wide.midpoint is None:
            pytest.skip("fixture produced no valuation")
        width = (wide.range_high - wide.range_low) / abs(wide.midpoint)
        if width > 1.0:
            assert wide.confidence < 0.75, (
                f"range width {width:.2f} still reported confidence {wide.confidence:.2f}"
            )

    def test_a_tight_range_on_good_data_keeps_confidence(self):
        rng = np.random.default_rng(22)
        tight = value_company(
            operating(
                peer_multiples={"ev_ebit": rng.normal(18.0, 0.6, 30)},
                own_multiple_history={"ev_ebit": rng.normal(18.0, 0.6, 1300)},
            ),
            SPEC,
        )
        assert tight.confidence >= 0.5


class TestRangeSpansMethods:
    def test_the_range_covers_every_method_that_ran(self):
        rng = np.random.default_rng(31)
        r = value_company(
            operating(
                peer_multiples={"ev_ebit": rng.normal(35.0, 2.0, 30)},
                own_multiple_history={"ev_ebit": rng.normal(12.0, 1.0, 1300)},
            ),
            SPEC,
        )
        used = [m.value_per_share for m in r.methods if m.used and m.value_per_share]
        if len(used) < 2:
            pytest.skip("fixture produced fewer than two usable methods")
        assert r.range_low <= min(used)
        assert r.range_high >= max(used)

    def test_the_range_does_not_extend_beyond_the_methods(self):
        """Padding an extreme method compounds it into a value nobody proposed."""
        rng = np.random.default_rng(32)
        r = value_company(
            operating(
                peer_multiples={"ev_ebit": rng.normal(35.0, 2.0, 30)},
                own_multiple_history={"ev_ebit": rng.normal(12.0, 1.0, 1300)},
            ),
            SPEC,
        )
        used = [m.value_per_share for m in r.methods if m.used and m.value_per_share]
        if len(used) < 2:
            pytest.skip("fixture produced fewer than two usable methods")
        # The upper bound may exceed the highest method only via the bull
        # scenario, never by an arbitrary percentage on top of it.
        scenario_high = max(
            [s["value_per_share"] for s in r.scenarios] or [0.0]
        )
        assert r.range_high <= max(max(used), scenario_high) * 1.001
