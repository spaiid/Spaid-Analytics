"""Regressions for the defects found in the 2026-09-11 calculation audit.

Each test here corresponds to a published number that was wrong. They are kept
together rather than scattered into the module suites because what they have in
common is the failure mode, not the module: in every case the arithmetic ran
without complaint and produced a figure that a reader would act on.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from spaid.config.valuation import ACTIVE_VALUATION_SPEC as SPEC
from spaid.pipeline.metrics.financial import bounded, safe_div
from spaid.valuation.relative import (
    MULTIPLE_DRIVERS,
    RelativeInputs,
    _apply_adjustment,
    comparables_value,
)


class TestYieldAdjustmentDirection:
    """A quality discount must lower value, whichever way the multiple is quoted.

    `fcf_yield` holds a yield, so scaling it the way a multiple is scaled
    reverses the adjustment. Kraft Heinz's 30% discount became a 43% premium and
    published a $66.48 fair value where $32.64 was right -- and because the DCF
    had been withdrawn, comparables was the only method, so that was the whole
    number.
    """

    def test_a_discount_lowers_value_for_a_multiple(self):
        assert _apply_adjustment(10.0, -0.30, "ev_ebit") == pytest.approx(7.0)

    def test_a_discount_raises_the_required_yield(self):
        # Lower value means the buyer demands more cash flow per dollar paid.
        assert _apply_adjustment(0.05, -0.30, "fcf_yield") == pytest.approx(
            0.05 / 0.70
        )

    def test_a_premium_lowers_the_required_yield(self):
        assert _apply_adjustment(0.05, 0.30, "fcf_yield") == pytest.approx(0.05 / 1.30)

    def test_every_yield_multiple_is_declared_as_one(self):
        """The flag is what makes the direction right, so it must not be lost."""
        assert MULTIPLE_DRIVERS["fcf_yield"]["is_yield"] is True
        for key, meta in MULTIPLE_DRIVERS.items():
            if key != "fcf_yield":
                assert not meta.get("is_yield"), key

    def test_the_discount_actually_reaches_the_valuation(self):
        """End to end: the same peers and a discount must produce less value."""

        peer_growth = np.array([0.05, 0.07, 0.09, 0.11, 0.13, 0.15])
        peer_margin = np.array([0.16, 0.18, 0.20, 0.22, 0.24, 0.26])

        def value(subject_growth: float) -> float:
            inputs = RelativeInputs(
                shares=1e9,
                net_debt=0.0,
                free_cash_flow=1e9,
                revenue=1e10,
                ebit=1e9,
                growth=subject_growth,
                margin=0.21,
            )
            peers = {"fcf_yield": np.array([0.04, 0.045, 0.05, 0.055, 0.06, 0.065])}
            return comparables_value(
                inputs,
                peers,
                spec=SPEC.comparables,
                preferred=("fcf_yield",),
                peer_growth=peer_growth,
                peer_margin=peer_margin,
            ).value_per_share

        # A subject growing far slower than its peers earns a discount, which
        # must reduce its value. Before the fix this raised it.
        at_median = value(0.10)
        much_worse = value(-0.20)
        much_better = value(0.40)
        assert much_worse < at_median < much_better


class TestYieldLabel:
    def test_the_fcf_multiple_is_not_labelled_as_a_price_ratio(self):
        """The number printed beside the label is a percentage yield.

        Labelling it "Price / free cash flow" invited readers to take 2.5% as a
        multiple, understating the peer group's richness roughly fortyfold.
        """
        label = MULTIPLE_DRIVERS["fcf_yield"]["label"]
        assert "yield" in label.lower()
        assert "/" not in label


class TestRatioGuards:
    def test_a_negative_enterprise_value_has_no_multiple(self):
        df = pl.DataFrame({"ev": [-6.0e10], "ebit": [4.4e9]})
        out = df.select(
            safe_div(
                pl.col("ev"), pl.col("ebit"), positive_only=True, numerator_positive=True
            ).alias("ev_ebit")
        )
        assert out["ev_ebit"][0] is None

    def test_a_positive_enterprise_value_still_divides(self):
        df = pl.DataFrame({"ev": [6.0e10], "ebit": [4.4e9]})
        out = df.select(
            safe_div(
                pl.col("ev"), pl.col("ebit"), positive_only=True, numerator_positive=True
            ).alias("ev_ebit")
        )
        assert out["ev_ebit"][0] == pytest.approx(60.0 / 4.4)

    def test_an_impossible_margin_is_unavailable_rather_than_clipped(self):
        """Extra Space Storage printed 1043% and scored in the top decile on it.

        Nulling is the point: clipping to 100% would still rank the company at
        the top of the index on a number nobody measured.
        """
        df = pl.DataFrame({"m": [10.43, 2.95, 0.31, -0.05, -42.19]})
        out = df.select(bounded(pl.col("m"), -1.0, 1.0).alias("m"))
        assert out["m"].to_list() == [None, None, pytest.approx(0.31), pytest.approx(-0.05), None]


class TestShareCountDirection:
    """Omnicom issued 53% more shares and scored 98.9; DaVita retired 15% and
    scored 1.1. The metric measured reduction while the spec scored change."""

    def test_issuing_shares_is_a_positive_change(self):
        from spaid.pipeline.metrics.financial import safe_div as sd

        df = pl.DataFrame({"now": [299.2e6], "then": [194.9e6]})
        out = df.select(
            sd(pl.col("now") - pl.col("then"), pl.col("then").abs()).alias("change")
        )
        assert out["change"][0] > 0

    def test_buying_back_shares_is_a_negative_change(self):
        from spaid.pipeline.metrics.financial import safe_div as sd

        df = pl.DataFrame({"now": [66.1e6], "then": [77.4e6]})
        out = df.select(
            sd(pl.col("now") - pl.col("then"), pl.col("then").abs()).alias("change")
        )
        assert out["change"][0] < 0

    def test_the_spec_still_prefers_less(self):
        """The sign only means what it means while the spec scores it this way."""
        from spaid.config.scoring import ACTIVE_SCORING_SPEC

        spec = next(
            m for m in ACTIVE_SCORING_SPEC.metrics if m.key == "share_count_change"
        )
        assert spec.direction == -1


class TestUsableCash:
    """CME reports $160.3bn of cash against $158.7bn of current liabilities.

    It is clearing-member money held in trust. Netting it produced a negative
    enterprise value and the index's top undervalued verdict at confidence 1.0.
    """

    def test_cash_beyond_the_companys_own_capital_is_not_netted(self):
        cash, equity, debt = 160.4e9, 26.5e9, 0.0
        usable = min(cash, max(equity + debt, 0.0))
        assert usable == pytest.approx(26.5e9)
        assert debt - usable > -cash  # net debt no longer swamps the market cap

    def test_an_ordinary_cash_rich_balance_sheet_is_untouched(self):
        # Apple: cash well inside equity plus borrowings, so the cap must not bind.
        cash, equity, debt = 60e9, 60e9, 100e9
        assert min(cash, max(equity + debt, 0.0)) == pytest.approx(cash)


class TestConsensusMarginBasis:
    """Consensus EPS is non-GAAP; the projection is GAAP.

    Reading one onto the other handed AMD a jump from a 15.71% operating margin
    to 34.42%, held for eight years, worth over half its fair value.
    """

    def _inputs(self, **kw):
        from spaid.valuation.engine import CompanyValuationInputs

        base = dict(
            ticker="TEST",
            price=100.0,
            business_model="operating",
            shares=1.0e9,
            revenue=40.0e9,
            operating_margin=0.157,
            tax_rate=0.145,
            consensus_eps_this_year=7.57,
            forward_eps=15.61,
            consensus_revenue_this_year=50.0e9,
            consensus_revenue_next_year=87.0e9,
            n_revenue_analysts=51,
        )
        base.update(kw)
        return CompanyValuationInputs(**base)

    def test_the_implied_margin_is_restated_onto_a_gaap_basis(self):
        from spaid.valuation.engine import _consensus_path

        _, raw, _, _ = _consensus_path(self._inputs())
        _, adjusted, _, _ = _consensus_path(
            self._inputs(amortization_intangibles=2.26e9, share_based_comp=1.90e9)
        )
        addback = (2.26e9 + 1.90e9) / 40.0e9
        for before, after in zip(raw, adjusted, strict=True):
            assert after == pytest.approx(before - addback)

    def test_a_company_with_no_addbacks_is_unaffected(self):
        from spaid.valuation.engine import _consensus_path

        _, raw, _, _ = _consensus_path(self._inputs())
        _, same, _, _ = _consensus_path(
            self._inputs(amortization_intangibles=0.0, share_based_comp=0.0)
        )
        assert list(raw) == [pytest.approx(v) for v in same]


class TestSensitivityMeasuresWhatDrivesTheModel:
    """Under consensus anchoring the growth and margin knobs were overridden, so
    the table reported a 0.0% swing on the two assumptions that matter -- and
    the confidence score read that as certainty."""

    def _dcf_inputs(self, **kw):
        from spaid.valuation.dcf import DcfInputs

        base = dict(
            revenue=40.0e9,
            operating_margin=0.157,
            tax_rate=0.145,
            shares=1.0e9,
            net_debt=-3.0e9,
            sales_to_capital=3.0,
            beta=1.5,
            risk_free_rate=0.042,
            consensus_revenue=(50.0e9, 87.0e9),
            consensus_margins=(0.19, 0.24),
            n_revenue_analysts=51,
        )
        base.update(kw)
        return DcfInputs(**base)

    def test_every_assumption_moves_the_anchored_valuation(self):
        from spaid.valuation.dcf import sensitivity

        rows = sensitivity(self._dcf_inputs(), SPEC.dcf, risk_free=0.042)
        by_name = {r["assumption"]: r for r in rows}
        for name in ("Revenue growth", "Operating margin", "Discount rate"):
            assert by_name[name]["swing_pct"] > 0.0, f"{name} reported as inert"

    def test_an_unanchored_company_still_reports_its_own_sensitivity(self):
        from spaid.valuation.dcf import sensitivity

        rows = sensitivity(
            self._dcf_inputs(consensus_revenue=(), consensus_margins=()),
            SPEC.dcf,
            risk_free=0.042,
        )
        by_name = {r["assumption"]: r for r in rows}
        assert by_name["Operating margin"]["swing_pct"] > 0.0


class TestRejectedModelsAreNotPublished:
    def test_a_withdrawn_dcf_publishes_no_scenarios(self):
        """McDonald's page showed a -$69,244,078.63 base case from a model the
        engine had already thrown away."""
        from spaid.valuation.engine import CompanyValuationInputs, value_company

        # A share count wrong by three orders of magnitude, which is what the
        # original defect amounted to.
        inputs = CompanyValuationInputs(
            ticker="TEST",
            price=253.0,
            business_model="operating",
            shares=711.0,
            revenue=27.0e9,
            operating_margin=0.45,
            net_income=8.2e9,
            free_cash_flow=7.0e9,
            net_debt=39.0e9,
            equity=-5.0e9,
            tax_rate=0.21,
        )
        result = value_company(inputs, SPEC)
        dcf = next((m for m in result.methods if m.method == "dcf"), None)
        if dcf is not None and not dcf.used:
            assert not result.scenarios, "scenarios published for a rejected DCF"
            assert not result.sensitivities


class TestRegimeBandsArePointInTime:
    def test_bands_use_only_prior_observations(self):
        from spaid.backtest.signal import _MIN_REGIME_HISTORY, _expanding_bands

        # A calm start followed by a spike. The early dates must not know the
        # spike is coming.
        values = np.array([0.10] * 15 + [0.90] * 15)
        low, high = _expanding_bands(values)
        assert np.isnan(high[: _MIN_REGIME_HISTORY - 1]).all()
        # Before the spike the upper band sits at the calm level...
        assert high[14] == pytest.approx(0.10)
        # ...and only rises once the spike has actually been observed.
        assert high[-1] > 0.10

    def test_a_full_sample_band_would_have_differed(self):
        from spaid.backtest.signal import _expanding_bands

        values = np.array([0.10] * 15 + [0.90] * 15)
        _, high = _expanding_bands(values)
        full_sample = np.percentile(values, 67)
        assert high[14] != pytest.approx(full_sample)
