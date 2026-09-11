"""Tests for the peer-multiple valuation and its blend.

Choosing one multiple and discarding the rest hides the largest source of
uncertainty in a relative valuation. The old code took the first preferred
multiple with enough peers, which resolved to EV/EBIT for 345 of 366 operating
companies; for AMD the five usable multiples ran from $140 to $267 a share.
"""

from __future__ import annotations

import numpy as np
import pytest

from spaid.config.valuation import ComparablesSpec
from spaid.valuation.relative import RelativeInputs, comparables_value

SPEC = ComparablesSpec()


def subject(**kw) -> RelativeInputs:
    defaults = dict(
        shares=1_000_000_000.0,
        net_debt=0.0,
        revenue=10_000_000_000.0,
        ebit=2_000_000_000.0,
        ebitda=2_600_000_000.0,
        net_income=1_500_000_000.0,
        free_cash_flow=1_600_000_000.0,
        forward_earnings=1_800_000_000.0,
        equity=8_000_000_000.0,
        growth=0.08,
        margin=0.20,
    )
    defaults.update(kw)
    return RelativeInputs(**defaults)


def peers(**kw) -> dict[str, np.ndarray]:
    """Tight peer groups, so each multiple's own spread is not the variable."""
    return {key: np.full(12, value) for key, value in kw.items()}


class TestBlending:
    def test_every_usable_multiple_contributes(self):
        result = comparables_value(
            subject(),
            peers(ev_ebit=18.0, ev_ebitda=14.0, forward_pe=20.0),
            spec=SPEC,
        )
        assert {c["multiple"] for c in result.components} == {
            "ev_ebit",
            "ev_ebitda",
            "forward_pe",
        }

    def test_the_estimate_is_the_median_across_multiples(self):
        result = comparables_value(
            subject(),
            peers(ev_ebit=18.0, ev_ebitda=14.0, forward_pe=20.0),
            spec=SPEC,
        )
        values = sorted(c["value_per_share"] for c in result.components)
        assert result.value_per_share == pytest.approx(values[1])

    def test_the_range_is_the_spread_across_multiples(self):
        result = comparables_value(
            subject(),
            peers(ev_ebit=18.0, ev_ebitda=14.0, forward_pe=20.0),
            spec=SPEC,
        )
        values = [c["value_per_share"] for c in result.components]
        assert result.value_low == pytest.approx(min(values))
        assert result.value_high == pytest.approx(max(values))

    def test_disagreeing_multiples_widen_the_range(self):
        tight = comparables_value(
            subject(), peers(ev_ebit=18.0, ev_ebitda=13.8), spec=SPEC
        )
        wide = comparables_value(
            subject(), peers(ev_ebit=18.0, ev_ebitda=30.0), spec=SPEC
        )
        assert (wide.value_high - wide.value_low) > (tight.value_high - tight.value_low)

    def test_a_multiple_with_too_few_peers_is_excluded(self):
        group = peers(ev_ebit=18.0)
        group["ev_ebitda"] = np.full(SPEC.min_peers - 1, 14.0)
        result = comparables_value(subject(), group, spec=SPEC)
        assert [c["multiple"] for c in result.components] == ["ev_ebit"]

    def test_the_representative_multiple_is_one_that_actually_ran(self):
        result = comparables_value(
            subject(),
            peers(ev_ebit=18.0, ev_ebitda=14.0, forward_pe=20.0),
            spec=SPEC,
        )
        assert result.multiple_used in {c["multiple"] for c in result.components}


class TestNearZeroDenominators:
    def test_a_near_zero_driver_is_excluded_from_the_blend(self):
        """Gilead's EBITDA net of an impairment was $0.27bn against a 737x EV."""
        result = comparables_value(
            subject(ebitda=30_000_000.0),
            peers(ev_ebit=18.0, ev_ebitda=14.0, forward_pe=20.0),
            spec=SPEC,
        )
        assert "ev_ebitda" not in {c["multiple"] for c in result.components}
        assert any("close to zero" in w for w in result.warnings)

    def test_the_range_is_not_blown_open_by_it(self):
        clean = comparables_value(
            subject(), peers(ev_ebit=18.0, forward_pe=20.0), spec=SPEC
        )
        distorted = comparables_value(
            subject(ebitda=30_000_000.0),
            peers(ev_ebit=18.0, ev_ebitda=14.0, forward_pe=20.0),
            spec=SPEC,
        )
        assert distorted.value_high == pytest.approx(clean.value_high)

    def test_a_negative_driver_never_produces_a_value(self):
        result = comparables_value(
            subject(ebit=-500_000_000.0),
            peers(ev_ebit=18.0, ev_ebitda=14.0),
            spec=SPEC,
        )
        assert [c["multiple"] for c in result.components] == ["ev_ebitda"]


class TestSingleMultiple:
    def test_a_lone_multiple_uses_its_own_peer_spread(self):
        group = {"ev_ebit": np.linspace(12.0, 24.0, 20)}
        result = comparables_value(subject(), group, spec=SPEC)
        assert len(result.components) == 1
        assert result.value_low < result.value_per_share < result.value_high

    def test_nothing_usable_returns_no_value_with_a_reason(self):
        result = comparables_value(subject(), {}, spec=SPEC)
        assert result.value_per_share is None
        assert result.warnings


class TestPresentation:
    def test_a_yield_is_described_as_a_percentage(self):
        """Two decimal places of 0.03 renders as 0.0 and reads as missing."""
        result = comparables_value(
            subject(), peers(fcf_yield=0.03, ev_ebit=18.0), spec=SPEC
        )
        assert "3.0%" in result.detail

    def test_the_detail_names_every_multiple_in_the_blend(self):
        result = comparables_value(
            subject(), peers(ev_ebit=18.0, ev_ebitda=14.0), spec=SPEC
        )
        assert "EV / EBIT" in result.detail
        assert "EV / EBITDA" in result.detail
