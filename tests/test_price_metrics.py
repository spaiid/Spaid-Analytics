"""Tests for the price-derived metric kernels.

Each of these replaces a formula the audit found wrong, so the tests are written
against the *definition* rather than against the previous implementation:

* maximum drawdown compared a trough with a peak up to two windows earlier, so a
  recovered stock carried a two-year-old drawdown for another year;
* downside volatility measured the dispersion of down days rather than
  semideviation, so a stock that fell 1% every single down day scored as safe;
* Amihud illiquidity divided by an epsilon on zero-volume days, producing a
  value around 1e19 that poisoned the rolling mean for three months.
"""

from __future__ import annotations

import numpy as np
import pytest

from spaid.pipeline.metrics.price import (
    ANNUALISE,
    _rolling_beta,
    amihud_illiquidity,
    downside_beta,
    downside_deviation,
    rolling_max_drawdown,
)


class TestRollingMaxDrawdown:
    def test_simple_decline(self):
        prices = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
        out = rolling_max_drawdown(prices, 5)
        assert out[-1] == pytest.approx(-0.40)

    def test_peak_must_be_inside_the_window(self):
        """The defect: a trough compared against a peak from before the window.

        The price peaks at 200 early, crashes to 100, then recovers steadily.
        Over the trailing five observations the peak is 150 and the trough 150,
        so the within-window drawdown is zero -- not the -50% measured from a
        peak that left the window long ago.
        """
        prices = np.array([200.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
        out = rolling_max_drawdown(prices, 5)
        assert out[-1] == pytest.approx(0.0, abs=1e-12)

    def test_naive_formulation_would_differ(self):
        """Pin the distinction the old implementation got wrong."""
        prices = np.array([200.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
        window = 5
        correct = rolling_max_drawdown(prices, window)[-1]

        # The old formula: rolling minimum of (price / rolling maximum - 1).
        ratios = []
        for t in range(len(prices)):
            lo = max(0, t - window + 1)
            ratios.append(prices[t] / prices[lo : t + 1].max() - 1.0)
        naive = min(ratios[-window:])
        assert naive < correct - 0.3

    def test_drawdown_is_never_positive(self):
        rng = np.random.default_rng(4)
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 500)))
        out = rolling_max_drawdown(prices, 60)
        finite = out[np.isfinite(out)]
        assert finite.size
        assert finite.max() <= 1e-12

    def test_insufficient_history_returns_nan(self):
        out = rolling_max_drawdown(np.array([100.0, 99.0]), 60)
        assert np.isnan(out).all()

    def test_monotone_rise_has_no_drawdown(self):
        prices = np.arange(100.0, 200.0)
        out = rolling_max_drawdown(prices, 20)
        assert np.nanmin(out) == pytest.approx(0.0, abs=1e-12)


class TestDownsideDeviation:
    def test_matches_the_semideviation_definition(self):
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.0, -0.015] * 20)
        out = downside_deviation(returns, len(returns), min_samples=10)
        expected = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)) * ANNUALISE
        assert out[-1] == pytest.approx(expected)

    def test_distinguishes_consistent_small_losses_from_one_large_one(self):
        """The definition that matters.

        A stock falling 1% on every down day has real downside risk. The
        dispersion of its down days is zero, so the old measure called it safe.
        """
        steady = np.array([0.01, -0.01] * 60)
        lumpy = np.array([0.01, -0.001] * 59 + [0.01, -0.08])

        steady_dd = downside_deviation(steady, len(steady), min_samples=10)[-1]
        lumpy_dd = downside_deviation(lumpy, len(lumpy), min_samples=10)[-1]
        assert steady_dd > 0.0

        # Under the rejected definition -- standard deviation of the down days
        # alone -- the steady stock scores zero risk.
        steady_down = steady[steady < 0]
        assert np.std(steady_down, ddof=1) == pytest.approx(0.0)
        assert steady_dd > 0.05

        assert lumpy_dd > 0.0

    def test_all_positive_returns_give_zero(self):
        returns = np.full(200, 0.01)
        out = downside_deviation(returns, 100, min_samples=10)
        assert out[-1] == pytest.approx(0.0)

    def test_requires_a_minimum_sample(self):
        returns = np.array([0.01, -0.01, 0.02])
        out = downside_deviation(returns, 3, min_samples=10)
        assert np.isnan(out[-1])


class TestAmihudIlliquidity:
    def test_zero_volume_days_are_excluded_not_divided_by(self):
        """The defect: one zero-volume print poisoned three months of readings."""
        n = 200
        returns = np.full(n, 0.01)
        volume = np.full(n, 1e9)
        volume[100] = 0.0  # a bad print with a real price move

        out = amihud_illiquidity(returns, volume, 63, min_samples=10)
        after = out[101:163]
        finite = after[np.isfinite(after)]
        assert finite.size
        # Without the guard this would be around 1e19.
        assert finite.max() < 1.0

    def test_value_matches_the_definition(self):
        returns = np.array([0.02, -0.01, 0.015, -0.005] * 30)
        volume = np.full(returns.size, 5e8)
        out = amihud_illiquidity(returns, volume, returns.size, min_samples=10)
        expected = np.mean(np.abs(returns) / volume) * 1e6
        assert out[-1] == pytest.approx(expected)

    def test_less_liquid_names_score_higher(self):
        returns = np.full(200, 0.02)
        liquid = amihud_illiquidity(returns, np.full(200, 1e10), 63, min_samples=10)[-1]
        illiquid = amihud_illiquidity(returns, np.full(200, 1e7), 63, min_samples=10)[-1]
        assert illiquid > liquid


class TestBeta:
    def test_matches_covariance_over_variance(self):
        rng = np.random.default_rng(9)
        market = rng.normal(0.0004, 0.01, 400)
        stock = 1.4 * market + rng.normal(0, 0.004, 400)
        out = _rolling_beta(stock, market, 400, min_samples=100)
        expected = np.cov(stock, market, ddof=0)[0, 1] / np.var(market)
        assert out[-1] == pytest.approx(expected, rel=1e-9)

    def test_recovers_a_known_beta(self):
        rng = np.random.default_rng(10)
        market = rng.normal(0.0004, 0.012, 800)
        stock = 0.7 * market + rng.normal(0, 0.001, 800)
        out = _rolling_beta(stock, market, 500, min_samples=100)
        assert out[-1] == pytest.approx(0.7, abs=0.05)

    def test_zero_market_variance_returns_nan(self):
        market = np.zeros(300)
        stock = np.full(300, 0.01)
        out = _rolling_beta(stock, market, 250, min_samples=50)
        assert np.isnan(out[-1])


class TestDownsideBeta:
    def test_uses_only_down_market_days(self):
        rng = np.random.default_rng(12)
        market = rng.normal(0.0, 0.012, 900)
        # Twice as sensitive when the market falls as when it rises.
        stock = np.where(market < 0, 2.0 * market, 0.5 * market) + rng.normal(0, 0.001, 900)

        full = _rolling_beta(stock, market, 800, min_samples=100)[-1]
        down = downside_beta(stock, market, 800, min_samples=100)[-1]
        assert down == pytest.approx(2.0, abs=0.15)
        assert down > full

    def test_requires_enough_down_days(self):
        market = np.full(300, 0.01)  # the market never falls
        stock = np.full(300, 0.01)
        out = downside_beta(stock, market, 250, min_samples=40)
        assert np.isnan(out[-1])


class TestNoLookAhead:
    """Every kernel must be computable from history alone.

    Truncating the series must not change any value that was already computed.
    """

    @pytest.mark.parametrize(
        "kernel",
        [
            lambda p, r, m, v: rolling_max_drawdown(p, 60),
            lambda p, r, m, v: downside_deviation(r, 60, min_samples=10),
            lambda p, r, m, v: amihud_illiquidity(r, v, 60, min_samples=10),
            lambda p, r, m, v: _rolling_beta(r, m, 60, min_samples=10),
            lambda p, r, m, v: downside_beta(r, m, 60, min_samples=10),
        ],
    )
    def test_truncation_does_not_change_earlier_values(self, kernel):
        rng = np.random.default_rng(15)
        n = 300
        returns = rng.normal(0.0005, 0.015, n)
        market = rng.normal(0.0004, 0.012, n)
        prices = 100 * np.exp(np.cumsum(returns))
        volume = rng.uniform(1e8, 1e9, n)

        full = kernel(prices, returns, market, volume)
        cut = 200
        truncated = kernel(prices[:cut], returns[:cut], market[:cut], volume[:cut])

        a, b = full[:cut], truncated
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 50, "expected overlapping finite values to compare"
        np.testing.assert_allclose(a[both], b[both], rtol=1e-12, atol=1e-12)
