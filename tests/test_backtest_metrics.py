"""Performance statistics, checked against returns whose answers are known.

Compounding, annualising and drawdown arithmetic are easy to get subtly wrong
and impossible to spot afterwards, so every case here uses a series short and
regular enough that the expected value can be written down.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from spaid.backtest import metrics as M


def daily_dates(n: int, start: date = date(2016, 1, 4)) -> list[date]:
    """`n` weekdays, so a year is roughly 252 of them."""
    out: list[date] = []
    day = start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


class TestReturns:
    def test_returns_come_out_of_an_equity_path_unchanged(self):
        equity = np.array([100.0, 110.0, 99.0])
        returns = M.to_returns(equity)
        assert returns[0] == pytest.approx(0.10)
        assert returns[1] == pytest.approx(-0.10)

    def test_total_return_is_the_whole_path(self):
        assert M.total_return(np.array([100.0, 110.0, 121.0])) == pytest.approx(0.21)

    def test_doubling_over_exactly_two_years_is_a_41_percent_cagr(self):
        """2^(1/2) - 1 = 0.414213..., measured on the calendar, not on bar count."""
        dates = [date(2016, 1, 1), date(2018, 1, 1)]
        equity = np.array([1.0, 2.0])
        assert M.cagr(equity, dates) == pytest.approx(math.sqrt(2.0) - 1.0, rel=1e-3)

    def test_a_flat_path_has_no_growth_and_no_volatility(self):
        dates = daily_dates(300)
        equity = np.ones(300)
        assert M.cagr(equity, dates) == pytest.approx(0.0)
        assert M.volatility(M.to_returns(equity)) == pytest.approx(0.0)

    def test_volatility_annualises_by_the_root_of_the_trading_year(self):
        rng = np.random.default_rng(11)
        daily = rng.normal(0.0, 0.01, 5000)
        assert M.volatility(daily) == pytest.approx(0.01 * math.sqrt(252), rel=0.05)


class TestRiskAdjusted:
    def test_sharpe_is_mean_over_deviation_annualised(self):
        returns = np.array([0.01, -0.005, 0.02, 0.0, 0.005] * 60)
        expected = returns.mean() / returns.std(ddof=1) * math.sqrt(252)
        assert M.sharpe(returns) == pytest.approx(expected)

    def test_sortino_treats_upside_and_downside_deviation_differently(self):
        """Sortino should exceed Sharpe when the big moves are upward, and fall
        below it when they are downward. That asymmetry is the whole reason the
        statistic exists; a Sortino that merely tracks Sharpe is not measuring
        downside risk."""
        # One symmetric series, shifted up and down by the same amount. The
        # standard deviation is identical in both cases, so Sharpe moves only
        # with the mean -- but the shift changes how much of the movement lands
        # below zero, which is the only thing Sortino counts.
        swings = np.array([0.02, -0.02] * 200)
        shifted_up = swings + 0.005    # losses shrink to -0.015
        shifted_down = swings - 0.005  # losses grow to -0.025

        assert M.volatility(shifted_up) == pytest.approx(M.volatility(shifted_down))
        assert M.sortino(shifted_up) > M.sharpe(shifted_up)
        assert M.sortino(shifted_down) < M.sharpe(shifted_down)

    def test_a_series_that_never_falls_has_no_downside_deviation(self):
        returns = np.full(300, 0.001)
        assert not np.isfinite(M.sortino(returns))

    def test_drawdown_is_measured_from_the_running_peak(self):
        equity = np.array([100.0, 120.0, 90.0, 150.0])
        # The worst point is 90 against a peak of 120: -25%.
        assert M.max_drawdown(equity) == pytest.approx(-0.25)

    def test_drawdown_duration_runs_from_peak_to_recovery(self):
        dates = [date(2020, 1, 1), date(2020, 2, 1), date(2020, 6, 1), date(2020, 7, 1)]
        equity = np.array([100.0, 80.0, 90.0, 110.0])
        out = M.drawdown_duration(equity, dates)
        assert out["start"] == date(2020, 1, 1)
        assert out["trough"] == date(2020, 2, 1)
        assert out["recovered"] == date(2020, 7, 1)
        assert out["days"] == (date(2020, 7, 1) - date(2020, 1, 1)).days

    def test_an_unrecovered_drawdown_is_still_counted(self):
        dates = [date(2020, 1, 1), date(2020, 6, 1), date(2021, 1, 1)]
        equity = np.array([100.0, 70.0, 80.0])
        out = M.drawdown_duration(equity, dates)
        assert out["recovered"] is None
        assert out["days"] > 300

    def test_calmar_is_growth_over_the_worst_fall(self):
        dates = [date(2016, 1, 1), date(2017, 1, 1)]
        equity = np.array([100.0, 110.0])
        # 10% growth against a 0% drawdown is undefined, so give it one.
        dates = [date(2016, 1, 1), date(2016, 6, 1), date(2017, 1, 1)]
        equity = np.array([100.0, 90.0, 110.0])
        assert M.calmar(equity, dates) == pytest.approx(M.cagr(equity, dates) / 0.10, rel=1e-6)


class TestBenchmarkComparison:
    def test_beta_of_a_doubled_benchmark_is_two(self):
        rng = np.random.default_rng(5)
        bench = rng.normal(0.0003, 0.01, 1000)
        strategy = bench * 2.0
        beta, alpha = M.beta_alpha(strategy, bench)
        assert beta == pytest.approx(2.0, rel=1e-6)
        assert alpha == pytest.approx(0.0, abs=1e-6)

    def test_a_constant_daily_edge_becomes_positive_alpha(self):
        rng = np.random.default_rng(6)
        bench = rng.normal(0.0003, 0.01, 1500)
        strategy = bench + 0.0002
        beta, alpha = M.beta_alpha(strategy, bench)
        assert beta == pytest.approx(1.0, rel=1e-6)
        # 0.02% a day compounds to about 5% a year.
        assert alpha == pytest.approx((1.0002**252) - 1.0, rel=0.02)

    def test_tracking_a_benchmark_exactly_leaves_no_tracking_error(self):
        rng = np.random.default_rng(7)
        bench = rng.normal(0.0003, 0.01, 500)
        assert M.tracking_error(bench.copy(), bench) == pytest.approx(0.0)

    def test_information_ratio_rises_as_the_excess_gets_steadier(self):
        rng = np.random.default_rng(8)
        bench = rng.normal(0.0003, 0.01, 2000)
        steady = bench + 0.0002
        noisy = bench + rng.normal(0.0002, 0.004, 2000)
        assert M.information_ratio(steady, bench) > M.information_ratio(noisy, bench)

    def test_capture_of_a_half_weight_position_is_about_half_each_way(self):
        rng = np.random.default_rng(9)
        bench = rng.normal(0.0, 0.012, 2000)
        strategy = bench * 0.5
        up, down = M.capture(strategy, bench)
        assert up == pytest.approx(0.5, rel=0.05)
        assert down == pytest.approx(0.5, rel=0.05)


class TestPeriods:
    def test_monthly_returns_compound_within_the_calendar_month(self):
        dates = [date(2020, 1, 2), date(2020, 1, 31), date(2020, 2, 28)]
        equity = np.array([100.0, 110.0, 121.0])
        keys, values = M.periodic_returns(equity, dates, freq="month")
        assert keys == ["2020-01", "2020-02"]
        assert values[0] == pytest.approx(0.10)
        assert values[1] == pytest.approx(0.10)

    def test_rolling_outperformance_counts_windows_not_days(self):
        n = 21 * 30
        dates = daily_dates(n)
        strategy = np.cumprod(np.full(n, 1.0005))
        bench = np.cumprod(np.full(n, 1.0001))
        out = M.rolling_outperformance(strategy, bench, dates, 12)
        # The strategy grows faster every day, so it wins every window.
        assert out["share"] == pytest.approx(1.0)
        assert out["windows"] == n - 12 * 21


class TestUncertainty:
    def test_a_bootstrap_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(12)
        returns = rng.normal(0.0005, 0.01, 2000)
        point = M.sharpe(returns)
        low, high = M.block_bootstrap_ci(
            returns,
            lambda s: s.mean() / s.std(ddof=1) * math.sqrt(252),
            n_samples=300,
        )
        assert low < point < high

    def test_a_short_series_gets_no_interval_rather_than_a_fake_one(self):
        low, high = M.block_bootstrap_ci(np.array([0.01, -0.01, 0.02]), np.mean)
        assert not np.isfinite(low) and not np.isfinite(high)


class TestDeflation:
    def test_more_trials_raise_the_bar(self):
        rng = np.random.default_rng(13)
        returns = rng.normal(0.0006, 0.01, 2500)
        few = M.deflated_sharpe(returns, n_trials=1)
        many = M.deflated_sharpe(returns, n_trials=500)
        assert many.expected_max_sharpe_annual > few.expected_max_sharpe_annual
        assert many.deflated < few.deflated

    def test_noise_searched_hard_does_not_survive_deflation(self):
        rng = np.random.default_rng(14)
        returns = rng.normal(0.0001, 0.01, 2000)
        out = M.deflated_sharpe(returns, n_trials=200)
        assert out.deflated < 0.95
        assert "not survive" in out.verdict

    def test_a_genuinely_strong_series_survives_a_modest_search(self):
        rng = np.random.default_rng(15)
        returns = rng.normal(0.0016, 0.008, 2500)  # Sharpe near 3
        out = M.deflated_sharpe(returns, n_trials=10)
        assert out.deflated > 0.95

    def test_too_few_observations_returns_a_reason_not_a_number(self):
        out = M.deflated_sharpe(np.array([0.01] * 10), n_trials=5)
        assert not np.isfinite(out.deflated)
        assert "too few" in out.verdict


class TestOverfittingProbability:
    def test_pure_noise_across_trials_gives_a_high_probability(self):
        rng = np.random.default_rng(16)
        trials = rng.normal(0.0, 0.04, (15, 600))
        out = M.probability_of_backtest_overfitting(trials)
        assert out["feasible"]
        assert out["pbo"] > 0.35

    def test_one_genuinely_better_trial_gives_a_low_probability(self):
        rng = np.random.default_rng(17)
        trials = rng.normal(0.0, 0.04, (15, 600))
        trials[0] += 0.03  # one trial is better in every period
        out = M.probability_of_backtest_overfitting(trials)
        assert out["pbo"] < 0.1

    def test_too_few_trials_is_reported_rather_than_computed(self):
        out = M.probability_of_backtest_overfitting(np.zeros((2, 500)))
        assert not out["feasible"]
        assert "trials" in out["reason"]


class TestSummary:
    def test_the_summary_carries_gross_free_statistics_and_its_benchmarks(self):
        n = 21 * 40
        dates = daily_dates(n)
        equity = np.cumprod(np.full(n, 1.0004))
        bench = np.cumprod(np.full(n, 1.0002))
        out = M.summarise(
            dates, equity, benchmarks={"SPY": bench}, n_trials=3, with_intervals=False
        )
        assert out["cagr"] > out["benchmarks"]["SPY"]["cagr"]
        assert out["benchmarks"]["SPY"]["excess_cagr"] == pytest.approx(
            out["cagr"] - out["benchmarks"]["SPY"]["cagr"]
        )
        assert out["positive_months"] == pytest.approx(1.0)
        assert "rolling_12m" in out["benchmarks"]["SPY"]

    def test_a_mismatched_benchmark_is_dropped_rather_than_aligned_by_guesswork(self):
        dates = daily_dates(300)
        equity = np.ones(300)
        out = M.summarise(
            dates, equity, benchmarks={"SPY": np.ones(200)}, with_intervals=False
        )
        assert out["benchmarks"] == {}
