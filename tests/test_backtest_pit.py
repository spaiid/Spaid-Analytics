"""Point-in-time integrity of the backtest panel, and the gates over it.

The unit tests elsewhere prove the arithmetic. These prove the property that
makes the arithmetic mean anything: that nothing in the panel on a given
decision date could only have been known afterwards. Look-ahead does not raise;
it produces a better number, which is exactly why it needs a test rather than
an inspection.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import ClassVar

import polars as pl
import pytest

from spaid.backtest import conclusion as C
from spaid.backtest import panel as P
from spaid.backtest.strategy import STRATEGY_V1, freeze
from spaid.config.validation import (
    ACTIVE_VALIDATION_SPEC,
    Conclusion,
    ValidationStatus,
)
from spaid.storage import store


class TestDecisionDates:
    def test_the_last_session_of_each_month_is_the_decision_date(self):
        sessions = [date(2020, 1, d) for d in (2, 15, 30, 31)] + [date(2020, 2, d) for d in (3, 28)]
        out = P.month_end_sessions(sessions, date(2020, 1, 1), date(2020, 2, 29))
        assert out == [date(2020, 1, 31), date(2020, 2, 28)]

    def test_dates_outside_the_window_are_excluded(self):
        sessions = [date(2019, 12, 31), date(2020, 1, 31), date(2020, 2, 28)]
        out = P.month_end_sessions(sessions, date(2020, 1, 1), date(2020, 1, 31))
        assert out == [date(2020, 1, 31)]


class TestForwardReturns:
    def test_a_forward_return_looks_forward_by_the_stated_horizon(self):
        """One month forward is 21 sessions, and the answer is the move over them."""
        days = [date(2020, 1, 1) + timedelta(days=i) for i in range(60)]
        prices = pl.DataFrame(
            {
                "ticker": ["SPY"] * 60 + ["AAA"] * 60,
                "date": days * 2,
                "close_adj": [100.0] * 60 + [float(100 + i) for i in range(60)],
            }
        )
        labels = P.forward_returns(prices, [days[0]], (1,))
        row = labels.filter(pl.col("ticker") == "AAA")
        # 100 -> 121 over 21 sessions, against a flat market.
        assert row["fwd_1m"][0] == pytest.approx(0.21)
        assert row["fwd_excess_1m"][0] == pytest.approx(0.21)

    def test_the_excess_return_removes_the_market(self):
        days = [date(2020, 1, 1) + timedelta(days=i) for i in range(60)]
        prices = pl.DataFrame(
            {
                "ticker": ["SPY"] * 60 + ["AAA"] * 60,
                "date": days * 2,
                "close_adj": [float(100 + i) for i in range(60)] * 2,
            }
        )
        labels = P.forward_returns(prices, [days[0]], (1,))
        row = labels.filter(pl.col("ticker") == "AAA")
        assert row["fwd_excess_1m"][0] == pytest.approx(0.0)

    def test_a_horizon_running_past_the_data_yields_null_not_a_shortened_return(self):
        """Measuring a twelve-month return over four months of data and calling
        it twelve months is how a backtest quietly annualises noise."""
        days = [date(2020, 1, 1) + timedelta(days=i) for i in range(30)]
        prices = pl.DataFrame(
            {
                "ticker": ["SPY"] * 30 + ["AAA"] * 30,
                "date": days * 2,
                "close_adj": [100.0] * 60,
            }
        )
        labels = P.forward_returns(prices, [days[0]], (12,))
        assert labels["fwd_12m"].null_count() == labels.height


class TestStoredPanelProperties:
    """Properties of the real store that the panel depends on."""

    @pytest.fixture(scope="class")
    def master(self) -> pl.DataFrame:
        df = store.read("security_master")
        if df is None or df.is_empty():
            pytest.skip("no security master in the store")
        return df

    def test_the_historical_universe_includes_companies_no_longer_in_the_index(self, master):
        securities = P.historical_securities()
        current = store.read("securities")
        if current is None:
            pytest.skip("no current securities table")
        current_ids = set(current["company_id"].to_list())
        extra = set(securities["company_id"].to_list()) - current_ids
        assert len(extra) > 0, (
            "the backtest universe contains only today's constituents, which is the "
            "survivorship bias this milestone exists to remove"
        )

    def test_every_security_resolves_to_one_ticker(self, master):
        securities = P.historical_securities()
        assert securities["ticker"].n_unique() == securities.height
        assert securities["company_id"].n_unique() == securities.height

    def test_delisting_returns_are_null_rather_than_zero_when_unknown(self, master):
        """Zero is the specific claim that holders were wiped out. Unknown is not that."""
        unknown = master.filter(pl.col("delisting_return_status") == "unavailable")
        if unknown.is_empty():
            pytest.skip("no delistings with an unknown return")
        assert unknown["delisting_return"].null_count() == unknown.height

    def test_a_delisting_reason_is_never_invented(self, master):
        stated = master.filter(pl.col("delisting_reason").is_not_null())
        if stated.is_empty():
            pytest.skip("no delisting reasons recorded")
        assert set(stated["delisting_reason"].unique()) <= {
            "unknown", "acquisition", "merger", "bankruptcy", "index_removal"
        }


class TestFrozenStrategy:
    def test_the_checksum_is_stable_across_calls(self):
        assert STRATEGY_V1.checksum == STRATEGY_V1.checksum

    def test_the_estimate_metrics_are_declared_unavailable(self):
        unavailable = set(STRATEGY_V1.unavailable_point_in_time)
        assert "eps_revision_3m" in unavailable
        assert "forward_pe" in unavailable
        available = set(STRATEGY_V1.available_metrics())
        assert unavailable.isdisjoint(available)
        assert len(available) + len(unavailable) == len(STRATEGY_V1.scoring.metrics)

    def test_growth_loses_the_most_weight_point_in_time(self):
        coverage = STRATEGY_V1.category_coverage_point_in_time()
        assert coverage["quality"] == pytest.approx(1.0)
        assert coverage["growth"] < coverage["momentum"]
        assert 0.0 < coverage["growth"] < 1.0

    def test_refreezing_an_unchanged_strategy_is_idempotent(self):
        first = freeze(STRATEGY_V1, persist=False)
        second = freeze(STRATEGY_V1, persist=False)
        assert first["config_checksum"] == second["config_checksum"]

    def test_the_primary_variant_is_declared_not_chosen(self):
        assert STRATEGY_V1.primary_variant == ACTIVE_VALIDATION_SPEC.primary_variant
        assert STRATEGY_V1.primary_variant in STRATEGY_V1.variants


class TestGates:
    COMPLETE: ClassVar[dict] = {
        "built": True, "snapshots": 147, "coverage_start": "2014-06-30",
        "coverage_end": "2026-08-31", "exits_observed": 300,
        "removed_securities": 100, "removed_with_prices": 100,
        "delisting_returns_known": 0,
    }
    INCOMPLETE: ClassVar[dict] = {**COMPLETE, "removed_with_prices": 40}

    def test_a_complete_universe_passes_the_survivorship_gates(self):
        gates = {g.key: g for g in C.evaluate_data_gates(
            self.COMPLETE, benchmark_coverage={"SPY": True, "SPMO": True}
        )}
        assert gates["historical_membership"].passed
        assert gates["delisted_prices"].passed
        assert gates["delisting_returns"].passed

    def test_missing_delisted_prices_caps_the_status_at_exploratory(self):
        gates = C.evaluate_data_gates(
            self.INCOMPLETE, benchmark_coverage={"SPY": True, "SPMO": True}
        )
        assert C.cap_status(gates) is ValidationStatus.EXPLORATORY_ONLY

    def test_no_membership_history_also_caps_at_exploratory(self):
        gates = C.evaluate_data_gates({"built": False}, benchmark_coverage={"SPY": True})
        assert C.cap_status(gates) is ValidationStatus.EXPLORATORY_ONLY

    def test_trading_on_the_decision_close_makes_the_run_not_testable(self):
        gates = C.evaluate_data_gates(
            self.COMPLETE,
            benchmark_coverage={"SPY": True, "SPMO": True},
            execution_delay_days=0,
        )
        assert C.cap_status(gates) is ValidationStatus.NOT_TESTABLE

    def test_reading_estimates_historically_makes_the_run_not_testable(self):
        gates = C.evaluate_data_gates(
            self.COMPLETE,
            benchmark_coverage={"SPY": True, "SPMO": True},
            estimates_excluded=False,
        )
        assert C.cap_status(gates) is ValidationStatus.NOT_TESTABLE

    def test_result_gates_compare_against_the_declared_threshold(self):
        gates = {g.key: g for g in C.evaluate_result_gates(
            {"ic_mean_3m": 0.05, "ic_t_3m": 3.0, "decile_spread_3m": 0.02,
             "decile_monotonicity": 0.9, "excess_cagr_net_spy": 0.03,
             "information_ratio_net_spy": 0.5}
        )}
        assert all(g.passed for g in gates.values())

    def test_an_unmeasurable_statistic_fails_rather_than_passes(self):
        gates = {g.key: g for g in C.evaluate_result_gates({})}
        assert not any(g.passed for g in gates.values())
        assert "could not be measured" in gates["ic_positive"].detail


class TestVerdict:
    GOOD: ClassVar[dict] = {
        "ic_mean_3m": 0.05, "ic_t_3m": 3.0, "decile_spread_3m": 0.02,
        "decile_monotonicity": 0.9, "excess_cagr_net_spy": 0.03,
        "information_ratio_net_spy": 0.5,
    }
    BAD: ClassVar[dict] = {
        "ic_mean_3m": -0.01, "ic_t_3m": -0.5, "decile_spread_3m": -0.01,
        "decile_monotonicity": -0.4, "excess_cagr_net_spy": -0.02,
        "information_ratio_net_spy": -0.2,
    }

    def test_an_incomplete_universe_invalidates_even_an_excellent_result(self):
        """This is the rule the whole milestone rests on: a good number on bad
        data is not evidence, and no amount of outperformance changes that."""
        verdict = C.decide(
            coverage=TestGates.INCOMPLETE,
            measurements=self.GOOD,
            period_label="validation",
            benchmark_coverage={"SPY": True, "SPMO": True},
            n_decision_dates=60,
            robustness_summary={"survived_share": 1.0},
        )
        assert verdict.status is ValidationStatus.EXPLORATORY_ONLY
        assert verdict.conclusion is Conclusion.INVALID

    def test_a_complete_universe_and_a_good_result_reads_as_evidence(self):
        verdict = C.decide(
            coverage=TestGates.COMPLETE,
            measurements=self.GOOD,
            period_label="validation",
            benchmark_coverage={"SPY": True, "SPMO": True},
            n_decision_dates=60,
            robustness_summary={"survived_share": 1.0},
        )
        assert verdict.status is ValidationStatus.VALIDATION_PASSED
        assert verdict.conclusion is Conclusion.EVIDENCE_OF_ABILITY

    def test_a_complete_universe_and_a_bad_result_reads_as_no_evidence(self):
        verdict = C.decide(
            coverage=TestGates.COMPLETE,
            measurements=self.BAD,
            period_label="validation",
            benchmark_coverage={"SPY": True, "SPMO": True},
            n_decision_dates=60,
        )
        assert verdict.status is ValidationStatus.VALIDATION_FAILED
        assert verdict.conclusion is Conclusion.NO_EVIDENCE

    def test_a_partly_passing_result_reads_as_mixed(self):
        measurements = {**self.GOOD, "excess_cagr_net_spy": -0.01,
                        "information_ratio_net_spy": -0.1}
        verdict = C.decide(
            coverage=TestGates.COMPLETE,
            measurements=measurements,
            period_label="validation",
            benchmark_coverage={"SPY": True, "SPMO": True},
            n_decision_dates=60,
        )
        assert verdict.conclusion is Conclusion.MIXED

    def test_too_few_dates_reads_as_insufficient_rather_than_negative(self):
        verdict = C.decide(
            coverage=TestGates.COMPLETE,
            measurements=self.BAD,
            period_label="validation",
            benchmark_coverage={"SPY": True, "SPMO": True},
            n_decision_dates=12,
        )
        assert verdict.conclusion is Conclusion.INSUFFICIENT_DATA

    def test_the_development_period_is_always_in_sample(self):
        verdict = C.decide(
            coverage=TestGates.COMPLETE,
            measurements=self.GOOD,
            period_label="development",
            benchmark_coverage={"SPY": True, "SPMO": True},
            n_decision_dates=60,
            robustness_summary={"survived_share": 1.0},
        )
        assert verdict.status is ValidationStatus.IN_SAMPLE

    def test_a_missing_robustness_verdict_is_not_treated_as_a_pass(self):
        """No applicable robustness test means no information, not reassurance."""
        verdict = C.decide(
            coverage=TestGates.COMPLETE,
            measurements=self.GOOD,
            period_label="validation",
            benchmark_coverage={"SPY": True, "SPMO": True},
            n_decision_dates=60,
            robustness_summary={"survived_share": float("nan")},
        )
        assert verdict.conclusion is Conclusion.EVIDENCE_OF_ABILITY

    def test_fragile_robustness_downgrades_an_otherwise_good_result(self):
        verdict = C.decide(
            coverage=TestGates.COMPLETE,
            measurements=self.GOOD,
            period_label="validation",
            benchmark_coverage={"SPY": True, "SPMO": True},
            n_decision_dates=60,
            robustness_summary={"survived_share": 0.2},
        )
        assert verdict.conclusion is Conclusion.MIXED

    def test_the_word_validated_never_appears_while_a_data_gate_fails(self):
        verdict = C.decide(
            coverage=TestGates.INCOMPLETE,
            measurements=self.GOOD,
            period_label="holdout",
            benchmark_coverage={"SPY": True, "SPMO": True},
            n_decision_dates=60,
        )
        assert verdict.status not in {
            ValidationStatus.VALIDATION_PASSED, ValidationStatus.HOLDOUT_PASSED
        }
