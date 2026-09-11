"""Simulator tests, on portfolios small enough to check with a pencil.

Every expected value in this file is derived in the test itself, in a comment,
from prices the test chose. That is deliberate: a backtest engine verified
against its own output is verified against nothing, and the failures that
matter here -- a dividend counted twice, a split applied to the wrong side of a
trade, a fill at the price that produced the decision -- all look perfectly
plausible in an equity curve.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from spaid.backtest.engine import (
    MISSING_SESSIONS_BEFORE_DELISTING,
    MarketData,
    benchmark_series,
    risk_matched_benchmark,
    simulate,
    target_weights,
)
from spaid.backtest.strategy import PortfolioRules
from spaid.config.validation import CostModel, ExecutionSpec

CAPITAL = 1000.0
FREE = CostModel(half_spread_bps=0.0, slippage_bps=0.0, commission_bps=0.0, min_cost_bps=0.0)
NEXT_DAY = ExecutionSpec(delay_days=1)


def sessions(n: int, start: date = date(2020, 1, 6)) -> list[date]:
    """`n` consecutive weekdays, so the fixture has no weekend gaps to explain."""
    out: list[date] = []
    day = start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def make_prices(
    series: dict[str, list[float | None]],
    days: list[date],
    *,
    dividends: dict[str, dict[int, float]] | None = None,
    splits: dict[str, dict[int, float]] | None = None,
) -> pl.DataFrame:
    """A price table from literal closes, with `None` meaning "did not trade"."""
    dividends = dividends or {}
    splits = splits or {}
    rows = []
    for ticker, closes in series.items():
        for i, close in enumerate(closes):
            if close is None:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "date": days[i],
                    "close_raw": float(close),
                    "close_adj": float(close),
                    "dividend": float(dividends.get(ticker, {}).get(i, 0.0)),
                    "split_ratio": float(splits.get(ticker, {}).get(i, 1.0)),
                }
            )
    return pl.DataFrame(rows)


def make_panel(rows: list[tuple[date, str, float]]) -> pl.DataFrame:
    """A scored panel: one row per decision date per security."""
    return pl.DataFrame(
        [
            {
                "date": d,
                "security_id": s,
                "ticker": s,
                "score": score,
                "rank": i + 1,
                "sector": "Test",
            }
            for i, (d, s, score) in enumerate(rows)
        ]
    )


def run(
    prices: pl.DataFrame,
    days: list[date],
    panel: pl.DataFrame,
    decisions: list[date],
    *,
    rules: PortfolioRules | None = None,
    costs: CostModel = FREE,
    execution: ExecutionSpec = NEXT_DAY,
    tickers: dict[str, str] | None = None,
):
    market = MarketData.build(prices, days)
    securities = sorted(panel["security_id"].unique().to_list())
    return simulate(
        panel,
        market,
        rules or PortfolioRules(label="t", description="test", top_n=len(securities)),
        execution=execution,
        costs=costs,
        decision_dates=decisions,
        start=days[0],
        end=days[-1],
        initial_capital=CAPITAL,
        security_tickers=tickers or {s: s for s in securities},
        sectors={s: "Test" for s in securities},
    )


class TestExecution:
    def test_a_single_holding_earns_the_price_move(self):
        """Buy 10 shares at 100, the price goes to 110, equity is 1100. Exactly."""
        days = sessions(4)
        prices = make_prices({"A": [100, 100, 110, 110]}, days)
        panel = make_panel([(days[0], "A", 70.0)])
        result = run(prices, days, panel, [days[0]])

        # Day 0: no position yet, so equity is the starting capital.
        assert result.equity[0] == pytest.approx(1000.0)
        # Day 1: the decision from day 0 fills at day 1's close of 100.
        assert result.equity[1] == pytest.approx(1000.0)
        # Day 2: 10 shares at 110.
        assert result.equity[2] == pytest.approx(1100.0)
        assert result.final_equity == pytest.approx(1100.0)

    def test_the_decision_price_is_never_the_fill_price(self):
        """A score computed at day 0's close cannot be traded at day 0's close.

        The price triples on day 1. If the engine filled on the decision date it
        would capture that move; filling the next session, it does not.
        """
        days = sessions(4)
        prices = make_prices({"A": [100, 300, 300, 300]}, days)
        panel = make_panel([(days[0], "A", 70.0)])
        result = run(prices, days, panel, [days[0]])

        # Bought at 300, not 100, so the tripling is missed entirely.
        assert result.final_equity == pytest.approx(1000.0)
        assert result.trades[0].price == pytest.approx(300.0)
        assert result.trades[0].trade_date == days[1]
        assert result.trades[0].decision_date == days[0]

    def test_a_longer_delay_moves_the_fill_further_out(self):
        days = sessions(5)
        prices = make_prices({"A": [100, 110, 120, 120, 120]}, days)
        panel = make_panel([(days[0], "A", 70.0)])

        one = run(prices, days, panel, [days[0]], execution=ExecutionSpec(delay_days=1))
        two = run(prices, days, panel, [days[0]], execution=ExecutionSpec(delay_days=2))

        # Filled at 110: 1000/110 shares, worth 1000/110*120 = 1090.909...
        assert one.final_equity == pytest.approx(1000.0 / 110.0 * 120.0)
        # Filled at 120: no gain left to capture.
        assert two.final_equity == pytest.approx(1000.0)

    def test_a_decision_too_close_to_the_end_never_trades(self):
        """There is no session left to fill on, so nothing is bought."""
        days = sessions(3)
        prices = make_prices({"A": [100, 100, 100]}, days)
        panel = make_panel([(days[2], "A", 70.0)])
        result = run(prices, days, panel, [days[2]])
        assert result.trades == []
        assert result.final_equity == pytest.approx(1000.0)


class TestCosts:
    def test_costs_come_out_of_the_purchase(self):
        """10 bps per side on a full deployment of 1000.

        Cash cannot go negative, so the engine spends `cash / (1 + rate)`:
        1000 / 1.001 = 999.000999 of stock, and 0.999001 of cost. At 110 the
        position is worth 999.000999 / 100 * 110 = 1098.901099.
        """
        days = sessions(4)
        prices = make_prices({"A": [100, 100, 110, 110]}, days)
        panel = make_panel([(days[0], "A", 70.0)])
        costs = CostModel(half_spread_bps=4.0, slippage_bps=5.0, commission_bps=1.0, min_cost_bps=0.0)
        assert costs.one_way_bps == pytest.approx(10.0)

        result = run(prices, days, panel, [days[0]], costs=costs)
        expected_notional = 1000.0 / 1.001
        assert result.trades[0].gross_value == pytest.approx(expected_notional)
        assert result.trades[0].cost == pytest.approx(expected_notional * 0.001)
        assert result.final_equity == pytest.approx(expected_notional / 100.0 * 110.0)

    def test_doubling_the_cost_rate_doubles_what_is_paid(self):
        days = sessions(4)
        prices = make_prices({"A": [100, 100, 100, 100]}, days)
        panel = make_panel([(days[0], "A", 70.0)])
        single = CostModel(half_spread_bps=5.0, slippage_bps=5.0, commission_bps=0.0, min_cost_bps=0.0)

        one = run(prices, days, panel, [days[0]], costs=single)
        two = run(prices, days, panel, [days[0]], costs=single.scaled(2.0))
        assert two.costs_paid.sum() == pytest.approx(one.costs_paid.sum() * 2.0, rel=1e-3)

    def test_a_free_backtest_and_a_charged_one_differ_by_the_costs(self):
        days = sessions(6)
        prices = make_prices({"A": [100, 100, 100, 100, 100, 100]}, days)
        panel = make_panel([(days[0], "A", 70.0)])
        charged = run(prices, days, panel, [days[0]],
                      costs=CostModel(half_spread_bps=10.0, slippage_bps=10.0,
                                      commission_bps=0.0, min_cost_bps=0.0))
        free = run(prices, days, panel, [days[0]])
        assert free.final_equity > charged.final_equity
        assert free.final_equity - charged.final_equity == pytest.approx(
            charged.costs_paid.sum(), rel=1e-6
        )


class TestCorporateActions:
    def test_a_dividend_pays_cash_and_does_not_vanish(self):
        """10 shares, a 2.00 dividend on day 2: 20.00 of cash appears."""
        days = sessions(4)
        prices = make_prices(
            {"A": [100, 100, 100, 100]}, days, dividends={"A": {2: 2.0}}
        )
        panel = make_panel([(days[0], "A", 70.0)])
        result = run(prices, days, panel, [days[0]])

        assert result.cash[1] == pytest.approx(0.0)
        assert result.cash[2] == pytest.approx(20.0)
        assert result.final_equity == pytest.approx(1020.0)

    def test_a_split_is_a_non_event_because_the_series_is_already_restated(self):
        """The provider restates its history for splits, so there is nothing to adjust.

        This is the test that would have caught the bug it was written for. The
        simulator originally multiplied the share count on the ex-date, as one
        would for a genuinely as-traded series. Against a series that is already
        split-adjusted, that turned Amazon's 2022 twenty-for-one into a
        twenty-fold gain out of nothing, and flattered every period containing a
        large split.
        """
        days = sessions(4)
        prices = make_prices({"A": [100, 100, 100, 100]}, days, splits={"A": {2: 2.0}})
        panel = make_panel([(days[0], "A", 70.0)])
        result = run(prices, days, panel, [days[0]])

        assert result.equity[2] == pytest.approx(1000.0)
        assert result.final_equity == pytest.approx(1000.0)

    def test_the_price_series_carries_the_split_so_the_return_is_continuous(self):
        """A restated series shows no jump through a split; the return is the real one."""
        days = sessions(4)
        # Twenty-for-one, already restated: the close simply continues, and the
        # 4% move on day 3 is the only thing that happened economically.
        prices = make_prices({"A": [100, 100, 104, 104]}, days, splits={"A": {2: 20.0}})
        panel = make_panel([(days[0], "A", 70.0)])
        result = run(prices, days, panel, [days[0]])
        assert result.final_equity == pytest.approx(1040.0)


class TestDelisting:
    def test_a_holding_that_stops_trading_is_liquidated_at_its_last_price(self):
        days = sessions(12)
        closes = [100.0] * 3 + [80.0] + [None] * 8
        prices = make_prices({"A": closes}, days)
        panel = make_panel([(days[0], "A", 70.0)])
        result = run(prices, days, panel, [days[0]])

        # Bought 10 shares at 100; the last observed price is 80.
        assert result.equity[3] == pytest.approx(800.0)
        delisting = [t for t in result.trades if t.reason == "delisting"]
        assert len(delisting) == 1
        assert delisting[0].price == pytest.approx(80.0)
        assert delisting[0].trade_date == days[3 + MISSING_SESSIONS_BEFORE_DELISTING]
        assert result.final_equity == pytest.approx(800.0)
        assert result.delistings[0]["treatment"] == "liquidated_at_last_observed_price"

    def test_a_short_gap_is_not_a_delisting(self):
        """One missing bar is a data gap; the position survives it."""
        days = sessions(6)
        prices = make_prices({"A": [100, 100, None, 100, 100, 100]}, days)
        panel = make_panel([(days[0], "A", 70.0)])
        result = run(prices, days, panel, [days[0]])
        assert [t for t in result.trades if t.reason == "delisting"] == []
        assert result.final_equity == pytest.approx(1000.0)

    def test_the_final_value_is_never_silently_zero(self):
        """A liquidation must be recorded with the price it used."""
        days = sessions(12)
        prices = make_prices({"A": [100.0] * 4 + [None] * 8}, days)
        panel = make_panel([(days[0], "A", 70.0)])
        result = run(prices, days, panel, [days[0]])
        assert result.delistings
        assert result.delistings[0]["final_price"] is not None
        assert result.final_equity > 0


class TestRebalancing:
    def test_two_names_are_held_in_equal_weight(self):
        days = sessions(4)
        prices = make_prices({"A": [100, 100, 120, 120], "B": [50, 50, 50, 50]}, days)
        panel = make_panel([(days[0], "A", 70.0), (days[0], "B", 60.0)])
        result = run(prices, days, panel, [days[0]])

        # 500 into each: 5 shares of A, 10 of B. A rises 20%, so equity is 1100.
        assert result.equity[1] == pytest.approx(1000.0)
        assert result.equity[2] == pytest.approx(1100.0)
        holdings = {h["security_id"]: h["target_weight"] for h in result.holdings}
        assert holdings == {"A": pytest.approx(0.5), "B": pytest.approx(0.5)}

    def test_weights_drift_between_rebalances_and_reset_at_one(self):
        """Without a rebalance the winner's weight grows; with one it returns to half."""
        days = sessions(8)
        prices = make_prices(
            {"A": [100, 100, 200, 200, 200, 200, 200, 200],
             "B": [100, 100, 100, 100, 100, 100, 100, 100]},
            days,
        )
        panel = make_panel(
            [(days[0], "A", 70.0), (days[0], "B", 60.0),
             (days[4], "A", 70.0), (days[4], "B", 60.0)]
        )
        result = run(prices, days, panel, [days[0], days[4]])

        # After A doubles, equity is 1500 and A is two thirds of it. The second
        # rebalance sells A back to half: 750 each.
        assert result.equity[2] == pytest.approx(1500.0)
        second = [h for h in result.holdings if h["rebalance_date"] == days[4]]
        by_id = {h["security_id"]: h for h in second}
        assert by_id["A"]["value"] == pytest.approx(750.0)
        assert by_id["B"]["value"] == pytest.approx(750.0)

    def test_turnover_counts_both_sides_of_the_trade(self):
        days = sessions(8)
        prices = make_prices(
            {"A": [100] * 8, "B": [100] * 8}, days
        )
        # Hold A, then swap entirely into B.
        panel = pl.concat(
            [make_panel([(days[0], "A", 70.0)]), make_panel([(days[4], "B", 70.0)])]
        )
        result = run(
            prices, days, panel, [days[0], days[4]],
            rules=PortfolioRules(label="t", description="one name", top_n=1),
        )
        # The swap sells 1000 and buys 1000 against equity of 1000: 2.0.
        assert result.turnover[result.turnover > 0].max() == pytest.approx(2.0, rel=1e-6)

    def test_an_unpriced_target_is_not_bought(self):
        """A name with no bar on the fill date is skipped, not filled at a guess."""
        days = sessions(4)
        prices = make_prices({"A": [100, None, 100, 100], "B": [50, 50, 50, 50]}, days)
        panel = make_panel([(days[0], "A", 90.0), (days[0], "B", 60.0)])
        result = run(prices, days, panel, [days[0]])
        bought = {t.security_id for t in result.trades}
        assert bought == {"B"}
        # Only half the capital was deployable, so the rest sat in cash.
        assert result.cash[1] == pytest.approx(500.0)


class TestTargetWeights:
    def test_ties_are_broken_deterministically(self):
        """Tied scores must not be ordered by whatever the frame happens to hold."""
        rows = [{"security_id": s, "score": 50.0, "sector": "X"} for s in "DCBA"]
        frame = pl.DataFrame(rows)
        rules = PortfolioRules(label="t", description="", top_n=2)
        first = target_weights(frame, rules, n_eligible=4)
        second = target_weights(frame.reverse(), rules, n_eligible=4)
        assert first == second
        assert set(first) == {"A", "B"}

    def test_the_top_decile_scales_with_the_universe(self):
        rules = PortfolioRules(label="d", description="", top_fraction=0.10)
        assert rules.size_for(500) == 50
        assert rules.size_for(43) == 4

    def test_a_sector_cap_limits_how_many_names_one_sector_supplies(self):
        frame = pl.DataFrame(
            [
                {"security_id": f"T{i}", "score": 100.0 - i, "sector": "Tech" if i < 8 else "Other"}
                for i in range(12)
            ]
        )
        rules = PortfolioRules(label="c", description="", top_n=4, max_sector_weight=0.5)
        chosen = target_weights(frame, rules, n_eligible=12)
        sectors = frame.filter(pl.col("security_id").is_in(list(chosen)))["sector"].to_list()
        assert sectors.count("Tech") <= 2

    def test_score_proportional_weighting_tilts_toward_the_best(self):
        frame = pl.DataFrame(
            [
                {"security_id": "A", "score": 80.0, "sector": "X"},
                {"security_id": "B", "score": 70.0, "sector": "X"},
                {"security_id": "C", "score": 60.0, "sector": "X"},
            ]
        )
        rules = PortfolioRules(label="p", description="", top_n=3, weighting="score_proportional")
        weights = target_weights(frame, rules, n_eligible=3)
        assert weights["A"] > weights["B"] > weights["C"]
        assert sum(weights.values()) == pytest.approx(1.0)


class TestBenchmarks:
    def test_a_benchmark_index_is_rebased_to_one(self):
        days = sessions(4)
        prices = make_prices({"SPY": [400, 404, 408, 412]}, days)
        market = MarketData.build(prices, days)
        index = benchmark_series(market, "SPY", days)
        assert index[0] == pytest.approx(1.0)
        assert index[-1] == pytest.approx(412.0 / 400.0)

    def test_an_unknown_benchmark_returns_nothing_rather_than_zeros(self):
        days = sessions(4)
        prices = make_prices({"SPY": [400, 404, 408, 412]}, days)
        market = MarketData.build(prices, days)
        assert benchmark_series(market, "NOPE", days) is None

    def test_risk_matching_scales_the_benchmark_to_the_strategy_volatility(self):
        rng = np.random.default_rng(3)
        bench_returns = rng.normal(0.0004, 0.008, 750)
        index = np.concatenate([[1.0], np.cumprod(1.0 + bench_returns)])
        # A strategy twice as volatile should scale the benchmark by about two.
        strategy_returns = bench_returns * 2.0
        scaled, scale = risk_matched_benchmark(index, strategy_returns)
        assert scale == pytest.approx(2.0, rel=0.05)
        scaled_vol = np.std(np.diff(scaled) / scaled[:-1], ddof=1)
        assert scaled_vol == pytest.approx(np.std(strategy_returns, ddof=1), rel=0.05)


class TestCashDrag:
    def test_uninvested_cash_earns_the_configured_rate_and_no_more(self):
        """The default is zero, so cash is a drag rather than a free bond."""
        days = sessions(4)
        prices = make_prices({"A": [100, 100, 100, 100]}, days)
        panel = make_panel([(days[0], "A", 70.0)])
        result = run(
            prices, days, panel, [days[0]],
            rules=PortfolioRules(label="half", description="", top_n=1,
                                 max_position_weight=0.5),
        )
        # Half deployed, half in cash earning nothing.
        assert result.cash[-1] == pytest.approx(500.0)
        assert result.final_equity == pytest.approx(1000.0)
