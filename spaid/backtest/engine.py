"""The simulator: a day-by-day reconstruction of what the strategy would have done.

Deliberately literal. There is no vectorised shortcut that computes the equity
curve from a weight matrix, because those shortcuts are where backtests acquire
their optimism: they hold positions that had stopped trading, rebalance for
free, and reinvest dividends on the day they are declared. This loop walks the
trading calendar one session at a time and does only what could actually have
been done on that session.

The rules, each of which costs the strategy return and each of which is the
reason the number can be believed:

**Decide on one day, trade on another.** Scores are computed from the close of
the decision date. The earliest possible fill is the next session's close, and
the delay is configurable so the robustness battery can widen it. A backtest
that rebalances at the same close it scored on has quietly bought at a price it
learned from.

**Trade on the split basis the data is actually stated in.** The provider
returns both close series restated onto today's share basis -- Nvidia's June
2015 close reads $0.56, not the $22 it traded at, because of the forty-fold of
splits since. `close_raw` and `close_adj` therefore differ only by dividends,
not by splits. So share counts here are on today's basis throughout and a split
is a non-event for the simulator: adjusting the count on the ex-date, as one
would for a genuinely as-traded series, would have multiplied the position by
twenty when Amazon split in 2022 and invented the money to do it.

`close_raw` still matters, because it is not dividend-adjusted: dividends are
paid into cash on the ex-date, in per-share amounts stated on the same basis.
Marking to `close_adj` instead would reinvest every dividend into the same stock
on the day it went ex, which nobody can do.

**Cash is cash.** Dividends land in it, costs come out of it, and it earns the
configured rate, which is zero by default. That is the cash drag, and it is
modelled rather than assumed away.

**A holding that stops trading is liquidated, and the assumption is recorded.**
When a position's price series ends, the position is closed at the last price
actually observed and a trade is written with reason `delisting`. Where the
security master knows the real delisting return it is used instead. Where it
does not -- which is currently everywhere -- the run carries a count of these
events so that no reader can mistake "sold at the last quote" for "we know what
holders received".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from spaid.backtest.strategy import PortfolioRules
from spaid.config.settings import SETTINGS
from spaid.config.validation import CostModel, ExecutionSpec

log = logging.getLogger(__name__)

# A position whose price has been missing this many consecutive sessions is
# treated as delisted rather than as a data gap. Five sessions is a week: long
# enough that an ordinary missing bar does not trigger a sale, short enough that
# a dead holding is not carried at a stale price for months.
MISSING_SESSIONS_BEFORE_DELISTING = 5


@dataclass
class Trade:
    decision_date: date | None
    trade_date: date
    security_id: str
    ticker: str
    side: str
    shares: float
    price: float
    gross_value: float
    cost: float
    reason: str


@dataclass
class SimulationResult:
    """One pass of the simulator: the equity path and everything that produced it."""

    dates: list[date]
    equity: np.ndarray
    cash: np.ndarray
    invested: np.ndarray
    n_positions: np.ndarray
    daily_return: np.ndarray
    costs_paid: np.ndarray
    # Traded notional on *both* sides as a fraction of equity: a full
    # replacement of the portfolio reads 2.0, not 1.0. Stated this way because
    # it is what the cost model charges against, and halving it to quote the
    # conventional one-way figure would understate what was actually traded.
    turnover: np.ndarray
    is_rebalance: np.ndarray
    trades: list[Trade]
    holdings: list[dict]  # one row per position per rebalance
    delistings: list[dict]
    contribution: dict[str, float]  # profit in currency, by security
    sector_contribution: dict[str, float]

    @property
    def final_equity(self) -> float:
        return float(self.equity[-1])


@dataclass
class MarketData:
    """Price matrices aligned to one calendar, indexed by ticker.

    Built once and reused across every variant and robustness run, because
    assembling it costs more than a simulation does.
    """

    sessions: list[date]
    tickers: list[str]
    close_raw: np.ndarray
    close_adj: np.ndarray
    dividend: np.ndarray
    session_index: dict[date, int] = field(default_factory=dict)
    ticker_index: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, prices: pl.DataFrame, sessions: list[date]) -> MarketData:
        tickers = sorted(prices["ticker"].unique().to_list())
        t_index = {t: i for i, t in enumerate(tickers)}
        s_index = {d: i for i, d in enumerate(sessions)}

        shape = (len(sessions), len(tickers))
        close_raw = np.full(shape, np.nan)
        close_adj = np.full(shape, np.nan)
        dividend = np.zeros(shape)

        wanted = prices.filter(pl.col("date").is_in(sessions)).select(
            ["date", "ticker", "close_raw", "close_adj", "dividend"]
        )
        for d, t, cr, ca, div in wanted.rows():
            i, j = s_index[d], t_index[t]
            close_raw[i, j] = cr if cr is not None else np.nan
            close_adj[i, j] = ca if ca is not None else np.nan
            if div:
                dividend[i, j] = div

        log.info(
            "market data: %d sessions x %d tickers, %.1f%% of cells priced",
            len(sessions), len(tickers), 100.0 * np.isfinite(close_raw).mean(),
        )
        return cls(
            sessions=sessions,
            tickers=tickers,
            close_raw=close_raw,
            close_adj=close_adj,
            dividend=dividend,
            session_index=s_index,
            ticker_index=t_index,
        )


# ---------------------------------------------------------------------------
# Target weights
# ---------------------------------------------------------------------------


def target_weights(
    ranked: pl.DataFrame,
    rules: PortfolioRules,
    *,
    n_eligible: int,
) -> dict[str, float]:
    """Which securities to hold on a decision date, and in what proportion.

    `ranked` is already filtered to the eligible, scored universe for one date
    and sorted best first. Ties are broken by security identifier, not by
    whatever order the frame happens to be in: an unstable sort on tied scores
    is how the previous build produced a 21% annual return for an
    alphabetically-ordered basket.
    """
    if ranked.is_empty():
        return {}

    size = rules.size_for(n_eligible)
    if size <= 0:
        return {}

    ordered = ranked.sort(["score", "security_id"], descending=[True, False])

    if rules.max_sector_weight is not None:
        ordered = _apply_sector_cap(ordered, size, rules.max_sector_weight)

    chosen = ordered.head(size)
    if chosen.is_empty():
        return {}

    if rules.weighting == "score_proportional":
        # Proportional to the score's distance above the selected set's floor,
        # so that twenty scores clustered between 60 and 66 do not produce
        # twenty near-identical weights that merely pretend to be a tilt.
        scores = chosen["score"].to_numpy().astype(float)
        floor = scores.min()
        raw = scores - floor
        if raw.sum() <= 0:
            raw = np.ones_like(scores)
        weights = raw / raw.sum()
    else:
        weights = np.full(chosen.height, 1.0 / chosen.height)

    if rules.max_position_weight is not None:
        # Cap and stop. Renormalising back to one after a cap hands the capped
        # weight straight back to the names that were capped, which cancels the
        # limit while appearing to honour it. What the cap leaves over stays in
        # cash, and the cash drag that follows is a real consequence of the rule.
        weights = np.minimum(weights, rules.max_position_weight)
        total = weights.sum()
        if total > 1.0:
            weights = weights / total

    return dict(zip(chosen["security_id"].to_list(), weights.tolist(), strict=True))


def _apply_sector_cap(ordered: pl.DataFrame, size: int, cap: float) -> pl.DataFrame:
    """Drop the lowest-ranked names in any sector that would exceed the cap.

    Applied by count rather than by weight because the portfolios here are
    equally weighted, so a cap on the number of names in a sector *is* a cap on
    its weight, and doing it by count keeps the remaining positions equal rather
    than introducing an implicit optimisation.
    """
    max_names = max(1, int(size * cap))
    if "sector" not in ordered.columns:
        return ordered
    return (
        ordered.with_columns(
            pl.col("score").rank("ordinal", descending=True).over("sector").alias("_sector_rank")
        )
        .filter(pl.col("_sector_rank") <= max_names)
        .drop("_sector_rank")
    )


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------


def simulate(
    panel: pl.DataFrame,
    market: MarketData,
    rules: PortfolioRules,
    *,
    execution: ExecutionSpec,
    costs: CostModel,
    decision_dates: list[date],
    start: date,
    end: date,
    initial_capital: float = 1_000_000.0,
    security_tickers: dict[str, str] | None = None,
    sectors: dict[str, str] | None = None,
    excluded_securities: frozenset[str] = frozenset(),
) -> SimulationResult:
    """Walk the calendar and return the equity path.

    `excluded_securities` supports the "remove the biggest winners" robustness
    test. It is applied at selection time, so the capital that would have gone
    into the excluded name goes into the next-ranked one rather than sitting in
    cash -- which is the question being asked ("was it only those names?"),
    rather than a different one ("what if the portfolio had been smaller?").
    """
    security_tickers = security_tickers or {}
    sectors = sectors or {}

    sessions = [d for d in market.sessions if start <= d <= end]
    if not sessions:
        raise ValueError(f"no trading sessions between {start} and {end}")
    first_i = market.session_index[sessions[0]]

    decisions = sorted(d for d in decision_dates if start <= d <= end)
    # A decision is executed `delay` sessions later. Decisions whose execution
    # date falls past the end of the window simply never trade, which is what
    # would happen in life.
    execute_on: dict[date, date] = {}
    for d in decisions:
        i = market.session_index.get(d)
        if i is None:
            continue
        j = i + execution.delay_days
        if j < len(market.sessions) and market.sessions[j] <= end:
            execute_on[market.sessions[j]] = d

    # The caller supplies the eligible, scored universe: deciding who may be
    # held is the panel layer's job, not the simulator's. An `is_eligible`
    # column is honoured if present, as a guard against passing the wrong frame.
    selectable = panel.filter(pl.col("score").is_not_null())
    if "is_eligible" in selectable.columns:
        selectable = selectable.filter(pl.col("is_eligible"))
    by_date: dict[date, pl.DataFrame] = {
        d: g.drop("date")
        for (d,), g in selectable.group_by(["date"], maintain_order=True)
    }

    cost_rate = costs.one_way_bps / 1e4
    daily_cash_rate = (1.0 + execution.cash_return_annual) ** (
        1.0 / SETTINGS.trading_days_per_year
    ) - 1.0

    shares: dict[str, float] = {}
    last_price: dict[str, float] = {}
    missing_streak: dict[str, int] = {}
    cost_basis: dict[str, float] = {}
    contribution: dict[str, float] = {}

    cash = initial_capital
    equity_path: list[float] = []
    cash_path: list[float] = []
    invested_path: list[float] = []
    count_path: list[int] = []
    return_path: list[float] = []
    cost_path: list[float] = []
    turnover_path: list[float] = []
    rebalance_path: list[bool] = []

    trades: list[Trade] = []
    holdings: list[dict] = []
    delistings: list[dict] = []
    previous_equity = initial_capital

    for offset, day in enumerate(sessions):
        i = first_i + offset
        day_costs = 0.0
        day_turnover = 0.0

        # ---- dividends, before anything else happens today ----------------
        # No split adjustment: see the module docstring. Both price series and
        # the dividend amounts are stated on today's share basis, so the share
        # count never changes on an ex-split date and the position's value is
        # continuous through it.
        for security_id in list(shares):
            ticker = security_tickers.get(security_id)
            j = market.ticker_index.get(ticker) if ticker else None
            if j is None:
                continue
            div = market.dividend[i, j]
            if div:
                cash += shares[security_id] * div

        cash *= 1.0 + daily_cash_rate

        # ---- delistings ----------------------------------------------------
        for security_id in list(shares):
            ticker = security_tickers.get(security_id)
            j = market.ticker_index.get(ticker) if ticker else None
            price = market.close_raw[i, j] if j is not None else np.nan
            if np.isfinite(price):
                last_price[security_id] = float(price)
                missing_streak[security_id] = 0
                continue
            streak = missing_streak.get(security_id, 0) + 1
            missing_streak[security_id] = streak
            if streak < MISSING_SESSIONS_BEFORE_DELISTING:
                continue

            final = last_price.get(security_id)
            held = shares.pop(security_id)
            if final is None:
                # Never priced at all: nothing can be assumed about what it was
                # worth, so the position is dropped and the event recorded as a
                # loss of information rather than a return of any size.
                delistings.append(
                    {
                        "date": day, "security_id": security_id, "ticker": ticker,
                        "shares": held, "final_price": None,
                        "treatment": "written_off_unpriced",
                    }
                )
                continue
            proceeds = held * final
            cost = abs(proceeds) * cost_rate
            cash += proceeds - cost
            day_costs += cost
            contribution[security_id] = contribution.get(security_id, 0.0) + (
                proceeds - cost_basis.get(security_id, 0.0)
            )
            trades.append(
                Trade(None, day, security_id, ticker or "", "sell", held, final,
                      proceeds, cost, "delisting")
            )
            delistings.append(
                {
                    "date": day, "security_id": security_id, "ticker": ticker,
                    "shares": held, "final_price": final,
                    "treatment": "liquidated_at_last_observed_price",
                }
            )

        # ---- rebalance ------------------------------------------------------
        decision_date = execute_on.get(day)
        if decision_date is not None:
            ranked = by_date.get(decision_date)
            if ranked is not None and not ranked.is_empty():
                if excluded_securities:
                    ranked = ranked.filter(
                        ~pl.col("security_id").is_in(list(excluded_securities))
                    )
                n_eligible = ranked.height
                targets = target_weights(ranked, rules, n_eligible=n_eligible)

                priced = {}
                for security_id in targets:
                    ticker = security_tickers.get(security_id)
                    j = market.ticker_index.get(ticker) if ticker else None
                    p = market.close_raw[i, j] if j is not None else np.nan
                    if np.isfinite(p) and p > 0:
                        priced[security_id] = float(p)
                # A target we cannot price on the execution day is not bought,
                # and its weight is not redistributed: pretending to fill at an
                # unobserved price is exactly the kind of small lie that makes a
                # backtest unfalsifiable.
                targets = {k: v for k, v in targets.items() if k in priced}

                equity_now = cash + sum(
                    shares.get(s, 0.0) * _price_or_last(market, i, security_tickers.get(s), last_price.get(s))
                    for s in shares
                )

                # sells first, so the cash they raise funds the buys
                for security_id in list(shares):
                    ticker = security_tickers.get(security_id)
                    j = market.ticker_index.get(ticker) if ticker else None
                    p = market.close_raw[i, j] if j is not None else np.nan
                    if not np.isfinite(p) or p <= 0:
                        continue
                    target_value = targets.get(security_id, 0.0) * equity_now
                    current_value = shares[security_id] * p
                    if current_value - target_value <= 0:
                        continue
                    sell_value = current_value - target_value
                    sell_shares = sell_value / p
                    cost = sell_value * cost_rate
                    cash += sell_value - cost
                    day_costs += cost
                    day_turnover += sell_value
                    shares[security_id] -= sell_shares
                    realised_basis = cost_basis.get(security_id, 0.0) * (
                        sell_shares / (shares[security_id] + sell_shares)
                        if (shares[security_id] + sell_shares) > 0
                        else 1.0
                    )
                    cost_basis[security_id] = cost_basis.get(security_id, 0.0) - realised_basis
                    contribution[security_id] = contribution.get(security_id, 0.0) + (
                        sell_value - cost - realised_basis
                    )
                    trades.append(
                        Trade(decision_date, day, security_id, ticker or "", "sell",
                              sell_shares, float(p), sell_value, cost, "rebalance")
                    )
                    if shares[security_id] <= 1e-9:
                        del shares[security_id]

                for security_id, weight in targets.items():
                    p = priced[security_id]
                    ticker = security_tickers.get(security_id, "")
                    target_value = weight * equity_now
                    current_value = shares.get(security_id, 0.0) * p
                    buy_value = target_value - current_value
                    if buy_value <= 1e-6:
                        continue
                    cost = buy_value * cost_rate
                    if buy_value + cost > cash:
                        # Never borrow. The shortfall is a real consequence of
                        # costs and of prices moving between decision and fill.
                        buy_value = max(0.0, cash / (1.0 + cost_rate))
                        cost = buy_value * cost_rate
                    if buy_value <= 1e-6:
                        continue
                    buy_shares = buy_value / p
                    cash -= buy_value + cost
                    day_costs += cost
                    day_turnover += buy_value
                    shares[security_id] = shares.get(security_id, 0.0) + buy_shares
                    cost_basis[security_id] = cost_basis.get(security_id, 0.0) + buy_value + cost
                    last_price[security_id] = p
                    missing_streak[security_id] = 0
                    trades.append(
                        Trade(decision_date, day, security_id, ticker, "buy",
                              buy_shares, p, buy_value, cost, "rebalance")
                    )

                for security_id in sorted(shares):
                    ticker = security_tickers.get(security_id, "")
                    p = last_price.get(security_id, float("nan"))
                    value = shares[security_id] * p
                    row = ranked.filter(pl.col("security_id") == security_id)
                    holdings.append(
                        {
                            "rebalance_date": decision_date,
                            "security_id": security_id,
                            "ticker": ticker,
                            "sector": sectors.get(security_id),
                            "score": float(row["score"][0]) if not row.is_empty() else None,
                            "rank": int(row["rank"][0]) if not row.is_empty() and row["rank"][0] is not None else None,
                            "target_weight": targets.get(security_id, 0.0),
                            "held_weight": value / equity_now if equity_now else None,
                            "shares": shares[security_id],
                            "price": p,
                            "value": value,
                        }
                    )

        # ---- mark to market --------------------------------------------------
        invested = 0.0
        for security_id, held in shares.items():
            ticker = security_tickers.get(security_id)
            invested += held * _price_or_last(
                market, i, ticker, last_price.get(security_id)
            )
        equity = cash + invested

        equity_path.append(equity)
        cash_path.append(cash)
        invested_path.append(invested)
        count_path.append(len(shares))
        cost_path.append(day_costs)
        turnover_path.append(day_turnover / previous_equity if previous_equity else 0.0)
        rebalance_path.append(decision_date is not None)
        return_path.append(
            equity / previous_equity - 1.0 if previous_equity else 0.0
        )
        previous_equity = equity

    # Unrealised profit on whatever is still held at the end belongs to those
    # names too, or every open winner would look like it contributed nothing.
    final_i = first_i + len(sessions) - 1
    for security_id, held in shares.items():
        ticker = security_tickers.get(security_id)
        value = held * _price_or_last(market, final_i, ticker, last_price.get(security_id))
        contribution[security_id] = contribution.get(security_id, 0.0) + (
            value - cost_basis.get(security_id, 0.0)
        )

    sector_contribution: dict[str, float] = {}
    for security_id, profit in contribution.items():
        sector = sectors.get(security_id) or "Unclassified"
        sector_contribution[sector] = sector_contribution.get(sector, 0.0) + profit

    return SimulationResult(
        dates=sessions,
        equity=np.array(equity_path),
        cash=np.array(cash_path),
        invested=np.array(invested_path),
        n_positions=np.array(count_path),
        daily_return=np.array(return_path),
        costs_paid=np.array(cost_path),
        turnover=np.array(turnover_path),
        is_rebalance=np.array(rebalance_path),
        trades=trades,
        holdings=holdings,
        delistings=delistings,
        contribution=contribution,
        sector_contribution=sector_contribution,
    )


def _price_or_last(
    market: MarketData, i: int, ticker: str | None, fallback: float | None
) -> float:
    """Today's traded price, or the last one observed if today has no bar.

    Carrying the last price is correct for a short gap and wrong for a dead
    company, which is why the delisting check runs first and removes the
    position before this can carry it indefinitely.
    """
    j = market.ticker_index.get(ticker) if ticker else None
    if j is not None:
        price = market.close_raw[i, j]
        if np.isfinite(price):
            return float(price)
    return float(fallback) if fallback is not None else 0.0


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def benchmark_series(
    market: MarketData, ticker: str, sessions: list[date]
) -> np.ndarray | None:
    """A benchmark's total-return index over the window, rebased to 1.0.

    Built from the adjusted close, which is the total-return series, so this is
    the benchmark *with* dividends reinvested -- the only fair comparison for a
    strategy whose own dividends are being counted.
    """
    j = market.ticker_index.get(ticker)
    if j is None:
        return None
    idx = [market.session_index[d] for d in sessions]
    values = market.close_adj[idx, j]
    if not np.isfinite(values).any():
        return None
    # Forward-fill a missing bar rather than dropping the session, so the
    # benchmark stays aligned with the strategy's own calendar.
    out = np.array(values, dtype=float)
    for k in range(1, out.size):
        if not np.isfinite(out[k]):
            out[k] = out[k - 1]
    first = next((v for v in out if np.isfinite(v)), None)
    if first is None or first == 0:
        return None
    out[~np.isfinite(out)] = first
    return out / first


def risk_matched_benchmark(
    benchmark_index: np.ndarray, strategy_returns: np.ndarray
) -> tuple[np.ndarray, float]:
    """The benchmark scaled to the strategy's own volatility, plus the scale used.

    A strategy that beats the market by holding more risk has not beaten the
    market; it has bought more of it. Mixing the benchmark with cash (or
    levering it) until its volatility matches the strategy's removes that
    explanation, and whatever excess survives is not simply exposure.
    """
    bench_returns = np.diff(benchmark_index) / benchmark_index[:-1]
    strat = strategy_returns[1:] if strategy_returns.size == benchmark_index.size else strategy_returns
    bench_vol = float(np.nanstd(bench_returns, ddof=1))
    strat_vol = float(np.nanstd(strat, ddof=1))
    if bench_vol <= 0:
        return benchmark_index, 1.0
    scale = strat_vol / bench_vol
    scaled = np.concatenate([[1.0], np.cumprod(1.0 + bench_returns * scale)])
    return scaled, scale
