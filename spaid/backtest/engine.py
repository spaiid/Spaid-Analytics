"""Portfolio simulation over walk-forward scores.

Design notes that keep the numbers honest:

* Positions chosen from the close of day `t` earn day `t+1`'s return. Scoring
  and trading on the same bar is the most common way a backtest invents money.
* Between rebalances weights **drift** with returns instead of being silently
  reset to equal weight each day. Resetting implies a free daily rebalance and
  understates turnover.
* Turnover is charged on the sum of absolute weight changes at each rebalance,
  in basis points per side.
* Everything is arithmetic. The old codebase built a log-return target and then
  compounded it as though it were arithmetic; the two must not be mixed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import polars as pl

from spaid.config import SETTINGS

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    dates: list
    equity: list[float]
    benchmark: list[float]
    daily_returns: list[float]
    drawdown: list[float]
    cagr: float
    sharpe: float
    max_drawdown: float
    volatility: float
    turnover: float
    hit_rate: float
    excess_cagr: float
    benchmark_cagr: float
    cost_bps: float
    top_n: int
    n_rebalances: int

    def to_dict(self) -> dict:
        return asdict(self)


def _target_weights(scores: np.ndarray, top_n: int, max_weight: float, long_short: bool) -> np.ndarray:
    w = np.zeros_like(scores, dtype=float)
    valid = np.isfinite(scores)
    if valid.sum() == 0:
        return w

    idx = np.where(valid)[0]
    order = idx[np.argsort(-scores[idx])]
    n = min(top_n, order.size)
    if n == 0:
        return w

    longs = order[:n]
    w[longs] = 1.0 / n
    if long_short:
        shorts = order[-n:]
        w[longs] = 0.5 / n
        w[shorts] = -0.5 / n

    w = np.clip(w, -max_weight, max_weight)
    gross = np.abs(w).sum()
    if gross > 0:
        w = w / gross
    return w


def run_backtest(
    scored: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    top_n: int | None = None,
    cost_bps: float | None = None,
    long_short: bool | None = None,
    rebalance_days: int | None = None,
    max_weight: float | None = None,
    score_col: str = "score",
) -> BacktestResult:
    cfg = SETTINGS.backtest
    top_n = top_n if top_n is not None else cfg.top_n
    cost_bps = cost_bps if cost_bps is not None else cfg.cost_bps
    long_short = long_short if long_short is not None else cfg.long_short
    rebalance_days = rebalance_days if rebalance_days is not None else cfg.rebalance_days
    max_weight = max_weight if max_weight is not None else cfg.max_weight

    from spaid.data.universe import benchmark_ticker

    bench = benchmark_ticker()

    if scored.is_empty():
        raise ValueError("no scored rows to back-test")

    dates = scored.select("date").unique().sort("date")["date"].to_list()
    tickers = scored.select("ticker").unique().sort("ticker")["ticker"].to_list()
    t_index = {t: i for i, t in enumerate(tickers)}
    d_index = {d: i for i, d in enumerate(dates)}
    n_d, n_t = len(dates), len(tickers)

    # daily simple returns, aligned to the scored panel
    rets = np.full((n_d, n_t), np.nan)
    px = (
        prices.filter(pl.col("ticker").is_in(tickers))
        .sort(["ticker", "date"])
        .with_columns((pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("r"))
        .filter(pl.col("date").is_in(dates))
        .select(["date", "ticker", "r"])
    )
    for d, t, r in px.iter_rows():
        if r is not None and np.isfinite(r):
            rets[d_index[d], t_index[t]] = r

    scores = np.full((n_d, n_t), np.nan)
    for d, t, s in scored.select(["date", "ticker", score_col]).iter_rows():
        if s is not None and np.isfinite(s):
            scores[d_index[d], t_index[t]] = s

    # benchmark, buy and hold
    bpx = (
        prices.filter(pl.col("ticker") == bench)
        .sort("date")
        .with_columns((pl.col("close") / pl.col("close").shift(1) - 1.0).alias("r"))
        .filter(pl.col("date").is_in(dates))
        .select(["date", "r"])
    )
    bench_r = np.zeros(n_d)
    for d, r in bpx.iter_rows():
        if r is not None and np.isfinite(r):
            bench_r[d_index[d]] = r

    rebalance_idx = set(range(0, n_d, max(1, rebalance_days)))

    w = np.zeros(n_t)
    equity = 1.0
    bench_equity = 1.0
    eq_curve, bench_curve, daily, turnovers = [], [], [], []

    for i in range(n_d):
        day_ret = 0.0
        if i > 0:
            r = rets[i]
            live = np.isfinite(r) & (w != 0.0)
            if live.any():
                day_ret = float(np.sum(w[live] * r[live]))
                # weights drift with the move rather than resetting for free
                w[live] = w[live] * (1.0 + r[live])
                gross = np.abs(w).sum()
                if gross > 0:
                    w = w / gross
            bench_equity *= 1.0 + bench_r[i]

        if i in rebalance_idx:
            target = _target_weights(scores[i], top_n, max_weight, long_short)
            if np.abs(target).sum() > 0 or np.abs(w).sum() > 0:
                turnover = float(np.abs(target - w).sum())
                turnovers.append(turnover)
                cost = turnover * cost_bps / 1e4
                day_ret -= cost
                w = target

        equity *= 1.0 + day_ret
        eq_curve.append(equity)
        bench_curve.append(bench_equity)
        daily.append(day_ret)

    r = np.array(daily)
    eq = np.array(eq_curve)
    years = max(n_d / 252.0, 1e-9)

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0

    vol = float(r.std(ddof=1) * np.sqrt(252)) if r.size > 1 else float("nan")
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.size > 1 and r.std(ddof=1) > 0 else float("nan")
    cagr = float(eq[-1] ** (1 / years) - 1.0)
    b_cagr = float(bench_curve[-1] ** (1 / years) - 1.0)

    result = BacktestResult(
        dates=[str(d) for d in dates],
        equity=[float(x) for x in eq_curve],
        benchmark=[float(x) for x in bench_curve],
        daily_returns=[float(x) for x in daily],
        drawdown=[float(x) for x in dd],
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=float(dd.min()),
        volatility=vol,
        turnover=float(np.mean(turnovers) * (252 / max(1, rebalance_days))) if turnovers else float("nan"),
        hit_rate=float((r > 0).mean()),
        excess_cagr=cagr - b_cagr,
        benchmark_cagr=b_cagr,
        cost_bps=cost_bps,
        top_n=top_n,
        n_rebalances=len(turnovers),
    )
    log.info(
        "backtest: top%d, %d rebalances, CAGR %.2f%% vs benchmark %.2f%%, Sharpe %.2f, maxDD %.1f%%",
        top_n, len(turnovers), cagr * 100, b_cagr * 100, sharpe, dd.min() * 100,
    )
    return result
