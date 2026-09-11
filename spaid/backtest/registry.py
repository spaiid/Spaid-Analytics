"""Where backtest results go, including the ones nobody wanted.

The registry is append-only by policy, not by accident. Two things depend on it:

* **Reproducibility.** A stored run carries the strategy checksum, the data
  version, the universe version and the code commit, so a number on the screen
  can always be traced to the four things that produced it.
* **Honesty about search.** The deflated Sharpe ratio needs the number of
  trials that were run, and the probability of backtest overfitting needs their
  return series. Both get worse as the count rises, which is precisely why the
  count must not be curated. A registry that quietly loses its failures makes
  every surviving result look better than it is.

Nothing here interprets a result. `spaid.backtest.conclusion` does that, from
what this module stored.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime

import numpy as np
import polars as pl

from spaid.backtest import provenance
from spaid.backtest.engine import SimulationResult
from spaid.storage import store
from spaid.storage.schema import (
    BACKTEST_DAILY,
    BACKTEST_POSITIONS,
    BACKTEST_RUNS,
    BACKTEST_TRADES,
    BENCHMARK_VALUES,
    EXPERIMENTS,
    HOLDOUT_AUDIT,
    PERFORMANCE_METRICS,
    ROBUSTNESS_TESTS,
    SIGNAL_DIAGNOSTICS,
    coerce,
)

log = logging.getLogger(__name__)


def _json(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def save_run(
    *,
    run_id: str,
    purpose: str,
    trial_type: str,
    variant_label: str | None,
    strategy_version: str,
    config_checksum: str,
    period_label: str,
    period_start: date,
    period_end: date,
    params: dict,
    costs: dict,
    validity: str,
    status: str,
    n_rebalances: int,
    results: dict,
    elapsed_seconds: float,
    error: str | None = None,
    coordinates: dict | None = None,
) -> None:
    """Write the run header. Every coordinate needed to reproduce it is required."""
    coords = coordinates or provenance.coordinates()
    row = coerce(
        pl.DataFrame(
            [
                {
                    "run_id": run_id,
                    "created_at": datetime.now(UTC),
                    "purpose": purpose,
                    "trial_type": trial_type,
                    "variant_label": variant_label,
                    "strategy_version": strategy_version,
                    "config_checksum": config_checksum,
                    "code_commit": coords["code_commit"],
                    "data_version": coords["data_version"],
                    "universe_version": coords["universe_version"],
                    "period_label": period_label,
                    "period_start": period_start,
                    "period_end": period_end,
                    "params": _json(params),
                    "costs": _json(costs),
                    "validity": validity,
                    "status": status,
                    "n_rebalances": n_rebalances,
                    "results": _json(results),
                    "error": error,
                    "elapsed_seconds": elapsed_seconds,
                }
            ]
        ),
        BACKTEST_RUNS,
    )
    store.upsert(row, "backtest_runs", source="spaid")


def save_simulation(run_id: str, result: SimulationResult, *, sectors: dict | None = None) -> None:
    """Persist the daily path, the positions and every trade of one simulation."""
    sectors = sectors or {}

    daily = coerce(
        pl.DataFrame(
            {
                "run_id": [run_id] * len(result.dates),
                "date": result.dates,
                "gross_equity": result.equity.tolist(),
                "net_equity": result.equity.tolist(),
                "cash": result.cash.tolist(),
                "invested": result.invested.tolist(),
                "n_positions": result.n_positions.tolist(),
                "gross_return": result.daily_return.tolist(),
                "net_return": result.daily_return.tolist(),
                "costs_paid": result.costs_paid.tolist(),
                "drawdown": _drawdown(result.equity).tolist(),
                "is_rebalance": result.is_rebalance.tolist(),
                "turnover": result.turnover.tolist(),
            }
        ),
        BACKTEST_DAILY,
    )
    store.upsert(daily, "backtest_daily", source="spaid")

    if result.holdings:
        positions = pl.DataFrame(result.holdings).with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.lit(None, dtype=pl.Utf8).alias("company_id"),
        )
        positions = coerce(positions, BACKTEST_POSITIONS).unique(
            subset=list(BACKTEST_POSITIONS.primary_key), keep="last"
        )
        store.upsert(positions, "backtest_positions", source="spaid")

    if result.trades:
        trades = pl.DataFrame(
            [
                {
                    "run_id": run_id,
                    "trade_id": f"{run_id}-{i:06d}",
                    "decision_date": t.decision_date,
                    "trade_date": t.trade_date,
                    "security_id": t.security_id,
                    "ticker": t.ticker,
                    "side": t.side,
                    "shares": t.shares,
                    "price": t.price,
                    "gross_value": t.gross_value,
                    # The cost model is a single per-side rate, so splitting it
                    # into named components here would invent a decomposition
                    # the model does not have. The total is what was charged.
                    "spread_cost": None,
                    "slippage_cost": None,
                    "commission": None,
                    "total_cost": t.cost,
                    "reason": t.reason,
                }
                for i, t in enumerate(result.trades)
            ]
        )
        store.upsert(coerce(trades, BACKTEST_TRADES), "backtest_trades", source="spaid")


def save_gross_and_net(
    run_id: str, net: SimulationResult, gross: SimulationResult
) -> None:
    """Overwrite the daily table with both paths, aligned on the same dates."""
    n = min(len(net.dates), len(gross.dates))
    daily = coerce(
        pl.DataFrame(
            {
                "run_id": [run_id] * n,
                "date": net.dates[:n],
                "gross_equity": gross.equity[:n].tolist(),
                "net_equity": net.equity[:n].tolist(),
                "cash": net.cash[:n].tolist(),
                "invested": net.invested[:n].tolist(),
                "n_positions": net.n_positions[:n].tolist(),
                "gross_return": gross.daily_return[:n].tolist(),
                "net_return": net.daily_return[:n].tolist(),
                "costs_paid": net.costs_paid[:n].tolist(),
                "drawdown": _drawdown(net.equity[:n]).tolist(),
                "is_rebalance": net.is_rebalance[:n].tolist(),
                "turnover": net.turnover[:n].tolist(),
            }
        ),
        BACKTEST_DAILY,
    )
    store.upsert(daily, "backtest_daily", source="spaid")


def _drawdown(equity: np.ndarray) -> np.ndarray:
    if equity.size == 0:
        return equity
    return equity / np.maximum.accumulate(equity) - 1.0


def save_benchmarks(series: dict[str, tuple[list[date], np.ndarray]]) -> None:
    """Store each benchmark's total-return index as it was used.

    Stored rather than recomputed because a refreshed price table would
    otherwise silently change what a historical comparison was made against.
    """
    frames = []
    for name, (dates, index) in series.items():
        if index is None or len(dates) != index.size:
            continue
        returns = np.concatenate([[np.nan], index[1:] / index[:-1] - 1.0])
        frames.append(
            pl.DataFrame(
                {
                    "benchmark": [name] * len(dates),
                    "date": dates,
                    "close_adj": index.tolist(),
                    "total_return_index": index.tolist(),
                    "daily_return": returns.tolist(),
                    "source": ["yahoo"] * len(dates),
                    "collected_at": [datetime.now(UTC)] * len(dates),
                }
            )
        )
    if not frames:
        return
    out = coerce(pl.concat(frames, how="vertical_relaxed"), BENCHMARK_VALUES)
    store.upsert(out, "benchmark_values", source="spaid")


def save_performance(run_id: str, scope: str, period_label: str, summary: dict) -> None:
    """Flatten a metric summary into rows, including its confidence intervals."""
    rows: list[dict] = []

    def add(metric: str, value, benchmark: str | None = None, detail=None,
            ci: dict | None = None, n_obs: int | None = None) -> None:
        if value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(numeric):
            return
        rows.append(
            {
                "run_id": run_id,
                "scope": scope,
                "benchmark": benchmark or "",
                "period_label": period_label,
                "metric": metric,
                "value": numeric,
                "ci_low": ci.get("low") if ci else None,
                "ci_high": ci.get("high") if ci else None,
                "n_obs": n_obs,
                "detail": _json(detail) if detail is not None else None,
            }
        )

    simple = (
        "total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown",
        "calmar", "positive_months", "turnover_annual", "turnover_annual_one_way",
        "turnover_per_rebalance", "total_costs", "final_equity", "years",
    )
    for metric in simple:
        add(metric, summary.get(metric),
            ci=summary.get(f"{metric}_ci"), n_obs=summary.get("n_sessions"))

    if "drawdown_duration" in summary:
        add("drawdown_duration_days", summary["drawdown_duration"].get("days"),
            detail=summary["drawdown_duration"])
    for key in ("best_month", "worst_month", "best_year", "worst_year"):
        if key in summary:
            add(key, summary[key].get("return"), detail=summary[key])
    if "deflated_sharpe" in summary:
        add("deflated_sharpe_probability", summary["deflated_sharpe"].get("probability"),
            detail=summary["deflated_sharpe"])

    for name, entry in summary.get("benchmarks", {}).items():
        for metric in (
            "total_return", "cagr", "excess_cagr", "alpha_annual", "beta",
            "tracking_error", "information_ratio", "upside_capture",
            "downside_capture", "volatility", "sharpe", "max_drawdown",
        ):
            add(metric, entry.get(metric), benchmark=name)
        for window in (12, 36, 60):
            rolling = entry.get(f"rolling_{window}m", {})
            add(f"rolling_{window}m_beat_share", rolling.get("share"),
                benchmark=name, detail=rolling)

    if not rows:
        return
    frame = coerce(pl.DataFrame(rows), PERFORMANCE_METRICS).unique(
        subset=list(PERFORMANCE_METRICS.primary_key), keep="last"
    )
    store.upsert(frame, "performance_metrics", source="spaid")


def save_signal_diagnostics(run_id: str, rows: list[dict]) -> None:
    """Persist information coefficients, buckets, cuts and ablations."""
    if not rows:
        return
    frame = pl.DataFrame(
        [{"run_id": run_id, **row} for row in rows], infer_schema_length=None
    )
    frame = coerce(frame, SIGNAL_DIAGNOSTICS)
    # A diagnostic with no date is a summary; the primary key includes `date`,
    # so those rows need a sentinel that cannot collide with a real one.
    frame = frame.with_columns(
        pl.col("date").fill_null(date(1900, 1, 1)),
        pl.col("horizon_months").fill_null(0),
        pl.col("bucket").fill_null(""),
    ).unique(subset=list(SIGNAL_DIAGNOSTICS.primary_key), keep="last")
    store.upsert(frame, "signal_diagnostics", source="spaid")


def save_robustness(rows: list[dict]) -> None:
    if not rows:
        return
    frame = coerce(pl.DataFrame(rows, infer_schema_length=None), ROBUSTNESS_TESTS)
    frame = frame.unique(subset=list(ROBUSTNESS_TESTS.primary_key), keep="last")
    store.upsert(frame, "robustness_tests", source="spaid")


def record_trial(
    *,
    trial_id: str,
    purpose: str,
    hypothesis: str,
    strategy_version: str,
    config_checksum: str,
    period_label: str,
    period_start: date,
    period_end: date,
    params: dict,
    costs: dict,
    run_id: str | None,
    outcome: str,
    results: dict,
    notes: str = "",
    coordinates: dict | None = None,
) -> None:
    """Append one trial to the permanent registry.

    Appending, never updating: `store.append` refuses a duplicate key, so a
    trial cannot be quietly rerun into a better result under the same identity.
    """
    coords = coordinates or provenance.coordinates()
    row = coerce(
        pl.DataFrame(
            [
                {
                    "trial_id": trial_id,
                    "created_at": datetime.now(UTC),
                    "purpose": purpose,
                    "hypothesis": hypothesis,
                    "strategy_version": strategy_version,
                    "config_checksum": config_checksum,
                    "code_commit": coords["code_commit"],
                    "data_version": coords["data_version"],
                    "universe_version": coords["universe_version"],
                    "period_label": period_label,
                    "period_start": period_start,
                    "period_end": period_end,
                    "params": _json(params),
                    "costs": _json(costs),
                    "run_id": run_id,
                    "outcome": outcome,
                    "results": _json(results),
                    "notes": notes,
                }
            ]
        ),
        EXPERIMENTS,
    )
    try:
        store.append(row, "experiments", source="spaid")
    except ValueError:
        log.warning("trial %s is already registered; not overwriting it", trial_id)


def trial_count(strategy_version: str | None = None) -> int:
    """How many trials have been run, for the deflated Sharpe ratio.

    Counts every registered trial, not the successful ones. That is the whole
    point of the correction.
    """
    trials = store.read("experiments")
    if trials is None or trials.is_empty():
        return 1
    if strategy_version:
        trials = trials.filter(pl.col("strategy_version") == strategy_version)
    return max(1, trials.height)


def record_holdout_access(
    *,
    audit_id: str,
    reason: str,
    strategy_version: str,
    config_checksum: str,
    holdout_start: date,
    holdout_end: date,
    run_id: str | None,
    results: dict,
) -> None:
    """Permanently record that the holdout was looked at, and why."""
    row = coerce(
        pl.DataFrame(
            [
                {
                    "audit_id": audit_id,
                    "requested_at": datetime.now(UTC),
                    "reason": reason,
                    "strategy_version": strategy_version,
                    "config_checksum": config_checksum,
                    "holdout_start": holdout_start,
                    "holdout_end": holdout_end,
                    "run_id": run_id,
                    "results": _json(results),
                }
            ]
        ),
        HOLDOUT_AUDIT,
    )
    store.append(row, "holdout_audit", source="spaid")


def holdout_accesses(strategy_version: str | None = None) -> pl.DataFrame:
    audit = store.read("holdout_audit")
    if audit is None or audit.is_empty():
        return pl.DataFrame(schema=dict(HOLDOUT_AUDIT.schema))
    if strategy_version:
        audit = audit.filter(pl.col("strategy_version") == strategy_version)
    return audit.sort("requested_at")


def trial_return_matrix(
    strategy_version: str, *, period_label: str | None = None
) -> tuple[np.ndarray, list[str]]:
    """Monthly net returns for every completed run, one row per run.

    Feeds the probability of backtest overfitting. Runs are aligned on their
    common dates, and any run that does not cover the full span is dropped
    rather than padded -- padding with zeros would make a short run look like a
    flat one and bias the ranking.
    """
    runs = store.read("backtest_runs")
    daily = store.read("backtest_daily")
    if runs is None or daily is None or runs.is_empty() or daily.is_empty():
        return np.zeros((0, 0)), []

    wanted = runs.filter(
        (pl.col("strategy_version") == strategy_version) & (pl.col("status") == "completed")
    )
    if period_label:
        wanted = wanted.filter(pl.col("period_label") == period_label)
    ids = wanted["run_id"].to_list()
    if len(ids) < 2:
        return np.zeros((0, 0)), []

    monthly = (
        daily.filter(pl.col("run_id").is_in(ids))
        .sort(["run_id", "date"])
        .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by(["run_id", "month"])
        .agg(pl.col("net_equity").last())
        .sort(["run_id", "month"])
        .with_columns(
            (pl.col("net_equity") / pl.col("net_equity").shift(1).over("run_id") - 1.0).alias("r")
        )
        .drop_nulls("r")
    )
    if monthly.is_empty():
        return np.zeros((0, 0)), []

    wide = monthly.pivot(on="run_id", index="month", values="r").sort("month")
    columns = [c for c in wide.columns if c != "month"]
    matrix = wide.select(columns).to_numpy().T
    keep = ~np.isnan(matrix).any(axis=1)
    return matrix[keep], [c for c, k in zip(columns, keep, strict=True) if k]
