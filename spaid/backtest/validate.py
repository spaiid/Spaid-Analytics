"""The validation run: one command, every predefined test, one honest verdict.

Order matters here, and it is fixed:

1. Freeze the strategy, so that what is being tested cannot move mid-test.
2. Build the point-in-time panel over the requested period.
3. Run *all three* portfolio variants -- not the best one, all of them -- and
   record each as its own trial.
4. Measure the ranking itself: information coefficients, buckets, cuts,
   categories, ablations, persistence.
5. Run the whole predefined robustness battery, without stopping when the
   results start to look bad.
6. Evaluate the gates and publish the verdict.

Nothing in this file chooses what to report after seeing the numbers. The
variants, the primary variant, the horizons, the robustness tests and every
threshold come from `spaid.config.validation`, which was written first.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import UTC, date, datetime

import numpy as np
import polars as pl

from spaid.backtest import conclusion as conclusion_mod
from spaid.backtest import metrics as metrics_mod
from spaid.backtest import panel as panel_mod
from spaid.backtest import provenance, registry
from spaid.backtest import signal as signal_mod
from spaid.backtest.engine import (
    MarketData,
    benchmark_series,
    risk_matched_benchmark,
    simulate,
)
from spaid.backtest.strategy import VARIANTS, StrategyVersion, freeze, get_strategy
from spaid.config.validation import (
    ACTIVE_VALIDATION_SPEC,
    ROBUSTNESS_BATTERY,
    Period,
    ValidationSpec,
)
from spaid.pipeline import security_master
from spaid.storage import store

log = logging.getLogger(__name__)


def _period_for(label: str, spec: ValidationSpec) -> Period:
    periods = spec.periods
    match label:
        case "development":
            return periods.development
        case "validation":
            return periods.validation
        case "holdout":
            return periods.holdout
        case "iterable":
            return periods.iterable
        case "full":
            return periods.full
    raise KeyError(f"unknown period {label!r}")


def _benchmark_indices(
    market: MarketData, sessions: list[date], tickers: tuple[str, ...]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for ticker in tickers:
        series = benchmark_series(market, ticker, sessions)
        if series is not None:
            out[ticker] = series
    return out


def run_validation(
    *,
    period_label: str = "development",
    strategy_version: str | None = None,
    spec: ValidationSpec | None = None,
    purpose: str = "Frozen Strategy Version 1 baseline",
    with_robustness: bool = True,
    persist: bool = True,
    allow_holdout: bool = False,
) -> dict:
    """Run the whole protocol over one period and return the verdict with its evidence."""
    spec = spec or ACTIVE_VALIDATION_SPEC
    strategy = get_strategy(strategy_version)
    started = time.time()

    if period_label == "holdout" and not allow_holdout:
        raise PermissionError(
            "The holdout period is sealed. Evaluating it requires "
            "spaid.backtest.holdout.evaluate(), which records an audit entry, so that the "
            "number of times it has been examined is always knowable."
        )

    frozen = freeze(strategy, persist=persist)
    period = _period_for(period_label, spec)
    coordinates = provenance.coordinates()
    log.info(
        "validation: %s over %s to %s, strategy %s (%s), data %s, universe %s",
        period_label, period.start, period.end, strategy.version,
        frozen["config_checksum"][:12], coordinates["data_version"],
        coordinates["universe_version"],
    )

    # ---- panel ------------------------------------------------------------
    # The rebalance-shift tests trade on days the month-end schedule never
    # visits, so those days are scored too. Doing this in one pass costs a
    # little time and buys the alternative: reusing a nearby month-end score,
    # which would answer the question with the very look-ahead it is asking
    # about.
    prices_for_grid = store.read("prices", required=True)
    sessions_all = panel_mod.trading_sessions(prices_for_grid)
    month_ends = panel_mod.month_end_sessions(sessions_all, period.start, period.end)
    shift_offsets = sorted(
        {
            overrides["rebalance_offset"]
            for test in ROBUSTNESS_BATTERY
            for _, overrides in test.variants
            if "rebalance_offset" in overrides
        }
    )
    extra_dates: list[date] = []
    if with_robustness:
        for offset in shift_offsets:
            extra_dates.extend(_shift_dates(month_ends, sessions_all, offset))

    built = panel_mod.build(
        strategy,
        start=period.start,
        end=period.end,
        horizons_months=spec.ic_horizons,
        extra_dates=sorted(set(extra_dates)),
    )
    panel = built["panel"]
    decision_dates = built["decision_dates"]
    prices = built["prices"]

    scored = panel.filter(pl.col("is_eligible") & pl.col("score").is_not_null())
    # Every diagnostic of the ranking uses the month-end grid only. Including
    # the shifted days would count almost the same cross-section several times
    # and shrink every standard error it touched.
    scored_primary = scored.filter(pl.col("date").is_in(decision_dates))
    sectors = dict(
        panel.select(["security_id", "sector"]).unique(subset=["security_id"]).iter_rows()
    )
    security_tickers = dict(
        panel.select(["security_id", "ticker"]).unique(subset=["security_id"]).iter_rows()
    )

    sessions = [d for d in panel_mod.trading_sessions(prices) if period.start <= d <= period.end]
    market = MarketData.build(prices, sessions)
    benchmarks = _benchmark_indices(market, sessions, strategy.benchmarks)
    benchmark_coverage = {
        t: (t in benchmarks and np.isfinite(benchmarks[t]).all()) for t in strategy.benchmarks
    }

    # ---- the declared portfolio variants ---------------------------------
    variant_results: dict[str, dict] = {}
    n_trials_before = registry.trial_count(strategy.version)

    for label in spec.portfolio_variants:
        rules = VARIANTS[label]
        outcome = _run_variant(
            label=label,
            rules=rules,
            strategy=strategy,
            spec=spec,
            panel=scored_primary,
            market=market,
            period=period,
            period_label=period_label,
            decision_dates=decision_dates,
            benchmarks=benchmarks,
            security_tickers=security_tickers,
            sectors=sectors,
            purpose=purpose,
            trial_type="primary" if label == spec.primary_variant else "variant",
            coordinates=coordinates,
            n_trials=n_trials_before + len(spec.portfolio_variants),
            persist=persist,
        )
        variant_results[label] = outcome

    primary = variant_results[spec.primary_variant]

    # ---- does the ranking itself work ------------------------------------
    diagnostics = _signal_diagnostics(
        scored_primary, strategy, spec, prices, decision_dates, primary
    )
    if persist:
        registry.save_signal_diagnostics(primary["run_id"], diagnostics["rows"])

    # ---- robustness -------------------------------------------------------
    robustness = {}
    if with_robustness:
        robustness = _robustness(
            strategy=strategy,
            spec=spec,
            panel=scored,
            market=market,
            period=period,
            period_label=period_label,
            decision_dates=decision_dates,
            benchmarks=benchmarks,
            security_tickers=security_tickers,
            sectors=sectors,
            base=primary,
            coordinates=coordinates,
            persist=persist,
            sessions=sessions_all,
        )

    # ---- how much search happened ----------------------------------------
    # The deflation has to know about every trial, including the ones that ran
    # after it was first computed.
    #
    # Each variant's deflated Sharpe was calculated inside `_run_variant`, which
    # happens before the robustness battery exists, so it was deflated against
    # roughly half the search that actually took place: the development period
    # reported an expected-maximum Sharpe of 0.4709 against a true 0.8603, and a
    # probability of 0.6784 where the honest figure is 0.3488 -- the published
    # number was about twice what the evidence supported. Recomputing here, once
    # the register is complete, is the only point at which the count is final.
    final_trials = registry.trial_count(strategy.version)
    for outcome in variant_results.values():
        for key, path in (("net_summary", outcome.get("net")),
                          ("gross_summary", outcome.get("gross"))):
            block = outcome.get(key)
            if not block or "deflated_sharpe" not in block or path is None:
                continue
            restated = metrics_mod.deflated_sharpe(
                metrics_mod.to_returns(path.equity), n_trials=final_trials
            )
            block["deflated_sharpe"] = {
                "sharpe_annual": restated.sharpe_annual,
                "expected_max_sharpe_annual": restated.expected_max_sharpe_annual,
                "probability": restated.deflated,
                "n_trials": restated.n_trials,
                "skew": restated.skew,
                "kurtosis": restated.kurtosis,
                "verdict": restated.verdict,
            }

    matrix, _trial_run_ids = registry.trial_return_matrix(strategy.version, period_label=period_label)
    pbo = (
        metrics_mod.probability_of_backtest_overfitting(matrix)
        if matrix.size
        else {"feasible": False, "reason": "fewer than two comparable runs are stored"}
    )

    # ---- the verdict ------------------------------------------------------
    coverage = security_master.coverage_report()
    measurements = _measurements(primary, diagnostics)
    verdict = conclusion_mod.decide(
        coverage=coverage,
        measurements=measurements,
        period_label=period_label,
        spec=spec,
        benchmark_coverage=benchmark_coverage,
        execution_delay_days=strategy.execution.delay_days,
        robustness_summary=robustness.get("summary"),
        n_decision_dates=len(decision_dates),
    )

    summary = {
        "strategy_version": strategy.version,
        "config_checksum": frozen["config_checksum"],
        "scoring_version": strategy.scoring.version,
        **coordinates,
        "period_label": period_label,
        "period_start": str(period.start),
        "period_end": str(period.end),
        "decision_dates": len(decision_dates),
        "universe_per_date": _universe_sizes(panel),
        "variants": {
            label: {
                "run_id": v["run_id"],
                "net": v["net_summary"],
                "gross": v["gross_summary"],
            }
            for label, v in variant_results.items()
        },
        "primary_variant": spec.primary_variant,
        "signal": diagnostics["summary"],
        "robustness": robustness,
        "probability_of_overfitting": pbo,
        "coverage": coverage,
        "verdict": verdict.to_dict(),
        "point_in_time_coverage": strategy.category_coverage_point_in_time(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if persist:
        store.write_json(summary, f"validation_{period_label}")
        # The risk-matched series is derived from the primary portfolio's own
        # volatility, so it only exists once a variant has run. It is stored
        # alongside the raw benchmarks so the chart can draw the comparison the
        # metrics table reports rather than only naming it.
        stored_benchmarks = {k: (sessions, v) for k, v in benchmarks.items()}
        matched = primary["benchmarks"].get("SPY (risk-matched)")
        if matched is not None:
            stored_benchmarks["SPY (risk-matched)"] = (sessions, matched)
        registry.save_benchmarks(stored_benchmarks)

    log.info(
        "validation complete: status %s, conclusion %s (%.1fs)",
        verdict.status.value, verdict.conclusion.value, summary["elapsed_seconds"],
    )
    return summary


# ---------------------------------------------------------------------------
# One variant
# ---------------------------------------------------------------------------


def _run_variant(
    *,
    label: str,
    rules,
    strategy: StrategyVersion,
    spec: ValidationSpec,
    panel: pl.DataFrame,
    market: MarketData,
    period: Period,
    period_label: str,
    decision_dates: list[date],
    benchmarks: dict[str, np.ndarray],
    security_tickers: dict[str, str],
    sectors: dict[str, str],
    purpose: str,
    trial_type: str,
    coordinates: dict,
    n_trials: int,
    persist: bool,
    costs=None,
    execution=None,
    excluded_securities: frozenset[str] = frozenset(),
    run_suffix: str = "",
) -> dict:
    """Simulate one portfolio twice -- with costs and without -- and store both."""
    costs = costs or strategy.costs
    execution = execution or strategy.execution
    started = time.time()
    run_id = provenance.run_id(f"bt-{period_label}-{label}{run_suffix}")

    net = simulate(
        panel, market, rules,
        execution=execution, costs=costs,
        decision_dates=decision_dates, start=period.start, end=period.end,
        security_tickers=security_tickers, sectors=sectors,
        excluded_securities=excluded_securities,
    )
    gross = simulate(
        panel, market, rules,
        execution=execution, costs=costs.scaled(0.0),
        decision_dates=decision_dates, start=period.start, end=period.end,
        security_tickers=security_tickers, sectors=sectors,
        excluded_securities=excluded_securities,
    )

    aligned = {k: v for k, v in benchmarks.items() if v.size == net.equity.size}
    risk_matched = None
    if "SPY" in aligned:
        risk_matched, scale = risk_matched_benchmark(
            aligned["SPY"], metrics_mod.to_returns(net.equity)
        )
        aligned = {**aligned, "SPY (risk-matched)": risk_matched}

    net_summary = metrics_mod.summarise(
        net.dates, net.equity, benchmarks=aligned,
        costs_paid=net.costs_paid, turnover=net.turnover, n_trials=n_trials,
    )
    gross_summary = metrics_mod.summarise(
        gross.dates, gross.equity, benchmarks=aligned,
        costs_paid=None, turnover=gross.turnover, n_trials=n_trials,
        with_intervals=False,
    )
    net_summary["delistings"] = len(net.delistings)
    net_summary["trades"] = len(net.trades)
    net_summary["holding_period"] = signal_mod.holding_periods(net.holdings)
    net_summary["contribution_by_stock"] = _top_contributors(net, security_tickers)
    net_summary["contribution_by_sector"] = sorted(
        (
            {"sector": k, "profit": v}
            for k, v in net.sector_contribution.items()
        ),
        key=lambda r: r["profit"],
        reverse=True,
    )
    if risk_matched is not None:
        net_summary["risk_match_scale"] = scale

    if persist:
        registry.save_run(
            run_id=run_id,
            purpose=purpose,
            trial_type=trial_type,
            variant_label=label,
            strategy_version=strategy.version,
            config_checksum=strategy.checksum,
            period_label=period_label,
            period_start=period.start,
            period_end=period.end,
            params={
                "variant": label,
                "rules": vars(rules),
                "delay_days": execution.delay_days,
                "excluded_securities": sorted(excluded_securities),
            },
            costs=costs.to_dict(),
            validity="pending",
            status="completed",
            n_rebalances=int(net.is_rebalance.sum()),
            results={
                "net_cagr": net_summary["cagr"],
                "gross_cagr": gross_summary["cagr"],
                "net_sharpe": net_summary["sharpe"],
                "max_drawdown": net_summary["max_drawdown"],
                "turnover_annual": net_summary.get("turnover_annual"),
                "benchmarks": {
                    k: {"excess_cagr": v["excess_cagr"], "information_ratio": v["information_ratio"]}
                    for k, v in net_summary["benchmarks"].items()
                },
            },
            elapsed_seconds=round(time.time() - started, 1),
            coordinates=coordinates,
        )
        registry.save_simulation(run_id, net, sectors=sectors)
        registry.save_gross_and_net(run_id, net, gross)
        registry.save_performance(run_id, "net", period_label, net_summary)
        registry.save_performance(run_id, "gross", period_label, gross_summary)
        registry.record_trial(
            trial_id=run_id,
            purpose=purpose,
            hypothesis=(
                f"The {label} portfolio built from frozen {strategy.version} beats SPY and "
                "SPMO after costs over the " + period_label + " period."
            ),
            strategy_version=strategy.version,
            config_checksum=strategy.checksum,
            period_label=period_label,
            period_start=period.start,
            period_end=period.end,
            params={"variant": label, "delay_days": execution.delay_days},
            costs=costs.to_dict(),
            run_id=run_id,
            outcome="completed",
            results={
                "net_cagr": net_summary["cagr"],
                "excess_vs_spy": net_summary["benchmarks"].get("SPY", {}).get("excess_cagr"),
            },
            notes=trial_type,
            coordinates=coordinates,
        )

    return {
        "run_id": run_id,
        "label": label,
        "net": net,
        "gross": gross,
        "net_summary": net_summary,
        "gross_summary": gross_summary,
        "benchmarks": aligned,
    }


def _top_contributors(net, security_tickers: dict[str, str], limit: int = 25) -> list[dict]:
    rows = [
        {"security_id": s, "ticker": security_tickers.get(s, s), "profit": p}
        for s, p in net.contribution.items()
    ]
    rows.sort(key=lambda r: r["profit"], reverse=True)
    return rows[:limit] + rows[-limit:] if len(rows) > 2 * limit else rows


def _universe_sizes(panel: pl.DataFrame) -> dict:
    per_date = panel.group_by("date").agg(
        pl.col("is_eligible").sum().alias("eligible"),
        pl.col("score").is_not_null().sum().alias("scored"),
        pl.len().alias("total"),
    )
    return {
        "median_eligible": float(per_date["eligible"].median()),
        "min_eligible": int(per_date["eligible"].min()),
        "max_eligible": int(per_date["eligible"].max()),
        "median_scored": float(per_date["scored"].median()),
    }


# ---------------------------------------------------------------------------
# Signal diagnostics
# ---------------------------------------------------------------------------


def _signal_diagnostics(
    scored: pl.DataFrame,
    strategy: StrategyVersion,
    spec: ValidationSpec,
    prices: pl.DataFrame,
    decision_dates: list[date],
    primary: dict,
) -> dict:
    """Every measurement of the ranking itself, flattened for storage."""
    rows: list[dict] = []
    summary: dict = {}

    ic = signal_mod.ic_across_horizons(scored, spec.ic_horizons)
    summary["information_coefficient"] = {
        str(h): {k: v for k, v in s.items() if k != "series"} for h, s in ic.items()
    }
    for horizon, entry in ic.items():
        for metric in ("ic_mean", "ic_median", "ic_std", "ic_t", "ic_ir", "hit_rate"):
            rows.append(
                {"kind": "ic", "horizon_months": horizon, "bucket": "", "metric": metric,
                 "value": entry[metric], "n_obs": entry["n_dates"], "date": None,
                 "detail": None}
            )
        for point in entry["series"]:
            rows.append(
                {"kind": "ic_series", "horizon_months": horizon, "bucket": "", "metric": "ic",
                 "value": point["ic"], "n_obs": point["n"],
                 "date": date.fromisoformat(point["date"]), "detail": None}
            )

    # buckets
    summary["buckets"] = {}
    for n_buckets, kind in ((10, "decile"), (5, "quintile")):
        for horizon in spec.ic_horizons:
            buckets = signal_mod.bucket_returns(
                scored, n_buckets=n_buckets, horizon_months=horizon
            )
            if buckets.is_empty():
                continue
            key = f"{kind}_{horizon}m"
            summary["buckets"][key] = {
                "rows": buckets.to_dicts(),
                "monotonicity": signal_mod.monotonicity(buckets),
                "spread": signal_mod.spread(buckets),
            }
            for row in buckets.iter_rows(named=True):
                rows.append(
                    {"kind": kind, "horizon_months": horizon, "bucket": str(row["bucket"]),
                     "metric": "mean_excess", "value": row["mean_excess"],
                     "n_obs": row["n"], "date": None, "detail": None}
                )
            rows.append(
                {"kind": kind, "horizon_months": horizon, "bucket": "spread",
                 "metric": "top_minus_bottom", "value": signal_mod.spread(buckets),
                 "n_obs": None, "date": None, "detail": None}
            )
            rows.append(
                {"kind": kind, "horizon_months": horizon, "bucket": "all",
                 "metric": "monotonicity", "value": signal_mod.monotonicity(buckets),
                 "n_obs": None, "date": None, "detail": None}
            )

    # score bands, as the interface shows them
    bands = signal_mod.band_returns(scored, horizon_months=3)
    if not bands.is_empty():
        summary["bands"] = bands.to_dicts()
        for row in bands.iter_rows(named=True):
            rows.append(
                {"kind": "band", "horizon_months": 3, "bucket": row["band"],
                 "metric": "mean_excess", "value": row["mean_excess"],
                 "n_obs": row["n"], "date": None, "detail": None}
            )

    # cuts: sector, size, calendar year, regime
    enriched = signal_mod.add_calendar_year(signal_mod.add_size_buckets(scored))
    regimes = signal_mod.classify_regimes(prices, decision_dates)
    if not regimes.is_empty():
        enriched = enriched.join(regimes, on="date", how="left")

    for column, kind in (
        ("sector", "sector"),
        ("size_bucket", "size"),
        ("calendar_year", "year"),
        ("trend_regime", "regime"),
        ("volatility_regime", "regime"),
        ("rate_regime", "regime"),
    ):
        if column not in enriched.columns:
            continue
        table = signal_mod.ic_by_group(enriched, column, horizon_months=3)
        if table.is_empty():
            continue
        summary.setdefault("cuts", {})[column] = table.to_dicts()
        for row in table.iter_rows(named=True):
            rows.append(
                {"kind": kind, "horizon_months": 3, "bucket": f"{column}:{row['group']}",
                 "metric": "ic_mean", "value": row["ic_mean"], "n_obs": row["n_obs"],
                 "date": None, "detail": None}
            )

    # each category on its own, and what removing it does
    categories = signal_mod.category_ic(scored, horizons=spec.ic_horizons)
    if not categories.is_empty():
        summary["categories"] = categories.to_dicts()
        for row in categories.iter_rows(named=True):
            rows.append(
                {"kind": "category", "horizon_months": row["horizon_months"],
                 "bucket": row["category"], "metric": "ic_mean", "value": row["ic_mean"],
                 "n_obs": row["n_dates"], "date": None, "detail": None}
            )

    ablation = signal_mod.ablations(scored, strategy.scoring, horizon_months=3)
    if not ablation.is_empty():
        summary["ablations"] = ablation.to_dicts()
        for row in ablation.iter_rows(named=True):
            rows.append(
                {"kind": "ablation", "horizon_months": 3, "bucket": row["removed"],
                 "metric": "ic_mean", "value": row["ic_mean"], "n_obs": row["n_dates"],
                 "date": None, "detail": None}
            )

    persistence = signal_mod.score_persistence(scored)
    summary["persistence"] = persistence
    summary["holding_period"] = primary["net_summary"]["holding_period"]
    rows.append(
        {"kind": "turnover", "horizon_months": 0, "bucket": "",
         "metric": "score_autocorrelation", "value": persistence["autocorrelation"],
         "n_obs": persistence.get("n_pairs"), "date": None, "detail": None}
    )
    rows.append(
        {"kind": "holding_period", "horizon_months": 0, "bucket": "",
         "metric": "mean_months", "value": summary["holding_period"]["mean_months"],
         "n_obs": summary["holding_period"]["n_spells"], "date": None, "detail": None}
    )
    return {"rows": rows, "summary": summary}


def _measurements(primary: dict, diagnostics: dict) -> dict:
    """The handful of numbers the predefined gates are evaluated against."""
    summary = diagnostics["summary"]
    ic3 = summary.get("information_coefficient", {}).get("3", {})
    decile = summary.get("buckets", {}).get("decile_3m", {})
    spy = primary["net_summary"]["benchmarks"].get("SPY", {})
    return {
        "ic_mean_3m": ic3.get("ic_mean"),
        "ic_t_3m": ic3.get("ic_t"),
        "decile_spread_3m": decile.get("spread"),
        "decile_monotonicity": decile.get("monotonicity"),
        "excess_cagr_net_spy": spy.get("excess_cagr"),
        "information_ratio_net_spy": spy.get("information_ratio"),
    }


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def _robustness(
    *,
    strategy: StrategyVersion,
    spec: ValidationSpec,
    panel: pl.DataFrame,
    market: MarketData,
    period: Period,
    period_label: str,
    decision_dates: list[date],
    benchmarks: dict[str, np.ndarray],
    security_tickers: dict[str, str],
    sectors: dict[str, str],
    base: dict,
    coordinates: dict,
    persist: bool,
    sessions: list[date],
) -> dict:
    """Run every declared robustness variant and compare it with the base run.

    Every test in the battery runs. None is skipped because the previous one
    was disappointing, and the verdict threshold was set in advance, so a test
    cannot be reclassified as "informational" after it fails.
    """
    base_rules = VARIANTS[spec.primary_variant]
    base_excess = base["net_summary"]["benchmarks"].get("SPY", {}).get("excess_cagr")
    base_cagr = base["net_summary"]["cagr"]
    thresholds = spec.robustness
    rows: list[dict] = []
    outcomes: list[dict] = []

    for test in ROBUSTNESS_BATTERY:
        for variant_label, overrides in test.variants:
            rules = base_rules
            costs = strategy.costs
            execution = strategy.execution
            excluded: frozenset[str] = frozenset()
            dates = decision_dates

            if "cost_factor" in overrides:
                costs = strategy.costs.scaled(overrides["cost_factor"])
            if "delay_days" in overrides:
                execution = replace(execution, delay_days=overrides["delay_days"])
            if "rebalance_offset" in overrides:
                # Shifted against the full trading calendar, the same one the
                # panel was scored on, so a month-end at the edge of the period
                # shifts to the same session in both places.
                dates = _shift_dates(decision_dates, sessions, overrides["rebalance_offset"])
            if "max_sector_weight" in overrides:
                rules = replace(base_rules, max_sector_weight=overrides["max_sector_weight"])
            if "weighting" in overrides:
                rules = replace(base_rules, weighting=overrides["weighting"])
            if "top_n" in overrides:
                rules = replace(base_rules, top_n=overrides["top_n"], label=f"top{overrides['top_n']}")
            if "drop_top_contributors" in overrides:
                excluded = _top_contributing_securities(
                    base["net"], overrides["drop_top_contributors"]
                )

            outcome = _run_variant(
                label=spec.primary_variant,
                rules=rules,
                strategy=strategy,
                spec=spec,
                panel=panel,
                market=market,
                period=period,
                period_label=period_label,
                decision_dates=dates,
                benchmarks=benchmarks,
                security_tickers=security_tickers,
                sectors=sectors,
                purpose=f"Robustness: {test.label} ({variant_label}) -- {test.question}",
                trial_type="robustness",
                coordinates=coordinates,
                n_trials=registry.trial_count(strategy.version) + 1,
                persist=persist,
                costs=costs,
                execution=execution,
                excluded_securities=excluded,
                run_suffix=f"-{test.key}-{variant_label.replace(' ', '')}",
            )

            excess = outcome["net_summary"]["benchmarks"].get("SPY", {}).get("excess_cagr")
            verdict = _robustness_verdict(
                base_excess, excess,
                weakened_below=(
                    thresholds.date_shift_weakened_below
                    if test.key == "rebalance_shift"
                    else thresholds.weakened_below
                ),
                failed_below=thresholds.failed_below,
            )
            outcomes.append(
                {
                    "test": test.key,
                    "label": test.label,
                    "question": test.question,
                    "variant": variant_label,
                    "run_id": outcome["run_id"],
                    "cagr": outcome["net_summary"]["cagr"],
                    "excess_cagr_vs_spy": excess,
                    "base_excess_cagr_vs_spy": base_excess,
                    "verdict": verdict,
                }
            )
            for metric, base_value, value in (
                ("excess_cagr_vs_spy", base_excess, excess),
                ("cagr", base_cagr, outcome["net_summary"]["cagr"]),
                ("sharpe", base["net_summary"]["sharpe"], outcome["net_summary"]["sharpe"]),
            ):
                rows.append(
                    {
                        "run_id": outcome["run_id"],
                        "base_run_id": base["run_id"],
                        "test_name": test.key,
                        "parameter": test.label,
                        "parameter_value": variant_label,
                        "metric": metric,
                        "base_value": base_value,
                        "value": value,
                        "delta": (value - base_value)
                        if value is not None and base_value is not None
                        else None,
                        "verdict": verdict,
                    }
                )

    if persist:
        registry.save_robustness(rows)

    survived = sum(1 for o in outcomes if o["verdict"] == "survived")
    weakened = sum(1 for o in outcomes if o["verdict"] == "weakened")
    failed = sum(1 for o in outcomes if o["verdict"] == "failed")
    # A robustness test asks whether an edge survives a change of assumption.
    # When the base run had no excess return there is no edge to preserve, and
    # counting those as failures would read as "the strategy was fragile" when
    # the truth is "there was nothing there to break". They are counted
    # separately and the share is taken over the tests that could apply.
    not_applicable = sum(1 for o in outcomes if o["verdict"] == "not_applicable")
    not_measurable = sum(1 for o in outcomes if o["verdict"] == "not_measurable")
    applicable = len(outcomes) - not_applicable - not_measurable
    return {
        "tests": outcomes,
        "summary": {
            "n_tests": len(outcomes),
            "survived": survived,
            "weakened": weakened,
            "failed": failed,
            "not_applicable": not_applicable,
            "not_measurable": not_measurable,
            "applicable": applicable,
            "survived_share": survived / applicable if applicable else float("nan"),
            "note": (
                "No robustness test applies: the base run produced no excess return over "
                "SPY, so there is no edge whose fragility could be tested."
                if applicable == 0
                else ""
            ),
        },
    }


def _robustness_verdict(
    base: float | None, value: float | None, *, weakened_below: float, failed_below: float
) -> str:
    if base is None or value is None or not np.isfinite(base) or not np.isfinite(value):
        return "not_measurable"
    if base <= 0:
        # Nothing to preserve: the base run had no excess return, so a variant
        # that also has none has not "failed" any test.
        return "not_applicable"
    ratio = value / base
    if value <= failed_below:
        return "failed"
    if ratio < weakened_below:
        return "weakened"
    return "survived"


def _shift_dates(decision_dates: list[date], sessions: list[date], offset: int) -> list[date]:
    """Move every rebalance a fixed number of sessions earlier or later."""
    index = {d: i for i, d in enumerate(sessions)}
    out: list[date] = []
    for d in decision_dates:
        i = index.get(d)
        if i is None:
            continue
        j = i + offset
        if 0 <= j < len(sessions):
            out.append(sessions[j])
    return sorted(set(out))


def _top_contributing_securities(net, n: int) -> frozenset[str]:
    """The n securities that produced the most profit in the base run."""
    ordered = sorted(net.contribution.items(), key=lambda kv: kv[1], reverse=True)
    return frozenset(s for s, _ in ordered[:n])


# ---------------------------------------------------------------------------
# The holdout
# ---------------------------------------------------------------------------


def evaluate_holdout(*, reason: str, strategy_version: str | None = None) -> dict:
    """Open the sealed period. Records an audit entry before returning anything.

    The audit is written whether or not the caller likes the result, and the
    table is append-only, so "how many times has the holdout been seen" is
    always answerable. A holdout examined repeatedly during iteration is not a
    holdout, and this is the mechanism that makes that visible rather than
    forgotten.
    """
    if not reason or len(reason.strip()) < 20:
        raise ValueError(
            "Evaluating the holdout requires a written reason of at least twenty "
            "characters, recorded permanently. If the reason is hard to write, that is "
            "the mechanism working."
        )
    spec = ACTIVE_VALIDATION_SPEC
    strategy = get_strategy(strategy_version)
    prior = registry.holdout_accesses(strategy.version)

    result = run_validation(
        period_label="holdout",
        strategy_version=strategy.version,
        purpose=f"Holdout evaluation: {reason}",
        allow_holdout=True,
    )
    audit_id = f"holdout-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    registry.record_holdout_access(
        audit_id=audit_id,
        reason=reason,
        strategy_version=strategy.version,
        config_checksum=strategy.checksum,
        holdout_start=spec.periods.holdout.start,
        holdout_end=spec.periods.holdout.end,
        run_id=result["variants"][spec.primary_variant]["run_id"],
        results=result["verdict"],
    )
    result["holdout_audit"] = {
        "audit_id": audit_id,
        "reason": reason,
        "previous_accesses": int(prior.height),
        "warning": (
            "This holdout has now been evaluated "
            f"{prior.height + 1} time(s) for {strategy.version}. Each evaluation after the "
            "first weakens it as evidence, because the strategy can be adjusted in "
            "response to what was seen."
        ),
    }
    return result
