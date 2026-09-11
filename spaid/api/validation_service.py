"""Assembling the validation view from stored artefacts.

Reads the JSON summary a validation run wrote, plus the stored daily paths and
trial registry, and turns them into the response models. It computes no
performance statistic of its own: if a number is on the validation screen, some
run produced it and the run's identifier is on the screen beside it.

The one thing this module does decide is wording -- turning a status enum into
the sentence a reader sees. That belongs here rather than in the client for the
same reason every other label does: the CLI and the screen must say the same
thing about the same result.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import date, datetime

import numpy as np
import polars as pl

from spaid.api import validation_schemas as V
from spaid.backtest import provenance
from spaid.backtest.strategy import VARIANTS, get_strategy
from spaid.config.validation import (
    ACTIVE_VALIDATION_SPEC,
    CONCLUSION_LABELS,
    STATUS_LABELS,
    Conclusion,
    ValidationStatus,
)
from spaid.pipeline import security_master
from spaid.storage import store

log = logging.getLogger(__name__)

ROBUSTNESS_VERDICTS = {
    "survived": "Survived",
    "weakened": "Weakened",
    "failed": "Failed",
    "not_applicable": "Not applicable — no excess return to preserve",
    "not_measurable": "Not measurable",
}


def _float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _stored(period_label: str) -> dict | None:
    return store.read_json(f"validation_{period_label}")


def available_periods() -> list[str]:
    return [
        label
        for label in ("development", "validation", "holdout", "full")
        if _stored(label) is not None
    ]


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def _strategy_summary(payload: dict) -> V.StrategySummary:
    strategy = get_strategy(payload.get("strategy_version"))
    stored = store.read("strategy_versions")
    frozen_at = None
    code_commit = payload.get("code_commit")
    status = "frozen"
    if stored is not None and not stored.is_empty():
        row = stored.filter(pl.col("strategy_version") == strategy.version)
        if not row.is_empty():
            frozen_at = row["frozen_at"][0]
            code_commit = row["code_commit"][0] or code_commit
            status = row["status"][0]
    return V.StrategySummary(
        version=strategy.version,
        name=strategy.name,
        scoring_version=strategy.scoring.version,
        config_checksum=payload.get("config_checksum", strategy.checksum),
        code_commit=code_commit,
        frozen_at=frozen_at,
        status=status,
        notes=strategy.notes,
        n_metrics=len(strategy.scoring.metrics),
        n_metrics_point_in_time=len(strategy.available_metrics()),
        unavailable_point_in_time=dict(strategy.unavailable_point_in_time),
        category_weights={c.value: w for c, w in strategy.scoring.category_weights.items()},
        category_coverage_point_in_time=payload.get(
            "point_in_time_coverage", strategy.category_coverage_point_in_time()
        ),
    )


def _universe_status(coverage: dict) -> V.UniverseStatus:
    if not coverage.get("built"):
        return V.UniverseStatus(
            built=False,
            reason=coverage.get("reason"),
            survivorship_status="not removed",
            survivorship_detail=(
                "Index membership has no history, so every historical ranking would be "
                "drawn from the companies that are in the index today."
            ),
        )

    removed = int(coverage.get("removed_securities") or 0)
    priced = int(coverage.get("removed_with_prices") or 0)
    missing = removed - priced
    if missing == 0:
        status = "removed"
        detail = (
            f"All {removed} companies the index dropped are priced through their "
            "membership, so the historical universe is complete."
        )
    else:
        status = "partially removed"
        detail = (
            f"Membership is reconstructed with {coverage.get('exits_observed')} observed "
            f"exits, and {priced} of {removed} removed companies are priced. The remaining "
            f"{missing} were acquired, merged or failed and no free source carries their "
            "history, so the universe still leans toward survivors and every result is "
            "biased upward by an amount that cannot be measured from this data."
        )

    version_row = store.read("universe_versions")
    method = None
    if version_row is not None and not version_row.is_empty():
        method = version_row.sort("built_at").tail(1)["method"][0]

    return V.UniverseStatus(
        built=True,
        universe_version=coverage.get("universe_version"),
        method=method,
        coverage_start=coverage.get("coverage_start"),
        coverage_end=coverage.get("coverage_end"),
        snapshots=coverage.get("snapshots"),
        securities=coverage.get("securities"),
        spells=coverage.get("spells"),
        exits_observed=coverage.get("exits_observed"),
        removed_securities=removed,
        removed_with_prices=priced,
        removed_without_prices=coverage.get("removed_without_prices"),
        delisting_returns_known=coverage.get("delisting_returns_known"),
        survivorship_status=status,
        survivorship_detail=detail,
    )


def _verdict(payload: dict) -> V.Verdict:
    raw = payload.get("verdict", {})
    status = ValidationStatus(raw.get("status", "not_testable"))
    conclusion = Conclusion(raw.get("conclusion", "insufficient_data"))
    return V.Verdict(
        status=status.value,
        status_label=STATUS_LABELS[status],
        conclusion=conclusion.value,
        conclusion_label=CONCLUSION_LABELS[conclusion],
        headline=raw.get("headline", ""),
        reasoning=raw.get("reasoning", []),
        limitations=raw.get("limitations", []),
        failed_gates=raw.get("failed_gates", []),
        data_gates=[V.Gate(**g) for g in raw.get("data_gates", [])],
        result_gates=[V.Gate(**g) for g in raw.get("result_gates", [])],
    )


def _periods(current: str) -> list[V.PeriodInfo]:
    spec = ACTIVE_VALIDATION_SPEC
    audit = store.read("holdout_audit")
    accesses = 0 if audit is None else audit.height
    out: list[V.PeriodInfo] = []
    for period in (spec.periods.development, spec.periods.validation, spec.periods.holdout):
        sealed = period.label == "holdout" and accesses == 0
        out.append(
            V.PeriodInfo(
                label=period.label,
                start=period.start,
                end=period.end,
                purpose=period.purpose,
                years=round(period.years, 2),
                evaluated=_stored(period.label) is not None,
                sealed=sealed,
                accesses=accesses if period.label == "holdout" else 0,
            )
        )
    if spec.periods.paper_starts is None:
        out.append(
            V.PeriodInfo(
                label="paper",
                start=date.today(),
                end=date.today(),
                purpose=(
                    "Begins on the first day a recommendation is recorded in the research "
                    "journal. No entries exist yet, so paper tracking has not started."
                ),
                years=0.0,
                evaluated=False,
            )
        )
    del current
    return out


def _benchmarks(entry: dict) -> list[V.BenchmarkComparison]:
    out: list[V.BenchmarkComparison] = []
    for name, values in (entry.get("benchmarks") or {}).items():
        out.append(
            V.BenchmarkComparison(
                benchmark=name,
                cagr=_float(values.get("cagr")),
                total_return=_float(values.get("total_return")),
                excess_cagr=_float(values.get("excess_cagr")),
                alpha_annual=_float(values.get("alpha_annual")),
                beta=_float(values.get("beta")),
                tracking_error=_float(values.get("tracking_error")),
                information_ratio=_float(values.get("information_ratio")),
                upside_capture=_float(values.get("upside_capture")),
                downside_capture=_float(values.get("downside_capture")),
                volatility=_float(values.get("volatility")),
                sharpe=_float(values.get("sharpe")),
                max_drawdown=_float(values.get("max_drawdown")),
                rolling_12m_beat_share=_float((values.get("rolling_12m") or {}).get("share")),
                rolling_36m_beat_share=_float((values.get("rolling_36m") or {}).get("share")),
                rolling_60m_beat_share=_float((values.get("rolling_60m") or {}).get("share")),
            )
        )
    return out


def _variants(payload: dict) -> list[V.VariantPerformance]:
    primary = payload.get("primary_variant")
    out: list[V.VariantPerformance] = []
    for label, entry in (payload.get("variants") or {}).items():
        net = entry.get("net", {})
        gross = entry.get("gross", {})
        rules = VARIANTS.get(label)
        deflated = net.get("deflated_sharpe") or {}
        out.append(
            V.VariantPerformance(
                label=label,
                description=rules.description if rules else "",
                is_primary=label == primary,
                run_id=entry.get("run_id", ""),
                net_cagr=_float(net.get("cagr")),
                gross_cagr=_float(gross.get("cagr")),
                total_return=_float(net.get("total_return")),
                volatility=_float(net.get("volatility")),
                sharpe=_float(net.get("sharpe")),
                sortino=_float(net.get("sortino")),
                max_drawdown=_float(net.get("max_drawdown")),
                drawdown_duration_days=(net.get("drawdown_duration") or {}).get("days"),
                calmar=_float(net.get("calmar")),
                positive_months=_float(net.get("positive_months")),
                turnover_annual_one_way=_float(net.get("turnover_annual_one_way")),
                total_costs=_float(net.get("total_costs")),
                n_trades=net.get("trades"),
                n_delistings=net.get("delistings"),
                mean_holding_months=_float(
                    (net.get("holding_period") or {}).get("mean_months")
                ),
                cagr_interval=V.Interval(**net["cagr_ci"]) if net.get("cagr_ci") else None,
                sharpe_interval=V.Interval(**net["sharpe_ci"]) if net.get("sharpe_ci") else None,
                deflated_sharpe_probability=_float(deflated.get("probability")),
                deflated_sharpe_verdict=deflated.get("verdict"),
                benchmarks=_benchmarks(net),
                best_month=net.get("best_month"),
                worst_month=net.get("worst_month"),
                best_year=net.get("best_year"),
                worst_year=net.get("worst_year"),
                yearly_returns=net.get("yearly_returns", []),
            )
        )
    out.sort(key=lambda v: (not v.is_primary, v.label))
    return out


def _equity_curve(run_id: str, benchmarks: list[str], *, max_points: int = 900) -> list[V.EquityPoint]:
    """The stored daily path, thinned for the chart but never smoothed.

    Thinning keeps every nth session rather than averaging, so a drawdown on the
    chart is a drawdown that happened rather than an artefact of the resampling.
    The last point is always kept, so the chart ends where the run ended.
    """
    daily = store.read("backtest_daily")
    if daily is None or daily.is_empty():
        return []
    rows = daily.filter(pl.col("run_id") == run_id).sort("date")
    if rows.is_empty():
        return []

    bench = store.read("benchmark_values")
    bench_wide: dict[str, dict[date, float]] = {}
    if bench is not None and not bench.is_empty():
        for name in benchmarks:
            series = bench.filter(pl.col("benchmark") == name)
            if not series.is_empty():
                bench_wide[name] = dict(
                    series.select(["date", "total_return_index"]).iter_rows()
                )

    step = max(1, rows.height // max_points)
    start_equity = float(rows["net_equity"][0])
    start_gross = float(rows["gross_equity"][0])
    keep = list(range(0, rows.height, step))
    if keep[-1] != rows.height - 1:
        keep.append(rows.height - 1)

    records = rows.to_dicts()
    out: list[V.EquityPoint] = []
    for i in keep:
        row = records[i]
        day = row["date"]
        out.append(
            V.EquityPoint(
                date=day,
                strategy=float(row["net_equity"]) / start_equity,
                strategy_gross=float(row["gross_equity"]) / start_gross,
                drawdown=_float(row["drawdown"]),
                benchmarks={
                    name: values[day] for name, values in bench_wide.items() if day in values
                },
            )
        )
    return out


def _assess_ic(ic_mean: float | None, ic_t: float | None) -> str:
    if ic_mean is None:
        return "Not measurable."
    if ic_mean <= 0:
        return (
            "Negative: over this period the ranking put the weaker performers higher, "
            "on average."
        )
    if ic_t is None or abs(ic_t) < 1.0:
        return "Positive but indistinguishable from noise at this sample size."
    if ic_t < 2.0:
        return "Positive, but short of the two-standard-error bar set in advance."
    return "Positive and statistically distinguishable from zero."


def _signal(payload: dict) -> V.SignalSummary | None:
    signal = payload.get("signal")
    if not signal:
        return None

    ic_rows: list[V.IcRow] = []
    for horizon, entry in sorted(
        (signal.get("information_coefficient") or {}).items(), key=lambda kv: int(kv[0])
    ):
        ic_rows.append(
            V.IcRow(
                horizon_months=int(horizon),
                ic_mean=_float(entry.get("ic_mean")),
                ic_median=_float(entry.get("ic_median")),
                ic_std=_float(entry.get("ic_std")),
                ic_t=_float(entry.get("ic_t")),
                ic_ir=_float(entry.get("ic_ir")),
                hit_rate=_float(entry.get("hit_rate")),
                n_dates=entry.get("n_dates"),
                assessment=_assess_ic(_float(entry.get("ic_mean")), _float(entry.get("ic_t"))),
            )
        )

    series: list[V.IcPoint] = []
    diagnostics = store.read("signal_diagnostics")
    run_id = (payload.get("variants") or {}).get(payload.get("primary_variant"), {}).get("run_id")
    if diagnostics is not None and not diagnostics.is_empty() and run_id:
        points = diagnostics.filter(
            (pl.col("run_id") == run_id)
            & (pl.col("kind") == "ic_series")
            & (pl.col("horizon_months") == 3)
        ).sort("date")
        series = [
            V.IcPoint(date=row["date"], ic=row["value"], n=row["n_obs"])
            for row in points.iter_rows(named=True)
            if row["value"] is not None
        ]

    buckets = signal.get("buckets") or {}

    def bucket_rows(key: str) -> list[V.BucketRow]:
        entry = buckets.get(key) or {}
        return [
            V.BucketRow(
                bucket=str(row.get("bucket")),
                mean_excess=_float(row.get("mean_excess")),
                median_excess=_float(row.get("median_excess")),
                hit_rate=_float(row.get("hit_rate")),
                mean_score=_float(row.get("mean_score")),
                n=row.get("n"),
            )
            for row in entry.get("rows", [])
        ]

    def group_rows(name: str) -> list[V.GroupRow]:
        return [
            V.GroupRow(
                group=str(row.get("group")),
                ic_mean=_float(row.get("ic_mean")),
                ic_t=_float(row.get("ic_t")),
                hit_rate=_float(row.get("hit_rate")),
                n_obs=row.get("n_obs"),
            )
            for row in (signal.get("cuts") or {}).get(name, [])
        ]

    regime_rows = (
        group_rows("trend_regime") + group_rows("volatility_regime") + group_rows("rate_regime")
    )

    ablations = [
        V.AblationRow(
            removed=row.get("removed", ""),
            ic_mean=_float(row.get("ic_mean")),
            ic_t=_float(row.get("ic_t")),
            delta_ic=_float(row.get("delta_ic")),
            reading=_ablation_reading(_float(row.get("delta_ic")), row.get("removed", "")),
        )
        for row in signal.get("ablations", [])
    ]

    decile = buckets.get("decile_3m") or {}
    holding = signal.get("holding_period") or {}
    return V.SignalSummary(
        ic=ic_rows,
        ic_series=series,
        deciles=bucket_rows("decile_3m"),
        quintiles=bucket_rows("quintile_3m"),
        bands=[
            V.BucketRow(
                bucket=str(row.get("band")),
                mean_excess=_float(row.get("mean_excess")),
                median_excess=_float(row.get("median_excess")),
                hit_rate=_float(row.get("hit_rate")),
                mean_score=_float(row.get("mean_score")),
                n=row.get("n"),
            )
            for row in signal.get("bands", [])
        ],
        decile_spread=_float(decile.get("spread")),
        decile_monotonicity=_float(decile.get("monotonicity")),
        by_sector=group_rows("sector"),
        by_size=group_rows("size_bucket"),
        by_year=group_rows("calendar_year"),
        by_regime=regime_rows,
        categories=[
            V.CategoryRow(
                category=row.get("category", ""),
                horizon_months=row.get("horizon_months", 0),
                ic_mean=_float(row.get("ic_mean")),
                ic_t=_float(row.get("ic_t")),
                ic_ir=_float(row.get("ic_ir")),
                hit_rate=_float(row.get("hit_rate")),
            )
            for row in signal.get("categories", [])
        ],
        ablations=ablations,
        score_autocorrelation=_float((signal.get("persistence") or {}).get("autocorrelation")),
        mean_holding_months=_float(holding.get("mean_months")),
        share_single_month_holdings=_float(holding.get("share_single_month")),
    )


def _ablation_reading(delta: float | None, removed: str) -> str:
    if removed.startswith("nothing") or delta is None:
        return "The composite as it stands."
    if delta > 0.002:
        return (
            f"Removing {removed} improves the information coefficient, so on this evidence "
            "it is carrying weight it has not earned."
        )
    if delta < -0.002:
        return f"Removing {removed} weakens the signal, so it is contributing."
    return f"Removing {removed} changes little either way."


def _robustness(payload: dict) -> list[V.RobustnessRow]:
    return [
        V.RobustnessRow(
            test=row.get("test", ""),
            label=row.get("label", ""),
            question=row.get("question", ""),
            variant=row.get("variant", ""),
            cagr=_float(row.get("cagr")),
            excess_cagr_vs_spy=_float(row.get("excess_cagr_vs_spy")),
            base_excess_cagr_vs_spy=_float(row.get("base_excess_cagr_vs_spy")),
            verdict=row.get("verdict", ""),
            verdict_label=ROBUSTNESS_VERDICTS.get(row.get("verdict", ""), row.get("verdict", "")),
        )
        for row in (payload.get("robustness") or {}).get("tests", [])
    ]


def _contribution(rows: list[dict], key: str) -> list[V.ContributionRow]:
    total = sum(abs(float(r.get("profit", 0.0))) for r in rows) or 1.0
    out = [
        V.ContributionRow(
            name=str(r.get(key) or r.get("ticker") or r.get("sector") or ""),
            profit=float(r.get("profit", 0.0)),
            share=float(r.get("profit", 0.0)) / total,
        )
        for r in rows
    ]
    out.sort(key=lambda r: r.profit, reverse=True)
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def get_report(period_label: str | None = None) -> V.ValidationReport:
    """The whole validation view for one period, from stored artefacts only."""
    periods = available_periods()
    label = period_label or (periods[0] if periods else "development")
    payload = _stored(label)

    if payload is None:
        coverage = security_master.coverage_report()
        return V.ValidationReport(
            available=False,
            period_label=label,
            message=(
                f"No validation run has been stored for the {label} period. Run "
                f"`spaid validate --period {label}` to produce one."
            ),
            universe=_universe_status(coverage),
            periods=_periods(label),
            data_version=provenance.data_version(),
            code_commit=provenance.code_commit(),
            universe_version=provenance.universe_version(),
        )

    spec = ACTIVE_VALIDATION_SPEC
    variants = _variants(payload)
    primary = next((v for v in variants if v.is_primary), variants[0] if variants else None)
    strategy = get_strategy(payload.get("strategy_version"))

    period = next(
        (
            V.PeriodInfo(
                label=p.label, start=p.start, end=p.end, purpose=p.purpose,
                years=round(p.years, 2), evaluated=True,
            )
            for p in (spec.periods.development, spec.periods.validation, spec.periods.holdout)
            if p.label == label
        ),
        None,
    )

    net = (payload.get("variants") or {}).get(payload.get("primary_variant"), {}).get("net", {})
    return V.ValidationReport(
        available=True,
        period_label=label,
        generated_at=_parse_datetime(payload.get("recorded_at")),
        strategy=_strategy_summary(payload),
        universe=_universe_status(payload.get("coverage") or {}),
        verdict=_verdict(payload),
        period=period,
        periods=_periods(label),
        decision_dates=payload.get("decision_dates", 0),
        universe_per_date=payload.get("universe_per_date", {}),
        data_version=payload.get("data_version"),
        code_commit=payload.get("code_commit"),
        universe_version=payload.get("universe_version"),
        variants=variants,
        primary_variant=payload.get("primary_variant"),
        equity_curve=_equity_curve(primary.run_id, list(strategy.benchmarks) + ["SPY (risk-matched)"])
        if primary
        else [],
        signal=_signal(payload),
        robustness=_robustness(payload),
        robustness_summary=(payload.get("robustness") or {}).get("summary", {}),
        overfitting=V.OverfittingSummary(
            **{
                k: v
                for k, v in (payload.get("probability_of_overfitting") or {}).items()
                if k in V.OverfittingSummary.model_fields
            }
        )
        if payload.get("probability_of_overfitting")
        else None,
        contribution_by_stock=_contribution(net.get("contribution_by_stock", []), "ticker"),
        contribution_by_sector=_contribution(net.get("contribution_by_sector", []), "sector"),
        costs=strategy.costs.to_dict(),
        execution={
            "delay_days": strategy.execution.delay_days,
            "rebalance": strategy.rebalance,
            "cash_return_annual": strategy.execution.cash_return_annual,
        },
    )


def _parse_datetime(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def get_trials(limit: int | None = None) -> V.TrialRegistry:
    """Every trial ever registered, newest first, including the failures."""
    trials = store.read("experiments")
    if trials is None or trials.is_empty():
        return V.TrialRegistry(
            trials=[],
            n_trials=0,
            note="No trials have been registered yet.",
        )

    rows = trials.sort("created_at", descending=True)
    if limit:
        rows = rows.head(limit)

    out: list[V.TrialRow] = []
    for row in rows.iter_rows(named=True):
        results = {}
        with contextlib.suppress(json.JSONDecodeError):
            results = json.loads(row["results"] or "{}")
        out.append(
            V.TrialRow(
                trial_id=row["trial_id"],
                created_at=row["created_at"],
                purpose=row["purpose"] or "",
                strategy_version=row["strategy_version"],
                config_checksum=row["config_checksum"] or "",
                code_commit=row["code_commit"],
                data_version=row["data_version"],
                universe_version=row["universe_version"],
                period_label=row["period_label"] or "",
                period_start=row["period_start"],
                period_end=row["period_end"],
                outcome=row["outcome"] or "",
                net_cagr=_float(results.get("net_cagr")),
                excess_vs_spy=_float(results.get("excess_vs_spy")),
                notes=row["notes"] or "",
            )
        )
    return V.TrialRegistry(
        trials=out,
        n_trials=trials.height,
        note=(
            f"{trials.height} trials registered. The count feeds the deflated Sharpe ratio, "
            "so trials are never deleted -- removing the disappointing ones would inflate "
            "the significance of whatever remained."
        ),
    )
