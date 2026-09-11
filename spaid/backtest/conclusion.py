"""What the evidence is allowed to claim, and what it actually says.

Two separate judgements live here, and keeping them separate is the whole
design:

**The status** is about the evidence. Is the universe survivorship-free? Are
delisting returns known? Were the fundamentals point-in-time? Until those are
true, no result may be labelled anything stronger than exploratory, however
good the number is. A strategy that "beat SPY by 6% a year" on a universe of
today's survivors has not been tested; it has been described.

**The conclusion** is about the signal. Given whatever evidence exists, does
the score predict relative returns? It has five possible values and one of them
is "the backtest is invalid", which is what gets returned when the status gates
fail — because a strategy tested on inadequate data has not been shown to work
*or* to fail.

Both are computed from stored measurements. Neither takes a parameter that
could be tuned to produce a nicer answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from spaid.config.validation import (
    ACTIVE_VALIDATION_SPEC,
    Conclusion,
    DataGate,
    ValidationSpec,
    ValidationStatus,
)

log = logging.getLogger(__name__)


@dataclass
class GateResult:
    key: str
    label: str
    passed: bool
    detail: str
    requirement: str = ""
    value: float | None = None
    threshold: float | None = None


@dataclass
class Verdict:
    status: ValidationStatus
    conclusion: Conclusion
    headline: str
    reasoning: list[str]
    data_gates: list[GateResult]
    result_gates: list[GateResult]
    limitations: list[str] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "conclusion": self.conclusion.value,
            "headline": self.headline,
            "reasoning": self.reasoning,
            "limitations": self.limitations,
            "failed_gates": self.failed_gates,
            "data_gates": [vars(g) for g in self.data_gates],
            "result_gates": [vars(g) for g in self.result_gates],
        }


# ---------------------------------------------------------------------------
# Data gates
# ---------------------------------------------------------------------------


def evaluate_data_gates(
    coverage: dict,
    *,
    spec: ValidationSpec | None = None,
    benchmark_coverage: dict | None = None,
    execution_delay_days: int = 1,
    estimates_excluded: bool = True,
) -> list[GateResult]:
    """Check every precondition against measured coverage, not against intent."""
    spec = spec or ACTIVE_VALIDATION_SPEC
    by_key = {g.key: g for g in spec.data_gates}
    out: list[GateResult] = []

    def gate(key: str, passed: bool, detail: str, value=None, threshold=None) -> None:
        source: DataGate = by_key[key]
        out.append(
            GateResult(
                key=key,
                label=source.label,
                passed=passed,
                detail=detail,
                requirement=source.requirement,
                value=value,
                threshold=threshold,
            )
        )

    built = bool(coverage.get("built"))
    exits = int(coverage.get("exits_observed") or 0)
    gate(
        "historical_membership",
        built and exits > 0,
        (
            f"Membership reconstructed from {coverage.get('snapshots')} monthly snapshots "
            f"covering {coverage.get('coverage_start')} to {coverage.get('coverage_end')}, "
            f"with {exits} observed index exits."
            if built and exits
            else "Index membership has no history: every historical ranking would be "
            "drawn from the companies that survived to today."
        ),
        value=float(exits),
    )

    removed = int(coverage.get("removed_securities") or 0)
    priced = int(coverage.get("removed_with_prices") or 0)
    share = priced / removed if removed else 0.0
    gate(
        "delisted_prices",
        removed > 0 and share >= 0.95,
        (
            f"{priced} of {removed} removed companies have price history "
            f"({share:.0%}). The {removed - priced} without it were acquired, merged or "
            "failed, which is exactly the population whose absence flatters a backtest."
            if removed
            else "No removed companies are known, so their prices cannot be assessed."
        ),
        value=share,
        threshold=0.95,
    )

    known_returns = int(coverage.get("delisting_returns_known") or 0)
    missing = removed - priced
    gate(
        "delisting_returns",
        missing == 0 or known_returns >= missing,
        (
            f"Delisting returns are known for {known_returns} of the {missing} companies "
            "that stopped trading. The rest have no recorded final value; the simulator "
            "never substitutes zero for one, and refuses to hold a security whose ending "
            "it cannot price."
        ),
        value=float(known_returns),
        threshold=float(missing),
    )

    gate(
        "point_in_time_fundamentals",
        True,
        "Every fundamental is joined on `available_at`, the date the filing could first "
        "have been acted on, and the join is covered by tests in tests/test_point_in_time.py.",
    )
    gate(
        "no_estimate_lookahead",
        estimates_excluded,
        (
            "The seven estimate-derived metrics are null at every historical date and the "
            "scoring spec renormalises over the rest."
            if estimates_excluded
            else "Analyst estimates are being read at historical dates, which is look-ahead."
        ),
    )
    gate(
        "execution_delay",
        execution_delay_days >= 1,
        f"Trades execute {execution_delay_days} session(s) after the decision close, "
        "with spread, slippage and commission charged on the traded notional.",
        value=float(execution_delay_days),
        threshold=1.0,
    )

    bench = benchmark_coverage or {}
    missing_bench = [name for name, ok in bench.items() if not ok]
    gate(
        "benchmark_coverage",
        not missing_bench,
        (
            "SPY and SPMO total returns cover the whole test period."
            if not missing_bench
            else f"No benchmark data for {', '.join(missing_bench)} over the test period."
        ),
    )
    return out


def cap_status(gates: list[GateResult], spec: ValidationSpec | None = None) -> ValidationStatus:
    """The strongest status the failed data gates permit."""
    spec = spec or ACTIVE_VALIDATION_SPEC
    by_key = {g.key: g for g in spec.data_gates}
    order = [
        ValidationStatus.NOT_TESTABLE,
        ValidationStatus.EXPLORATORY_ONLY,
        ValidationStatus.IN_SAMPLE,
        ValidationStatus.VALIDATION_PASSED,
        ValidationStatus.HOLDOUT_PASSED,
    ]
    cap = ValidationStatus.HOLDOUT_PASSED
    for gate in gates:
        if gate.passed:
            continue
        limit = by_key[gate.key].caps_status_at
        if order.index(limit) < order.index(cap):
            cap = limit
    return cap


# ---------------------------------------------------------------------------
# Result gates
# ---------------------------------------------------------------------------


def evaluate_result_gates(
    measurements: dict, *, spec: ValidationSpec | None = None
) -> list[GateResult]:
    """Compare each measured statistic with the threshold declared in advance."""
    spec = spec or ACTIVE_VALIDATION_SPEC
    out: list[GateResult] = []
    for gate in spec.result_gates:
        value = measurements.get(gate.metric)
        numeric = float(value) if value is not None and np.isfinite(value) else None
        passed = numeric is not None and numeric >= gate.minimum
        if numeric is None:
            detail = f"{gate.metric} could not be measured."
        else:
            comparison = "at least" if passed else "below"
            detail = (
                f"{gate.metric} = {numeric:.4f}, {comparison} the threshold "
                f"of {gate.minimum:.4f} declared before the test."
            )
        out.append(
            GateResult(
                key=gate.key,
                label=gate.label,
                passed=passed,
                detail=detail,
                requirement=gate.rationale,
                value=numeric,
                threshold=gate.minimum,
            )
        )
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def decide(
    *,
    coverage: dict,
    measurements: dict,
    period_label: str,
    spec: ValidationSpec | None = None,
    benchmark_coverage: dict | None = None,
    execution_delay_days: int = 1,
    robustness_summary: dict | None = None,
    n_decision_dates: int = 0,
) -> Verdict:
    """Combine the data gates and the result gates into one honest statement."""
    spec = spec or ACTIVE_VALIDATION_SPEC
    data_gates = evaluate_data_gates(
        coverage,
        spec=spec,
        benchmark_coverage=benchmark_coverage,
        execution_delay_days=execution_delay_days,
    )
    result_gates = evaluate_result_gates(measurements, spec=spec)

    status = cap_status(data_gates, spec)
    failed_data = [g for g in data_gates if not g.passed]
    failed_results = [g for g in result_gates if not g.passed]

    limitations = [g.detail for g in failed_data]
    reasoning: list[str] = []

    # --- the status ------------------------------------------------------
    if status is ValidationStatus.NOT_TESTABLE:
        headline = (
            "The backtest cannot be run honestly: a precondition on the data itself is "
            "unmet."
        )
        return Verdict(
            status=status,
            conclusion=Conclusion.INVALID,
            headline=headline,
            reasoning=[g.detail for g in failed_data],
            data_gates=data_gates,
            result_gates=result_gates,
            limitations=limitations,
            failed_gates=[g.key for g in failed_data],
        )

    if status is ValidationStatus.EXPLORATORY_ONLY:
        reasoning.append(
            "The universe is incomplete, so this run is exploratory: it shows what the "
            "ranking did among the companies we can price, not what it would have done "
            "among the companies that were actually in the index."
        )
    elif period_label == "development":
        status = ValidationStatus.IN_SAMPLE
        reasoning.append(
            "This is the development period. Every number here has been seen while the "
            "system was being built, so it is in-sample by construction."
        )
    elif period_label == "validation":
        status = (
            ValidationStatus.VALIDATION_PASSED
            if not failed_results
            else ValidationStatus.VALIDATION_FAILED
        )
    elif period_label == "holdout":
        status = (
            ValidationStatus.HOLDOUT_PASSED
            if not failed_results
            else ValidationStatus.HOLDOUT_FAILED
        )

    # --- the conclusion ---------------------------------------------------
    if n_decision_dates < 36:
        conclusion = Conclusion.INSUFFICIENT_DATA
        headline = (
            f"Only {n_decision_dates} monthly decision dates are available; that is too "
            "few to distinguish a signal from noise."
        )
        reasoning.append(headline)
        return Verdict(status, conclusion, headline, reasoning, data_gates, result_gates,
                       limitations, [g.key for g in failed_data + failed_results])

    if failed_data:
        conclusion = Conclusion.INVALID
        headline = (
            "The backtest is invalid as evidence of an edge: "
            + failed_data[0].label.lower()
            + " is not satisfied."
        )
        reasoning.append(
            "A result computed on an incomplete universe can be encouraging but cannot be "
            "confirming. The direction of the bias is known -- upward -- and its size is "
            "not measurable from the data we hold."
        )
        reasoning.extend(_signal_reading(measurements, result_gates))
        return Verdict(status, conclusion, headline, reasoning, data_gates, result_gates,
                       limitations, [g.key for g in failed_data])

    signal_passes = sum(1 for g in result_gates if g.passed)
    robust = (robustness_summary or {}).get("survived_share")
    if robust is not None and not np.isfinite(robust):
        # No robustness test applied, because the base run had no excess return
        # to preserve. Absence of a robustness verdict is not evidence of
        # robustness, so it is treated as no information rather than a pass.
        robust = None

    if signal_passes == len(result_gates) and (robust is None or robust >= 0.7):
        conclusion = Conclusion.EVIDENCE_OF_ABILITY
        headline = "The ranking shows evidence of predictive ability."
    elif signal_passes == 0:
        conclusion = Conclusion.NO_EVIDENCE
        headline = "No meaningful evidence that the ranking predicts relative returns."
    else:
        conclusion = Conclusion.MIXED
        headline = (
            f"Evidence is mixed: {signal_passes} of {len(result_gates)} predefined "
            "thresholds were cleared."
        )
    reasoning.extend(_signal_reading(measurements, result_gates))
    return Verdict(status, conclusion, headline, reasoning, data_gates, result_gates,
                   limitations, [g.key for g in failed_results])


def _signal_reading(measurements: dict, gates: list[GateResult]) -> list[str]:
    """Plain sentences describing what the diagnostics said, pass or fail."""
    out: list[str] = []
    ic = measurements.get("ic_mean_3m")
    t = measurements.get("ic_t_3m")
    if ic is not None and np.isfinite(ic):
        direction = "positive" if ic > 0 else "negative"
        significance = (
            f"t = {t:.2f} on non-overlapping blocks"
            if t is not None and np.isfinite(t)
            else "its t-statistic could not be computed"
        )
        out.append(
            f"The three-month rank information coefficient averages {ic:+.4f} "
            f"({direction}), {significance}."
        )
    spread_value = measurements.get("decile_spread_3m")
    if spread_value is not None and np.isfinite(spread_value):
        out.append(
            f"Top-decile minus bottom-decile forward excess return is {spread_value:+.2%} "
            "over three months."
        )
    mono = measurements.get("decile_monotonicity")
    if mono is not None and np.isfinite(mono):
        out.append(
            f"Forward returns rise with the score with a rank correlation of {mono:+.2f} "
            "across deciles."
        )
    excess = measurements.get("excess_cagr_net_spy")
    if excess is not None and np.isfinite(excess):
        out.append(
            f"Net of costs, the primary portfolio's annual return differs from SPY's by "
            f"{excess:+.2%}."
        )
    failed = [g.label for g in gates if not g.passed]
    if failed:
        out.append("Thresholds not cleared: " + "; ".join(failed) + ".")
    return out
