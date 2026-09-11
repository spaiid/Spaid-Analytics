"""Response models for the validation view.

Same contract as the rest of the API: every judgement is made in Python and
arrives at the client as a label with an explanation beside it. The interface
has no rule anywhere that decides whether an information coefficient is good,
what status a run may claim, or which limitation matters -- because those are
exactly the decisions that must not drift between the CLI, the stored artefact
and the screen.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class Gate(BaseModel):
    """One precondition or threshold, with the measurement behind it."""

    key: str
    label: str
    passed: bool
    detail: str
    requirement: str = ""
    value: float | None = None
    threshold: float | None = None


class Verdict(BaseModel):
    status: str
    status_label: str
    conclusion: str
    conclusion_label: str
    headline: str
    reasoning: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    failed_gates: list[str] = Field(default_factory=list)
    data_gates: list[Gate] = Field(default_factory=list)
    result_gates: list[Gate] = Field(default_factory=list)


class StrategySummary(BaseModel):
    version: str
    name: str
    scoring_version: str
    config_checksum: str
    code_commit: str | None = None
    frozen_at: datetime | None = None
    status: str = "frozen"
    notes: str = ""
    n_metrics: int = 0
    n_metrics_point_in_time: int = 0
    unavailable_point_in_time: dict[str, str] = Field(default_factory=dict)
    category_weights: dict[str, float] = Field(default_factory=dict)
    category_coverage_point_in_time: dict[str, float] = Field(default_factory=dict)


class UniverseStatus(BaseModel):
    """What the historical universe can and cannot support, as measurements."""

    built: bool
    universe_version: str | None = None
    method: str | None = None
    coverage_start: str | None = None
    coverage_end: str | None = None
    snapshots: int | None = None
    securities: int | None = None
    spells: int | None = None
    exits_observed: int | None = None
    removed_securities: int | None = None
    removed_with_prices: int | None = None
    removed_without_prices: int | None = None
    delisting_returns_known: int | None = None
    survivorship_status: str = "unknown"
    survivorship_detail: str = ""
    reason: str | None = None


class PeriodInfo(BaseModel):
    label: str
    start: date
    end: date
    purpose: str
    years: float
    evaluated: bool = False
    sealed: bool = False
    accesses: int = 0


class BenchmarkComparison(BaseModel):
    benchmark: str
    cagr: float | None = None
    total_return: float | None = None
    excess_cagr: float | None = None
    alpha_annual: float | None = None
    beta: float | None = None
    tracking_error: float | None = None
    information_ratio: float | None = None
    upside_capture: float | None = None
    downside_capture: float | None = None
    volatility: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    rolling_12m_beat_share: float | None = None
    rolling_36m_beat_share: float | None = None
    rolling_60m_beat_share: float | None = None


class Interval(BaseModel):
    low: float | None = None
    high: float | None = None
    level: float = 0.95
    method: str = ""


class VariantPerformance(BaseModel):
    """One portfolio variant, gross and net, against every benchmark."""

    label: str
    description: str = ""
    is_primary: bool = False
    run_id: str
    net_cagr: float | None = None
    gross_cagr: float | None = None
    total_return: float | None = None
    volatility: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float | None = None
    drawdown_duration_days: int | None = None
    calmar: float | None = None
    positive_months: float | None = None
    turnover_annual_one_way: float | None = None
    total_costs: float | None = None
    n_trades: int | None = None
    n_delistings: int | None = None
    mean_holding_months: float | None = None
    cagr_interval: Interval | None = None
    sharpe_interval: Interval | None = None
    deflated_sharpe_probability: float | None = None
    deflated_sharpe_verdict: str | None = None
    benchmarks: list[BenchmarkComparison] = Field(default_factory=list)
    best_month: dict | None = None
    worst_month: dict | None = None
    best_year: dict | None = None
    worst_year: dict | None = None
    yearly_returns: list[dict] = Field(default_factory=list)


class EquityPoint(BaseModel):
    date: date
    strategy: float
    strategy_gross: float | None = None
    drawdown: float | None = None
    benchmarks: dict[str, float] = Field(default_factory=dict)


class BucketRow(BaseModel):
    bucket: str
    mean_excess: float | None = None
    median_excess: float | None = None
    hit_rate: float | None = None
    mean_score: float | None = None
    n: int | None = None


class IcRow(BaseModel):
    horizon_months: int
    ic_mean: float | None = None
    ic_median: float | None = None
    ic_std: float | None = None
    ic_t: float | None = None
    ic_ir: float | None = None
    hit_rate: float | None = None
    n_dates: int | None = None
    assessment: str = ""


class IcPoint(BaseModel):
    date: date
    ic: float
    n: int | None = None


class GroupRow(BaseModel):
    group: str
    ic_mean: float | None = None
    ic_t: float | None = None
    hit_rate: float | None = None
    n_obs: int | None = None


class CategoryRow(BaseModel):
    category: str
    horizon_months: int
    ic_mean: float | None = None
    ic_t: float | None = None
    ic_ir: float | None = None
    hit_rate: float | None = None


class AblationRow(BaseModel):
    removed: str
    ic_mean: float | None = None
    ic_t: float | None = None
    delta_ic: float | None = None
    reading: str = ""


class SignalSummary(BaseModel):
    """Whether the ranking carries information, separately from the portfolio."""

    ic: list[IcRow] = Field(default_factory=list)
    ic_series: list[IcPoint] = Field(default_factory=list)
    ic_series_horizon_months: int = 3
    deciles: list[BucketRow] = Field(default_factory=list)
    quintiles: list[BucketRow] = Field(default_factory=list)
    bands: list[BucketRow] = Field(default_factory=list)
    decile_spread: float | None = None
    decile_monotonicity: float | None = None
    by_sector: list[GroupRow] = Field(default_factory=list)
    by_size: list[GroupRow] = Field(default_factory=list)
    by_year: list[GroupRow] = Field(default_factory=list)
    by_regime: list[GroupRow] = Field(default_factory=list)
    categories: list[CategoryRow] = Field(default_factory=list)
    ablations: list[AblationRow] = Field(default_factory=list)
    score_autocorrelation: float | None = None
    mean_holding_months: float | None = None
    share_single_month_holdings: float | None = None


class RobustnessRow(BaseModel):
    test: str
    label: str
    question: str
    variant: str
    cagr: float | None = None
    excess_cagr_vs_spy: float | None = None
    base_excess_cagr_vs_spy: float | None = None
    verdict: str
    verdict_label: str = ""


class ContributionRow(BaseModel):
    name: str
    profit: float
    share: float | None = None


class TrialRow(BaseModel):
    trial_id: str
    created_at: datetime
    purpose: str
    strategy_version: str
    config_checksum: str
    code_commit: str | None = None
    data_version: str | None = None
    universe_version: str | None = None
    period_label: str
    period_start: date | None = None
    period_end: date | None = None
    outcome: str
    net_cagr: float | None = None
    excess_vs_spy: float | None = None
    notes: str = ""


class OverfittingSummary(BaseModel):
    feasible: bool
    pbo: float | None = None
    n_trials: int | None = None
    n_partitions: int | None = None
    verdict: str = ""
    reason: str = ""


class ValidationReport(BaseModel):
    """Everything the validation view shows for one period."""

    available: bool
    period_label: str
    message: str | None = None
    generated_at: datetime | None = None

    strategy: StrategySummary | None = None
    universe: UniverseStatus | None = None
    verdict: Verdict | None = None

    period: PeriodInfo | None = None
    periods: list[PeriodInfo] = Field(default_factory=list)
    decision_dates: int = 0
    universe_per_date: dict = Field(default_factory=dict)

    data_version: str | None = None
    code_commit: str | None = None
    universe_version: str | None = None

    variants: list[VariantPerformance] = Field(default_factory=list)
    primary_variant: str | None = None
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    signal: SignalSummary | None = None
    robustness: list[RobustnessRow] = Field(default_factory=list)
    robustness_summary: dict = Field(default_factory=dict)
    overfitting: OverfittingSummary | None = None
    contribution_by_stock: list[ContributionRow] = Field(default_factory=list)
    contribution_by_sector: list[ContributionRow] = Field(default_factory=list)
    costs: dict = Field(default_factory=dict)
    execution: dict = Field(default_factory=dict)


class TrialRegistry(BaseModel):
    trials: list[TrialRow] = Field(default_factory=list)
    n_trials: int = 0
    note: str = ""
