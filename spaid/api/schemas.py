"""API response models.

These are the contract between the domain layer and the interface. Two rules
follow from the product brief and are enforced by the shapes here:

* **The frontend never computes a canonical metric.** Anything the interface
  displays as a number, a label, a band or a verdict is computed in Python and
  serialised. The client formats and arranges; it does not decide what a score
  means or whether an information coefficient is good.
* **Missing is explicit.** Every value that can be absent is Optional, and
  wherever absence has a reason there is a companion field carrying it. A null
  with no explanation is a bug.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["good", "warning", "critical", "neutral"]


class Assessment(BaseModel):
    """A qualitative reading of a number, decided server-side."""

    level: Level
    text: str


class DataFreshness(BaseModel):
    as_of: date | None = None
    age_hours: float | None = None
    is_stale: bool = False
    detail: str | None = None


class MetricDetail(BaseModel):
    """One scored metric, with everything needed to explain it."""

    key: str
    label: str
    category: str
    description: str = ""
    raw_value: float | None = None
    display_value: str | None = None
    unit: str = "ratio"
    score: float | None = Field(None, description="0-100 within the peer group")
    percentile: float | None = None
    peer_basis: str | None = None
    peer_group: str | None = None
    peer_count: int | None = None
    peer_median: float | None = None
    peer_median_display: str | None = None
    weight: float = 0.0
    contribution: float | None = Field(
        None, description="Points this metric contributed to its category score"
    )
    status: str = "scored"
    status_detail: str | None = None
    direction: int = 1
    inputs_as_of: date | None = None


class CategoryDetail(BaseModel):
    key: str
    label: str
    score: float | None = None
    weight: float
    contribution: float | None = None
    coverage: float
    n_metrics: int
    n_missing: int
    assessment: Assessment | None = None
    metrics: list[MetricDetail] = Field(default_factory=list)


class ScenarioValue(BaseModel):
    label: str
    value_per_share: float
    probability: float
    discount_rate: float | None = None
    terminal_growth: float | None = None
    summary: str | None = None


class MethodValue(BaseModel):
    method: str
    label: str
    value_per_share: float | None = None
    weight: float
    used: bool
    reason: str | None = Field(None, description="Why this method was down-weighted or skipped")


class Sensitivity(BaseModel):
    assumption: str
    description: str
    low_value: float
    high_value: float
    base_value: float
    swing_pct: float | None = None


class FairValue(BaseModel):
    price: float
    bear: float | None = None
    base: float | None = None
    bull: float | None = None
    range_low: float | None = None
    range_high: float | None = None
    midpoint: float | None = None
    upside: float | None = Field(None, description="Fraction to the midpoint, not a percentage")
    classification: str
    classification_label: str
    confidence: float | None = None
    confidence_label: str | None = None
    business_model: str
    methods: list[MethodValue] = Field(default_factory=list)
    scenarios: list[ScenarioValue] = Field(default_factory=list)
    sensitivities: list[Sensitivity] = Field(default_factory=list)
    assumptions: dict = Field(default_factory=dict)
    caveats: list[str] = Field(
        default_factory=list, description="Reasons this estimate may be wrong"
    )
    method_dispersion: float | None = None
    spec_version: str


class RiskDetail(BaseModel):
    volatility: float | None = None
    beta: float | None = None
    downside_beta: float | None = None
    downside_deviation: float | None = None
    max_drawdown_1y: float | None = None
    max_drawdown_5y: float | None = None
    dollar_volume_median: float | None = None
    spread_bps: float | None = None
    days_to_earnings: int | None = None
    distress_score: float | None = None
    distress_label: str | None = None
    piotroski: int | None = None
    risk_score: float | None = None
    risk_label: str | None = None
    assessment: Assessment | None = None


class TrapSignal(BaseModel):
    key: str
    label: str
    fired: bool
    value: float | None = None
    detail: str | None = None


class ValueTrap(BaseModel):
    trap_score: float | None = None
    classification: str
    classification_label: str
    signals: list[TrapSignal] = Field(default_factory=list)
    n_fired: int = 0


class ConfidenceComponent(BaseModel):
    key: str
    label: str
    score: float
    weight: float
    detail: str | None = None


class Confidence(BaseModel):
    score: float
    label: str
    components: list[ConfidenceComponent] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ScoreChange(BaseModel):
    window: str
    delta: float | None = None
    detail: str | None = None


class OpportunityRow(BaseModel):
    """One row of the ranked list."""

    rank: int | None = None
    company_id: str
    ticker: str
    name: str
    sector: str | None = None
    industry: str | None = None
    business_model: str | None = None
    price: float | None = None
    score: float | None = None
    band: str | None = None
    quality: float | None = None
    growth: float | None = None
    momentum: float | None = None
    valuation: float | None = None
    coverage: float | None = None
    confidence: float | None = None
    confidence_label: str | None = None
    valuation_class: str | None = None
    valuation_label: str | None = None
    # A compact form for the ranked table, where "Insufficient confidence to
    # classify" does not fit in a column. Supplied by the API rather than
    # truncated by the client, because shortening a verdict is still deciding
    # what it says.
    valuation_label_short: str | None = None
    upside: float | None = None
    fair_value_low: float | None = None
    fair_value_high: float | None = None
    risk_score: float | None = None
    trap_class: str | None = None
    score_change_1w: float | None = None
    score_change_1m: float | None = None
    market_cap: float | None = None
    top_reason: str | None = None
    days_to_earnings: int | None = None


class OpportunitiesResponse(BaseModel):
    as_of: date | None = None
    universe_size: int = 0
    scored_count: int = 0
    rows: list[OpportunityRow] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    bands: list[str] = Field(default_factory=list)
    spec_version: str | None = None
    freshness: DataFreshness = Field(default_factory=DataFreshness)
    notes: list[str] = Field(default_factory=list)


class FinancialTrendPoint(BaseModel):
    period_end: date
    available_at: date
    value: float


class FinancialTrend(BaseModel):
    concept: str
    label: str
    unit: str = "currency"
    points: list[FinancialTrendPoint] = Field(default_factory=list)


class PeerRow(BaseModel):
    company_id: str
    ticker: str
    name: str
    score: float | None = None
    quality: float | None = None
    growth: float | None = None
    momentum: float | None = None
    valuation: float | None = None
    ev_ebit: float | None = None
    fcf_yield: float | None = None
    revenue_growth_1y: float | None = None
    operating_margin: float | None = None
    market_cap: float | None = None
    is_self: bool = False


class SourceRecord(BaseModel):
    field: str
    source: str
    as_of: date | None = None
    detail: str | None = None


class StockDetail(BaseModel):
    company_id: str
    ticker: str
    name: str
    sector: str | None = None
    industry: str | None = None
    business_model: str | None = None
    as_of: date | None = None
    price: float | None = None
    price_change_1d: float | None = None
    market_cap: float | None = None
    # Score
    score: float | None = None
    rank: int | None = None
    universe_size: int | None = None
    percentile: float | None = None
    band: str | None = None
    coverage: float | None = None
    categories: list[CategoryDetail] = Field(default_factory=list)
    score_changes: list[ScoreChange] = Field(default_factory=list)
    # Valuation, risk, traps, confidence
    fair_value: FairValue | None = None
    risk: RiskDetail | None = None
    value_trap: ValueTrap | None = None
    confidence: Confidence | None = None
    # Evidence
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    price_history: dict = Field(default_factory=dict)
    financial_trends: list[FinancialTrend] = Field(default_factory=list)
    peers: list[PeerRow] = Field(default_factory=list)
    sources: list[SourceRecord] = Field(default_factory=list)
    freshness: DataFreshness = Field(default_factory=DataFreshness)
    spec_versions: dict = Field(default_factory=dict)


class HealthRow(BaseModel):
    name: str
    layer: str
    status: str
    rows: int | None = None
    entities: int | None = None
    updated_at: str | None = None
    age_hours: float | None = None
    max_age_hours: float | None = None
    span: str | None = None
    source: str | None = None
    detail: str | None = None
    coverage: float | None = None


class HealthWarning(BaseModel):
    severity: Literal["info", "warning", "critical"]
    message: str
    table: str | None = None


class HealthResponse(BaseModel):
    tables: list[HealthRow] = Field(default_factory=list)
    warnings: list[HealthWarning] = Field(default_factory=list)
    checked_at: datetime


class StatusResponse(BaseModel):
    as_of: date | None = None
    universe_size: int = 0
    scored_count: int = 0
    has_scores: bool = False
    scoring_version: str | None = None
    valuation_version: str | None = None
    portfolio_version: str | None = None
    freshness: DataFreshness = Field(default_factory=DataFreshness)
    pipeline_running: bool = False
    pipeline_stage: str | None = None
    pipeline_progress: float | None = None
    pipeline_error: str | None = None
    message: str | None = None
