/**
 * types.ts — a 1:1 mirror of spaid/api/schemas.py.
 *
 * Rules carried over from the Python contract:
 *  - Optional[...] becomes `| null`. Fields with a default are always present
 *    in the JSON body, so they are non-optional here.
 *  - Fractions arrive as FRACTIONS (0.12 = 12%): upside, coverage, percentile,
 *    confidence, margins, growth, swing_pct, probability, weight.
 *  - Scores arrive on a 0-100 scale.
 *  - The client never derives a verdict: every qualitative reading
 *    (`assessment`, `band`, `*_label`) is decided server-side.
 *
 * Dates are ISO strings: `date` -> "YYYY-MM-DD", `datetime` -> ISO 8601.
 */

/** Python: `Level = Literal["good", "warning", "critical", "neutral"]` */
export type Level = 'good' | 'warning' | 'critical' | 'neutral';

/** ISO "YYYY-MM-DD". */
export type IsoDate = string;
/** ISO 8601 date-time. */
export type IsoDateTime = string;

/** A qualitative reading of a number, decided server-side. */
export interface Assessment {
  level: Level;
  text: string;
}

export interface DataFreshness {
  as_of: IsoDate | null;
  age_hours: number | null;
  is_stale: boolean;
  detail: string | null;
}

/**
 * Status of a single scored metric. The API types this as a plain `str`;
 * these four are the documented values. Unknown values must be handled
 * gracefully (treat as "missing").
 */
export type MetricStatus = 'scored' | 'missing' | 'not_applicable' | 'thin_peers';

/** One scored metric, with everything needed to explain it. */
export interface MetricDetail {
  key: string;
  label: string;
  category: string;
  description: string;
  raw_value: number | null;
  display_value: string | null;
  unit: string;
  /** 0-100 within the peer group. */
  score: number | null;
  /** Fraction, 0-1. */
  percentile: number | null;
  peer_basis: string | null;
  peer_group: string | null;
  peer_count: number | null;
  peer_median: number | null;
  peer_median_display: string | null;
  /** Fraction of the category weight. */
  weight: number;
  /** Points this metric contributed to its category score. */
  contribution: number | null;
  /** One of MetricStatus; the API types it as an open string. */
  status: MetricStatus | string;
  status_detail: string | null;
  /** +1 when higher is better, -1 when lower is better. */
  direction: number;
  inputs_as_of: IsoDate | null;
}

export interface CategoryDetail {
  key: string;
  label: string;
  /** 0-100. */
  score: number | null;
  /** Fraction of the composite. */
  weight: number;
  /** Points contributed to the composite. */
  contribution: number | null;
  /** Fraction, 0-1. */
  coverage: number;
  n_metrics: number;
  n_missing: number;
  assessment: Assessment | null;
  metrics: MetricDetail[];
}

export interface ScenarioValue {
  label: string;
  value_per_share: number;
  /** Fraction, 0-1. */
  probability: number;
  discount_rate: number | null;
  terminal_growth: number | null;
  summary: string | null;
}

export interface MethodValue {
  method: string;
  label: string;
  value_per_share: number | null;
  /** Fraction, 0-1. */
  weight: number;
  used: boolean;
  /** Why this method was down-weighted or skipped. */
  reason: string | null;
}

export interface Sensitivity {
  assumption: string;
  description: string;
  low_value: number;
  high_value: number;
  base_value: number;
  /** Fraction, not a percentage. */
  swing_pct: number | null;
}

export interface FairValue {
  price: number;
  bear: number | null;
  base: number | null;
  bull: number | null;
  range_low: number | null;
  range_high: number | null;
  midpoint: number | null;
  /** Fraction to the midpoint, not a percentage. */
  upside: number | null;
  classification: string;
  classification_label: string;
  /** Fraction, 0-1. */
  confidence: number | null;
  confidence_label: string | null;
  business_model: string;
  methods: MethodValue[];
  scenarios: ScenarioValue[];
  sensitivities: Sensitivity[];
  assumptions: Record<string, unknown>;
  /** Reasons this estimate may be wrong. */
  caveats: string[];
  method_dispersion: number | null;
  spec_version: string;
}

export interface RiskDetail {
  volatility: number | null;
  beta: number | null;
  downside_beta: number | null;
  downside_deviation: number | null;
  max_drawdown_1y: number | null;
  max_drawdown_5y: number | null;
  dollar_volume_median: number | null;
  spread_bps: number | null;
  days_to_earnings: number | null;
  distress_score: number | null;
  distress_label: string | null;
  piotroski: number | null;
  risk_score: number | null;
  risk_label: string | null;
  assessment: Assessment | null;
}

export interface TrapSignal {
  key: string;
  label: string;
  fired: boolean;
  value: number | null;
  detail: string | null;
}

export interface ValueTrap {
  trap_score: number | null;
  classification: string;
  classification_label: string;
  signals: TrapSignal[];
  n_fired: number;
}

export interface ConfidenceComponent {
  key: string;
  label: string;
  /** 0-100. */
  score: number;
  /** Fraction, 0-1. */
  weight: number;
  detail: string | null;
}

export interface Confidence {
  /** 0-100. */
  score: number;
  label: string;
  components: ConfidenceComponent[];
  caveats: string[];
}

export interface ScoreChange {
  window: string;
  /** Points, signed. */
  delta: number | null;
  detail: string | null;
}

/** One row of the ranked list. */
export interface OpportunityRow {
  rank: number | null;
  company_id: string;
  ticker: string;
  name: string;
  sector: string | null;
  industry: string | null;
  business_model: string | null;
  price: number | null;
  /** 0-100. */
  score: number | null;
  band: string | null;
  quality: number | null;
  growth: number | null;
  momentum: number | null;
  valuation: number | null;
  /** Fraction, 0-1. */
  coverage: number | null;
  /** Fraction, 0-1. */
  confidence: number | null;
  confidence_label: string | null;
  valuation_class: string | null;
  valuation_label: string | null;
  /** Fraction to the fair-value midpoint. */
  upside: number | null;
  fair_value_low: number | null;
  fair_value_high: number | null;
  risk_score: number | null;
  trap_class: string | null;
  score_change_1w: number | null;
  score_change_1m: number | null;
  market_cap: number | null;
  top_reason: string | null;
  days_to_earnings: number | null;
}

export interface OpportunitiesResponse {
  as_of: IsoDate | null;
  universe_size: number;
  scored_count: number;
  rows: OpportunityRow[];
  sectors: string[];
  bands: string[];
  spec_version: string | null;
  freshness: DataFreshness;
  notes: string[];
}

export interface FinancialTrendPoint {
  period_end: IsoDate;
  available_at: IsoDate;
  value: number;
}

export interface FinancialTrend {
  concept: string;
  label: string;
  /** e.g. "currency" | "ratio" | "percent". */
  unit: string;
  points: FinancialTrendPoint[];
}

export interface PeerRow {
  company_id: string;
  ticker: string;
  name: string;
  score: number | null;
  quality: number | null;
  growth: number | null;
  momentum: number | null;
  valuation: number | null;
  ev_ebit: number | null;
  fcf_yield: number | null;
  revenue_growth_1y: number | null;
  operating_margin: number | null;
  market_cap: number | null;
  is_self: boolean;
}

export interface SourceRecord {
  field: string;
  source: string;
  as_of: IsoDate | null;
  detail: string | null;
}

/**
 * `StockDetail.price_history` is an untyped dict server-side. It is emitted as
 * parallel arrays; every field is treated as possibly absent.
 */
export interface PriceHistory {
  dates?: IsoDate[];
  close?: number[];
  volume?: number[];
}

export interface StockDetail {
  company_id: string;
  ticker: string;
  name: string;
  sector: string | null;
  industry: string | null;
  business_model: string | null;
  as_of: IsoDate | null;
  price: number | null;
  /** Fraction. */
  price_change_1d: number | null;
  market_cap: number | null;
  // Score
  /** 0-100. */
  score: number | null;
  rank: number | null;
  universe_size: number | null;
  /** Fraction, 0-1. */
  percentile: number | null;
  band: string | null;
  /** Fraction, 0-1. */
  coverage: number | null;
  categories: CategoryDetail[];
  score_changes: ScoreChange[];
  // Valuation, risk, traps, confidence
  fair_value: FairValue | null;
  risk: RiskDetail | null;
  value_trap: ValueTrap | null;
  confidence: Confidence | null;
  // Evidence
  strengths: string[];
  weaknesses: string[];
  risks: string[];
  catalysts: string[];
  price_history: PriceHistory;
  financial_trends: FinancialTrend[];
  peers: PeerRow[];
  sources: SourceRecord[];
  freshness: DataFreshness;
  spec_versions: Record<string, string>;
}

export interface HealthRow {
  name: string;
  layer: string;
  status: string;
  rows: number | null;
  entities: number | null;
  updated_at: string | null;
  age_hours: number | null;
  max_age_hours: number | null;
  span: string | null;
  source: string | null;
  detail: string | null;
  /** Fraction, 0-1. */
  coverage: number | null;
}

export interface HealthWarning {
  severity: 'info' | 'warning' | 'critical';
  message: string;
  table: string | null;
}

export interface HealthResponse {
  tables: HealthRow[];
  warnings: HealthWarning[];
  checked_at: IsoDateTime;
}

export interface StatusResponse {
  as_of: IsoDate | null;
  universe_size: number;
  scored_count: number;
  has_scores: boolean;
  scoring_version: string | null;
  valuation_version: string | null;
  portfolio_version: string | null;
  freshness: DataFreshness;
  pipeline_running: boolean;
  pipeline_stage: string | null;
  /** Fraction, 0-1. */
  pipeline_progress: number | null;
  pipeline_error: string | null;
  message: string | null;
}

/* -------------------------------------------------------------------------
   Endpoint-only shapes (not Pydantic models in schemas.py).
   ------------------------------------------------------------------------- */

/** POST /api/pipeline/run */
export interface PipelineStartResponse {
  started: boolean;
}

/** GET /api/pipeline/status */
export interface PipelineStatus {
  running: boolean;
  stage: string | null;
  /** Fraction, 0-1. */
  progress: number | null;
  error: string | null;
}

/** Error bodies: `{detail: string}` with a non-2xx status. */
export interface ApiErrorBody {
  detail: string;
}

/** Query parameters for GET /api/opportunities. */
export interface OpportunitiesQuery {
  limit?: number;
  sector?: string;
  band?: string;
  min_score?: number;
  search?: string;
}
