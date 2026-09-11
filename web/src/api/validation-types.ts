/**
 * validation-types.ts — a 1:1 mirror of spaid/api/validation_schemas.py.
 *
 * Same rules as types.ts: fractions arrive as fractions, every qualitative
 * reading is decided server-side, and anything that can be absent is `| null`.
 *
 * The one thing worth stating twice, because the whole page depends on it: the
 * client never decides what a status means. `status_label`, `conclusion_label`,
 * `verdict_label`, `assessment` and `reading` all arrive as sentences. A view
 * that inferred "validated" from a good-looking number would be exactly the
 * failure this milestone exists to prevent.
 */

import type { IsoDate, IsoDateTime } from './types';

export interface Gate {
  key: string;
  label: string;
  passed: boolean;
  detail: string;
  requirement: string;
  value: number | null;
  threshold: number | null;
}

/** The nine statuses declared in spaid/config/validation.py. */
export type ValidationStatusKey =
  | 'not_testable'
  | 'exploratory_only'
  | 'in_sample'
  | 'validation_passed'
  | 'validation_failed'
  | 'holdout_passed'
  | 'holdout_failed'
  | 'paper_tracking'
  | 'live_tracking';

export type ConclusionKey =
  | 'evidence_of_predictive_ability'
  | 'no_meaningful_evidence'
  | 'evidence_is_mixed'
  | 'insufficient_data'
  | 'backtest_invalid';

export interface Verdict {
  status: ValidationStatusKey;
  status_label: string;
  conclusion: ConclusionKey;
  conclusion_label: string;
  headline: string;
  reasoning: string[];
  limitations: string[];
  failed_gates: string[];
  data_gates: Gate[];
  result_gates: Gate[];
}

export interface StrategySummary {
  version: string;
  name: string;
  scoring_version: string;
  config_checksum: string;
  code_commit: string | null;
  frozen_at: IsoDateTime | null;
  status: string;
  notes: string;
  n_metrics: number;
  n_metrics_point_in_time: number;
  unavailable_point_in_time: Record<string, string>;
  category_weights: Record<string, number>;
  category_coverage_point_in_time: Record<string, number>;
}

export interface UniverseStatus {
  built: boolean;
  universe_version: string | null;
  method: string | null;
  coverage_start: string | null;
  coverage_end: string | null;
  snapshots: number | null;
  securities: number | null;
  spells: number | null;
  exits_observed: number | null;
  removed_securities: number | null;
  removed_with_prices: number | null;
  removed_without_prices: number | null;
  delisting_returns_known: number | null;
  survivorship_status: string;
  survivorship_detail: string;
  reason: string | null;
}

export interface PeriodInfo {
  label: string;
  start: IsoDate;
  end: IsoDate;
  purpose: string;
  years: number;
  evaluated: boolean;
  sealed: boolean;
  accesses: number;
}

export interface BenchmarkComparison {
  benchmark: string;
  cagr: number | null;
  total_return: number | null;
  excess_cagr: number | null;
  alpha_annual: number | null;
  beta: number | null;
  tracking_error: number | null;
  information_ratio: number | null;
  upside_capture: number | null;
  downside_capture: number | null;
  volatility: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  rolling_12m_beat_share: number | null;
  rolling_36m_beat_share: number | null;
  rolling_60m_beat_share: number | null;
}

export interface Interval {
  low: number | null;
  high: number | null;
  level: number;
  method: string;
}

export interface PeriodReturn {
  period: string;
  return: number;
}

export interface VariantPerformance {
  label: string;
  description: string;
  is_primary: boolean;
  run_id: string;
  net_cagr: number | null;
  gross_cagr: number | null;
  total_return: number | null;
  volatility: number | null;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number | null;
  drawdown_duration_days: number | null;
  calmar: number | null;
  positive_months: number | null;
  turnover_annual_one_way: number | null;
  total_costs: number | null;
  n_trades: number | null;
  n_delistings: number | null;
  mean_holding_months: number | null;
  cagr_interval: Interval | null;
  sharpe_interval: Interval | null;
  deflated_sharpe_probability: number | null;
  deflated_sharpe_verdict: string | null;
  benchmarks: BenchmarkComparison[];
  best_month: PeriodReturn | null;
  worst_month: PeriodReturn | null;
  best_year: PeriodReturn | null;
  worst_year: PeriodReturn | null;
  yearly_returns: PeriodReturn[];
}

export interface EquityPoint {
  date: IsoDate;
  strategy: number;
  strategy_gross: number | null;
  drawdown: number | null;
  benchmarks: Record<string, number>;
}

export interface BucketRow {
  bucket: string;
  mean_excess: number | null;
  median_excess: number | null;
  hit_rate: number | null;
  mean_score: number | null;
  n: number | null;
}

export interface IcRow {
  horizon_months: number;
  ic_mean: number | null;
  ic_median: number | null;
  ic_std: number | null;
  ic_t: number | null;
  ic_ir: number | null;
  hit_rate: number | null;
  n_dates: number | null;
  assessment: string;
}

export interface IcPoint {
  date: IsoDate;
  ic: number;
  n: number | null;
}

export interface GroupRow {
  group: string;
  ic_mean: number | null;
  ic_t: number | null;
  hit_rate: number | null;
  n_obs: number | null;
}

export interface CategoryRow {
  category: string;
  horizon_months: number;
  ic_mean: number | null;
  ic_t: number | null;
  ic_ir: number | null;
  hit_rate: number | null;
}

export interface AblationRow {
  removed: string;
  ic_mean: number | null;
  ic_t: number | null;
  delta_ic: number | null;
  reading: string;
}

export interface SignalSummary {
  ic: IcRow[];
  ic_series: IcPoint[];
  ic_series_horizon_months: number;
  deciles: BucketRow[];
  quintiles: BucketRow[];
  bands: BucketRow[];
  decile_spread: number | null;
  decile_monotonicity: number | null;
  by_sector: GroupRow[];
  by_size: GroupRow[];
  by_year: GroupRow[];
  by_regime: GroupRow[];
  categories: CategoryRow[];
  ablations: AblationRow[];
  score_autocorrelation: number | null;
  mean_holding_months: number | null;
  share_single_month_holdings: number | null;
}

export interface RobustnessRow {
  test: string;
  label: string;
  question: string;
  variant: string;
  cagr: number | null;
  excess_cagr_vs_spy: number | null;
  base_excess_cagr_vs_spy: number | null;
  verdict: string;
  verdict_label: string;
}

export interface ContributionRow {
  name: string;
  profit: number;
  share: number | null;
}

export interface TrialRow {
  trial_id: string;
  created_at: IsoDateTime;
  purpose: string;
  strategy_version: string;
  config_checksum: string;
  code_commit: string | null;
  data_version: string | null;
  universe_version: string | null;
  period_label: string;
  period_start: IsoDate | null;
  period_end: IsoDate | null;
  outcome: string;
  net_cagr: number | null;
  excess_vs_spy: number | null;
  notes: string;
}

export interface OverfittingSummary {
  feasible: boolean;
  pbo: number | null;
  n_trials: number | null;
  n_partitions: number | null;
  verdict: string;
  reason: string;
}

export interface RobustnessSummary {
  n_tests?: number;
  survived?: number;
  weakened?: number;
  failed?: number;
  not_applicable?: number;
  not_measurable?: number;
  applicable?: number;
  survived_share?: number | null;
  note?: string;
}

export interface UniversePerDate {
  median_eligible?: number;
  min_eligible?: number;
  max_eligible?: number;
  median_scored?: number;
}

export interface CostSummary {
  half_spread_bps?: number;
  slippage_bps?: number;
  commission_bps?: number;
  min_cost_bps?: number;
  one_way_bps?: number;
}

export interface ExecutionSummary {
  delay_days?: number;
  rebalance?: string;
  cash_return_annual?: number;
}

export interface ValidationReport {
  available: boolean;
  period_label: string;
  message: string | null;
  generated_at: IsoDateTime | null;
  strategy: StrategySummary | null;
  universe: UniverseStatus | null;
  verdict: Verdict | null;
  period: PeriodInfo | null;
  periods: PeriodInfo[];
  decision_dates: number;
  universe_per_date: UniversePerDate;
  data_version: string | null;
  code_commit: string | null;
  universe_version: string | null;
  variants: VariantPerformance[];
  primary_variant: string | null;
  equity_curve: EquityPoint[];
  signal: SignalSummary | null;
  robustness: RobustnessRow[];
  robustness_summary: RobustnessSummary;
  overfitting: OverfittingSummary | null;
  contribution_by_stock: ContributionRow[];
  contribution_by_sector: ContributionRow[];
  costs: CostSummary;
  execution: ExecutionSummary;
}

export interface TrialRegistry {
  trials: TrialRow[];
  n_trials: number;
  note: string;
}
