import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import type { StockDetail } from '/Users/jspaid/Spaid-Analytics/web/src/api/types';
import {
  ConfidencePanel,
  EvidenceLists,
  FairValueCard,
  FinancialTrends,
  PeerTable,
  PriceHistoryCard,
  RiskPanel,
  ScoreBreakdown,
  SourcesTable,
  StockHeader,
  StockLoadError,
  ValueTrapPanel,
} from '/Users/jspaid/Spaid-Analytics/web/src/components/stock';
import { StockDetailView } from '/Users/jspaid/Spaid-Analytics/web/src/views/StockDetailView';

import { readFileSync, writeFileSync } from 'node:fs';
import { CategoryCard } from '/Users/jspaid/Spaid-Analytics/web/src/components/stock';

const full: StockDetail = {
  company_id: 'c-1',
  ticker: 'AAPL',
  name: 'Apple Inc.',
  sector: 'Information Technology',
  industry: 'Technology Hardware',
  business_model: 'operating',
  as_of: '2026-09-04',
  price: 231.44,
  price_change_1d: -0.0123,
  market_cap: 3.4e12,
  score: 68.52,
  rank: 12,
  universe_size: 480,
  percentile: 0.974,
  band: 'Very attractive',
  coverage: 0.86,
  categories: [
    {
      key: 'quality',
      label: 'Quality',
      score: 74.2,
      weight: 0.3,
      contribution: 22.26,
      coverage: 0.9,
      n_metrics: 11,
      n_missing: 1,
      assessment: { level: 'good', text: 'Ranks well ahead of its peers on this dimension.' },
      metrics: [
        {
          key: 'roic',
          label: 'Return on invested capital',
          category: 'quality',
          description: 'Operating profit after tax over invested capital.',
          raw_value: 0.412,
          display_value: '41.2%',
          unit: 'percent',
          score: 96.1,
          percentile: 0.961,
          peer_basis: 'sector',
          peer_group: 'Information Technology',
          peer_count: 64,
          peer_median: 0.121,
          peer_median_display: '12.1%',
          weight: 0.18,
          contribution: 17.3,
          status: 'scored',
          status_detail: null,
          direction: 1,
          inputs_as_of: '2026-06-30',
        },
        {
          key: 'gross_margin',
          label: 'Gross margin',
          category: 'quality',
          description: 'Gross profit over revenue.',
          raw_value: null,
          display_value: null,
          unit: 'percent',
          score: null,
          percentile: null,
          peer_basis: null,
          peer_group: null,
          peer_count: null,
          peer_median: null,
          peer_median_display: null,
          weight: 0.12,
          contribution: null,
          status: 'not_applicable',
          status_detail: 'This metric is not meaningful for this kind of business.',
          direction: 1,
          inputs_as_of: null,
        },
        {
          key: 'interest_coverage',
          label: 'Interest coverage',
          category: 'quality',
          description: 'EBIT over interest expense.',
          raw_value: null,
          display_value: null,
          unit: 'times',
          score: null,
          percentile: null,
          peer_basis: 'sector',
          peer_group: 'Information Technology',
          peer_count: 3,
          peer_median: null,
          peer_median_display: null,
          weight: 0.1,
          contribution: null,
          status: 'thin_peers',
          status_detail: 'Too few comparable companies to rank against.',
          direction: 1,
          inputs_as_of: null,
        },
        {
          key: 'fcf_margin',
          label: 'Free cash flow margin',
          category: 'quality',
          description: 'Free cash flow over revenue.',
          raw_value: null,
          display_value: null,
          unit: 'percent',
          score: null,
          percentile: null,
          peer_basis: null,
          peer_group: null,
          peer_count: null,
          peer_median: null,
          peer_median_display: null,
          weight: 0.08,
          contribution: null,
          status: 'missing',
          status_detail: 'No data available for this metric.',
          direction: 1,
          inputs_as_of: null,
        },
      ],
    },
    {
      key: 'growth',
      label: 'Growth',
      score: 51.4,
      weight: 0.25,
      contribution: 12.85,
      coverage: 0.62,
      n_metrics: 8,
      n_missing: 3,
      assessment: { level: 'neutral', text: 'Roughly in line with its peers. Based on 62% of the metrics in this category.' },
      metrics: [],
    },
    {
      key: 'momentum',
      label: 'Momentum',
      score: null,
      weight: 0.25,
      contribution: null,
      coverage: 0.1,
      n_metrics: 9,
      n_missing: 8,
      assessment: { level: 'neutral', text: 'Not enough data to score this category.' },
      metrics: [],
    },
    {
      key: 'valuation',
      label: 'Valuation',
      score: 62.0,
      weight: 0.2,
      contribution: 12.4,
      coverage: 1,
      n_metrics: 10,
      n_missing: 0,
      assessment: { level: 'good', text: 'Ranks above its peers on this dimension.' },
      metrics: [],
    },
  ],
  score_changes: [
    { window: '1 week', delta: 2.4, detail: null },
    { window: '1 month', delta: -1.1, detail: 'Momentum cooled.' },
    { window: '3 months', delta: null, detail: null },
  ],
  fair_value: {
    price: 231.44,
    bear: 180.2,
    base: 262.5,
    bull: 331.9,
    range_low: 172.0,
    range_high: 345.5,
    midpoint: 258.75,
    upside: 0.1179,
    classification: 'undervalued',
    classification_label: 'Undervalued',
    confidence: 0.62,
    confidence_label: 'Moderate',
    business_model: 'operating',
    methods: [
      { method: 'dcf', label: 'Discounted cash flow', value_per_share: 264.2, weight: 0.5, used: true, reason: null },
      { method: 'multiples', label: 'Peer multiples', value_per_share: 251.0, weight: 0.5, used: true, reason: null },
      { method: 'ddm', label: 'Dividend discount', value_per_share: null, weight: 0, used: false, reason: 'Pays no dividend.' },
    ],
    scenarios: [
      { label: 'Bear', value_per_share: 180.2, probability: 0.25, discount_rate: 0.095, terminal_growth: 0.015, summary: 'Margins compress.' },
      { label: 'Base', value_per_share: 262.5, probability: 0.5, discount_rate: 0.085, terminal_growth: 0.025, summary: null },
      { label: 'Bull', value_per_share: 331.9, probability: 0.25, discount_rate: 0.08, terminal_growth: null, summary: 'Services keep compounding.' },
    ],
    sensitivities: [
      { assumption: 'Discount rate', description: '±100 bps', low_value: 210.1, high_value: 318.4, base_value: 262.5, swing_pct: 0.412 },
      { assumption: 'Terminal growth', description: '±50 bps', low_value: 238.0, high_value: 291.2, base_value: 262.5, swing_pct: 0.202 },
      { assumption: 'Exit multiple', description: '±2 turns', low_value: 250.0, high_value: 275.0, base_value: 262.5, swing_pct: null },
    ],
    assumptions: {},
    caveats: ['Forward estimates come from a thin set of analysts.', 'The peer set includes two much larger companies.'],
    method_dispersion: 0.05,
    spec_version: 'v2',
  },
  risk: {
    volatility: 0.284,
    beta: 1.14,
    downside_beta: 1.31,
    downside_deviation: 0.191,
    max_drawdown_1y: -0.212,
    max_drawdown_5y: -0.394,
    dollar_volume_median: 8.4e9,
    spread_bps: 1.8,
    days_to_earnings: 23,
    distress_score: 4.12,
    distress_label: 'Safe zone',
    piotroski: 7,
    risk_score: 41.2,
    risk_label: 'Below average',
    assessment: { level: 'good', text: 'Carries less risk than the typical company in the index.' },
  },
  value_trap: {
    trap_score: 22.5,
    classification: 'attractive_value',
    classification_label: 'Attractively undervalued',
    signals: [
      { key: 'falling_estimates', label: 'Falling estimates', fired: true, value: -0.04, detail: 'Forward EPS revised down.' },
      { key: 'distress_risk', label: 'Financial distress risk', fired: false, value: 4.12, detail: null },
      { key: 'declining_margins', label: 'Declining margins', fired: false, value: null, detail: null },
    ],
    n_fired: 1,
  },
  confidence: {
    score: 64.8,
    label: 'Moderate',
    components: [
      { key: 'coverage', label: 'Metric coverage', score: 86, weight: 0.3, detail: '86% of metrics scored.' },
      { key: 'data_age', label: 'Data age', score: 72, weight: 0.25, detail: null },
      { key: 'valuation', label: 'Valuation agreement', score: 48, weight: 0.25, detail: 'Methods differ by 5%.' },
      { key: 'stability', label: 'Score stability', score: 55, weight: 0.2, detail: null },
    ],
    caveats: ['Three growth metrics are missing.', 'Only two valuation methods ran.'],
  },
  strengths: ['Return on invested capital ranks in the top 4% of its sector.'],
  weaknesses: ['Revenue growth trails the sector median.'],
  risks: ['The shares fell 21% peak to trough in the past year.'],
  catalysts: ['Earnings in 23 days.'],
  price_history: {
    dates: ['2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04'],
    close: [228.1, 230.4, 229.0, 231.44],
    volume: [1, 2, 3, 4],
  },
  financial_trends: [
    {
      concept: 'revenue',
      label: 'Revenue',
      unit: 'currency',
      points: [
        { period_end: '2025-12-31', available_at: '2026-02-01', value: 1.24e11 },
        { period_end: '2026-03-31', available_at: '2026-05-01', value: 9.5e10 },
        { period_end: '2026-06-30', available_at: '2026-08-01', value: 8.6e10 },
      ],
    },
    {
      concept: 'net_income',
      label: 'Net income',
      unit: 'currency',
      points: [
        { period_end: '2025-12-31', available_at: '2026-02-01', value: 3.4e10 },
        { period_end: '2026-03-31', available_at: '2026-05-01', value: 2.4e10 },
      ],
    },
  ],
  peers: [
    { company_id: 'c-1', ticker: 'AAPL', name: 'Apple Inc.', score: 68.5, quality: 74.2, growth: 51.4, momentum: null, valuation: 62, ev_ebit: 22.1, fcf_yield: 0.041, revenue_growth_1y: 0.062, operating_margin: 0.31, market_cap: 3.4e12, is_self: true },
    { company_id: 'c-2', ticker: 'MSFT', name: 'Microsoft Corp.', score: 71.2, quality: 78, growth: 63, momentum: 55, valuation: 51, ev_ebit: 26.4, fcf_yield: 0.033, revenue_growth_1y: 0.121, operating_margin: 0.44, market_cap: 3.1e12, is_self: false },
  ],
  sources: [
    { field: 'Revenue', source: 'SEC XBRL company facts', as_of: '2026-08-01', detail: 'Point-in-time.' },
    { field: 'Price', source: 'Yahoo Finance', as_of: '2026-09-04', detail: null },
  ],
  freshness: { as_of: '2026-09-04', age_hours: 140.5, is_stale: true, detail: 'The last pipeline run was six days ago.' },
  spec_versions: { scoring: 'v3', valuation: 'v2' },
};

const sparse: StockDetail = {
  ...full,
  ticker: 'ZZZZ',
  name: 'Empty Co.',
  sector: null,
  industry: null,
  business_model: null,
  as_of: null,
  price: null,
  price_change_1d: null,
  market_cap: null,
  score: null,
  rank: null,
  universe_size: null,
  percentile: null,
  band: null,
  coverage: null,
  categories: [],
  score_changes: [],
  fair_value: null,
  risk: null,
  value_trap: null,
  confidence: null,
  strengths: [],
  weaknesses: [],
  risks: [],
  catalysts: [],
  price_history: {},
  financial_trends: [],
  peers: [],
  sources: [],
  freshness: { as_of: null, age_hours: null, is_stale: false, detail: null },
};


const part = process.argv[4] ?? 'top';

const top = (
  <>
    <div className="card"><StockHeader detail={full} /></div>
    <FairValueCard ticker={full.ticker} fairValue={full.fair_value} />
    <ScoreBreakdown ticker={full.ticker} categories={full.categories} composite={full.score} coverage={full.coverage} />
  </>
);

const bottom = (
  <>
    <div className="card">
      <div className="sd-cats">
        <CategoryCard category={full.categories[0]} ticker={full.ticker} open onToggle={() => undefined} />
      </div>
    </div>
    <div className="sd-panels">
      <RiskPanel risk={full.risk} ticker={full.ticker} />
      <ValueTrapPanel valueTrap={full.value_trap} ticker={full.ticker} />
      <ConfidencePanel confidence={full.confidence} ticker={full.ticker} />
    </div>
    <EvidenceLists detail={full} />
    <PriceHistoryCard ticker={full.ticker} priceHistory={full.price_history} />
    <FinancialTrends ticker={full.ticker} trends={full.financial_trends} />
    <PeerTable peers={full.peers} ticker={full.ticker} />
    <SourcesTable sources={full.sources} freshness={full.freshness} ticker={full.ticker} />
  </>
);

const page = renderToStaticMarkup(
  <MemoryRouter>
    <main className="content stack">{part === 'top' ? top : bottom}</main>
  </MemoryRouter>,
);

const base = '/Users/jspaid/Spaid-Analytics/web/src/';
const css = ['styles/tokens.css', 'styles/base.css', 'styles/components.css', 'styles/charts.css', 'components/stock/stock.css']
  .map((file) => readFileSync(base + file, 'utf8'))
  .join('\n');

const out = process.argv[2] ?? 'preview.html';
const theme = process.argv[3] ?? 'dark';
writeFileSync(out, `<!doctype html><html lang="en" data-theme="${theme}"><head><meta charset="utf-8"><style>${css}</style></head><body>${page}</body></html>`);
console.log('wrote', out);
