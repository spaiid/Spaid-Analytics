import { writeFileSync } from 'node:fs';
import { renderToStaticMarkup } from 'react-dom/server';
import type { OpportunitiesResponse, OpportunityRow } from '../src/api/types';
import {
  OpportunitiesTable, OpportunityCharts, OpportunityFilters, OpportunityHeader,
  bandOrderMap, countByBand, EMPTY_FILTERS,
} from '../src/components/opportunities';

const BANDS = ['Very attractive', 'Attractive', 'Neutral', 'Unattractive', 'Very unattractive'];
const SECTORS = ['Information Technology', 'Financials', 'Health Care', 'Energy', 'Consumer Discretionary', 'Utilities', 'Communication Services'];
const NAMES = ['Northwind Traders', 'Contoso Energy', 'Fabrikam Health', 'Adventure Works', 'Litware Bancorp', 'Proseware Media', 'Tailspin Utilities', 'Wide World Importers', 'Lucerne Publishing', 'Graphic Design Institute', 'Coho Vineyard', 'Trey Research'];
const VAL: Array<[string, string]> = [
  ['significantly_undervalued', 'Significantly undervalued'], ['undervalued', 'Undervalued'],
  ['fairly_valued', 'Fairly valued'], ['overvalued', 'Overvalued'],
  ['significantly_overvalued', 'Significantly overvalued'], ['insufficient_confidence', 'Insufficient confidence to classify'],
];
const CONF = ['High', 'Moderate', 'Low', 'Very low'];

function mk(i: number): OpportunityRow {
  const score = 74 - i * 3.1;
  const band = score >= 68 ? BANDS[0] : score >= 60 ? BANDS[1] : score >= 42 ? BANDS[2] : score >= 33 ? BANDS[3] : BANDS[4];
  const v = VAL[i % VAL.length];
  const missing = i === 9;
  return {
    rank: missing ? null : i + 1,
    company_id: `c${i}`,
    ticker: ['NWTR', 'CNEN', 'FBHC', 'ADVW', 'LTWB', 'PRSM', 'TLSU', 'WWIM', 'LUCP', 'GDIN', 'COHV', 'TRYR'][i],
    name: NAMES[i],
    sector: i === 10 ? null : SECTORS[i % SECTORS.length],
    industry: null, business_model: 'operating',
    price: 18 + i * 7.3,
    score: missing ? null : score,
    band: missing ? null : band,
    quality: missing ? null : 40 + ((i * 13) % 55),
    growth: missing ? null : (i === 3 ? 0 : 30 + ((i * 21) % 65)),
    momentum: missing ? null : 25 + ((i * 31) % 70),
    valuation: missing || i === 4 ? null : 35 + ((i * 17) % 60),
    coverage: 0.9, confidence: missing ? null : 0.86 - i * 0.06,
    confidence_label: missing ? null : CONF[Math.min(3, Math.floor(i / 3))],
    valuation_class: missing ? null : v[0], valuation_label: missing ? null : v[1],
    upside: missing || i === 5 ? null : 0.42 - i * 0.09,
    fair_value_low: 10, fair_value_high: 30,
    risk_score: missing ? null : 20 + ((i * 23) % 70),
    trap_class: 'Not a trap',
    score_change_1w: missing ? null : (i % 3 === 0 ? 2.6 - i * 0.4 : i % 3 === 1 ? -1.4 - i * 0.2 : 0),
    score_change_1m: missing ? null : (i % 2 === 0 ? -4.1 + i * 0.5 : 5.2 - i * 0.3),
    market_cap: 1e9 * (i + 2),
    top_reason: i === 6 ? null : ['Free cash flow yield', 'Return on invested capital', 'Revenue growth, 3 year', 'Gross margin trend', 'Net debt to EBITDA'][i % 5],
    days_to_earnings: missing ? null : (i === 2 ? 0 : 3 + i * 6),
  };
}

const rows = Array.from({ length: 12 }, (_, i) => mk(i));
const response: OpportunitiesResponse = {
  as_of: '2026-09-09', universe_size: 503, scored_count: 11, rows, sectors: SECTORS, bands: BANDS,
  spec_version: 'scoring-2.1',
  freshness: { as_of: '2026-09-09', age_hours: 30.5, is_stale: true, detail: 'Prices are a day behind the last close.' },
  notes: ['1 company is listed without a score because too little of the evidence was available to compute one.'],
};

const counts = countByBand(rows, BANDS);
const body = renderToStaticMarkup(
  <div className="app">
    <nav className="sidebar" />
    <div className="main">
      <header className="topbar"><h1>Spaid Analytics</h1></header>
      <main className="content stack">
        <OpportunityHeader data={response} shown={rows.length} bandCounts={counts} selectedBand="Attractive" onSelectBand={() => {}} />
        <section className="card">
          <div className="card-head"><h2 className="card-title">Ranked list</h2></div>
          <p className="card-sub">Sorted by rank. Sort by any column header, open a row for the full explanation, and scroll the table sideways for risk, score changes and earnings.</p>
          <div className="card-body">
            <OpportunityFilters sectors={SECTORS} bands={BANDS} showUnbanded value={EMPTY_FILTERS} onChange={() => {}} onReset={() => {}} shown={12} total={12} unscoredHidden={0} />
            <OpportunitiesTable rows={rows} bandOrder={bandOrderMap(BANDS)} onOpen={() => {}} />
            <div className="opp-notes">{response.notes.map((n) => <p className="note" key={n}>{n}</p>)}</div>
          </div>
        </section>
        <OpportunityCharts rows={rows} onSelectSector={() => {}} />
      </main>
    </div>
  </div>,
);

const css = ['tokens', 'base', 'components', 'charts'].map((n) => `<link rel="stylesheet" href="../src/styles/${n}.css">`).join('\n');
const html = `<!doctype html><html lang="en" data-theme="THEME"><head><meta charset="utf-8">${css}
<link rel="stylesheet" href="../src/components/opportunities/opportunities.css"></head><body>${body}</body></html>`;
writeFileSync('.smoke/dark.html', html.replace('THEME', 'dark'));
writeFileSync('.smoke/light.html', html.replace('THEME', 'light'));
console.log('wrote .smoke/dark.html and .smoke/light.html');
