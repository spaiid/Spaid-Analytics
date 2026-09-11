import { renderToStaticMarkup } from 'react-dom/server';
import type { OpportunitiesResponse, OpportunityRow } from '../src/api/types';
import {
  OpportunitiesTable,
  OpportunityCharts,
  OpportunityFilters,
  OpportunityHeader,
  applyFilters,
  bandOrderMap,
  binScores,
  countByBand,
  meanScoreBySector,
  EMPTY_FILTERS,
  UNBANDED,
} from '../src/components/opportunities';

const BANDS = ['Very attractive', 'Attractive', 'Neutral', 'Unattractive', 'Very unattractive'];

function row(over: Partial<OpportunityRow>): OpportunityRow {
  return {
    rank: 1, company_id: 'c1', ticker: 'AAA', name: 'Alpha Industries', sector: 'Energy',
    industry: 'Oil', business_model: 'operating', price: 12.5, score: 71.2, band: 'Very attractive',
    quality: 64.1, growth: 51.0, momentum: 80.4, valuation: 66.6, coverage: 0.92, confidence: 0.71,
    confidence_label: 'Moderate', valuation_class: 'undervalued', valuation_label: 'Undervalued',
    upside: 0.184, fair_value_low: 13, fair_value_high: 19, risk_score: 41.2, trap_class: 'Not a trap',
    score_change_1w: 2.4, score_change_1m: -1.1, market_cap: 4.2e9, top_reason: 'Free cash flow yield',
    days_to_earnings: 12, ...over,
  };
}

const rows: OpportunityRow[] = [
  row({}),
  row({ rank: 2, company_id: 'c2', ticker: 'BBB', name: 'Beta Bank', sector: 'Financials', band: 'Attractive', score: 64.0, quality: null, growth: 0, momentum: 49.5, valuation: null, upside: null, confidence: 0, confidence_label: 'Very low', valuation_class: 'insufficient_confidence', valuation_label: 'Insufficient confidence to classify', score_change_1w: 0, score_change_1m: null, days_to_earnings: 0, top_reason: null }),
  row({ rank: 3, company_id: 'c3', ticker: 'CCC', name: 'Gamma Grid', sector: 'Utilities', band: 'Neutral', score: 47.3, upside: -0.22, valuation_class: 'overvalued', valuation_label: 'Overvalued', risk_score: null, score_change_1w: -3.2 }),
  row({ rank: 4, company_id: 'c4', ticker: 'DDD', name: 'Delta Dynamics', sector: 'Energy', band: 'Unattractive', score: 36.5, valuation_class: 'significantly_overvalued', valuation_label: 'Significantly overvalued' }),
  row({ rank: 5, company_id: 'c5', ticker: 'EEE', name: 'Epsilon Ltd', sector: null, band: 'Very unattractive', score: 29.0, valuation_class: 'significantly_undervalued', valuation_label: 'Significantly undervalued' }),
  row({ rank: null, company_id: 'c6', ticker: 'FFF', name: 'Zeta Holdings', sector: 'Utilities', band: null, score: null, quality: null, growth: null, momentum: null, valuation: null, confidence: null, confidence_label: null, valuation_class: null, valuation_label: null, upside: null, risk_score: null, score_change_1w: null, score_change_1m: null, days_to_earnings: null, top_reason: null }),
];

const response: OpportunitiesResponse = {
  as_of: '2026-09-09', universe_size: 503, scored_count: 5, rows, sectors: ['Energy', 'Financials', 'Utilities'],
  bands: BANDS, spec_version: 'scoring-2.1',
  freshness: { as_of: '2026-09-09', age_hours: 30.5, is_stale: true, detail: 'Prices are a day behind.' },
  notes: ['1 company is listed without a score because too little evidence was available.'],
};

const problems: string[] = [];
function check(name: string, ok: boolean, extra = '') {
  if (!ok) problems.push(`FAIL ${name} ${extra}`);
  else console.log(`ok   ${name}`);
}

// ---- pure derivations
const f1 = applyFilters(rows, { ...EMPTY_FILTERS, minScore: 50 });
check('minScore keeps only >= 50', f1.rows.every((r) => (r.score ?? -1) >= 50) && f1.rows.length === 2, JSON.stringify(f1.rows.map((r) => r.ticker)));
check('minScore reports hidden unscored', f1.unscoredHidden === 1, String(f1.unscoredHidden));
check('search matches name, case-insensitive', applyFilters(rows, { ...EMPTY_FILTERS, search: 'beta b' }).rows.length === 1);
check('search matches ticker', applyFilters(rows, { ...EMPTY_FILTERS, search: 'ccc' }).rows.length === 1);
check('sector filter', applyFilters(rows, { ...EMPTY_FILTERS, sector: 'Energy' }).rows.length === 2);
check('band filter', applyFilters(rows, { ...EMPTY_FILTERS, band: 'Neutral' }).rows.length === 1);
check('unbanded filter', applyFilters(rows, { ...EMPTY_FILTERS, band: UNBANDED }).rows.length === 1);
check('no filters keeps everything', applyFilters(rows, EMPTY_FILTERS).rows.length === rows.length);

const counts = countByBand(rows, BANDS);
check('band counts cover every band + unbanded', counts.length === BANDS.length + 1);
check('band counts sum to rows', counts.reduce((s, c) => s + c.count, 0) === rows.length);

const bins = binScores([29, 36.5, 47.3, 64, 71.2], 5);
check('bins are 5 wide and cover the range', bins[0].from === 25 && bins[bins.length - 1].to === 75);
check('bin counts sum to input', bins.reduce((s, b) => s + b.count, 0) === 5);
check('bin edges land in the upper bin', binScores([50], 5)[0].from === 50);

const means = meanScoreBySector(rows);
check('sector means sorted high to low', means.sectors[0].mean >= means.sectors[means.sectors.length - 1].mean);
check('sector means exclude null-sector rows', means.withoutSector === 1);
check('overall mean over scored rows', means.overallMean !== null && Math.abs(means.overallMean - (71.2 + 64 + 47.3 + 36.5 + 29) / 5) < 1e-9);

// ---- markup
const html = renderToStaticMarkup(
  <>
    <OpportunityHeader data={response} shown={rows.length} bandCounts={counts} selectedBand="Neutral" onSelectBand={() => {}} />
    <OpportunityFilters sectors={response.sectors} bands={response.bands} showUnbanded value={EMPTY_FILTERS} onChange={() => {}} onReset={() => {}} shown={4} total={6} unscoredHidden={1} />
    <OpportunitiesTable rows={rows} bandOrder={bandOrderMap(BANDS)} onOpen={() => {}} />
    <OpportunityCharts rows={rows} onSelectSector={() => {}} />
  </>,
);

check('no NaN in markup', !html.includes('NaN'));
check('no undefined attribute', !html.includes('="undefined"'));
check('no literal undefined text', !html.includes('>undefined<'));
check('no null text', !html.includes('>null<'));
check('sortable headers expose aria-sort', (html.match(/aria-sort=/g) ?? []).length >= 12, String((html.match(/aria-sort=/g) ?? []).length));
check('rank column sorted ascending by default', html.includes('aria-sort="ascending"'));
check('rows are focusable', (html.match(/tabindex="0"/g) ?? []).length >= rows.length);
check('rows carry an accessible name', html.includes('aria-label="Open AAA, Alpha Industries"'));
check('meters use role=meter', html.includes('role="meter"'));
check('missing score renders the hatched empty meter or a dash', html.includes('missing-missing'));
check('not-a-zero: missing cells never print 0', !/<td[^>]*class="num"[^>]*>0<\/td>/.test(html));
check('zero sub-score still prints', html.includes('>0.0<'));
check('charts are images with labels', (html.match(/role="img"/g) ?? []).length === 2, String((html.match(/role="img"/g) ?? []).length));
check('histogram aria-label describes the data', /aria-label="Histogram of composite scores for the 5 scored companies shown, in 5-point bins from 25 to 75/.test(html), html.slice(html.indexOf('Histogram'), html.indexOf('Histogram') + 200));
check('sector chart aria-label describes the data', /aria-label="Diverging bar chart of mean composite score by sector/.test(html));
check('every chart ships a table twin toggle', (html.match(/>Table<\/button>/g) ?? []).length === 2);
check('band pills carry icon + text', html.includes('Very attractive') && html.includes('pill-good'));
check('valuation pill tones applied', html.includes('pill-warning') && html.includes('pill-serious'));
check('score deltas carry a direction class', html.includes('stat-delta up') && html.includes('stat-delta down') && html.includes('stat-delta flat'));
check('confidence shows value and label', html.includes('71.0%') && html.includes('Moderate'));
check('upside is signed', html.includes('+18.4%') && html.includes('−22.0%'));
check('filters have labels', (html.match(/<label class="t-label"/g) ?? []).length === 4);
check('result count is a live region', html.includes('role="status" aria-live="polite"'));
check('stale freshness surfaces', html.includes('Stale'));
check('band chips are toggle buttons', html.includes('aria-pressed="true"') && html.includes('class="band-chip"'));

if (problems.length > 0) {
  console.error('\n' + problems.join('\n'));
  process.exit(1);
}
console.log(`\nAll ${(html.match(/ok /g) ?? []).length || ''}checks passed. markup ${html.length} bytes`);
