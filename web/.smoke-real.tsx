import { readFileSync, writeFileSync } from 'node:fs';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import type { StockDetail } from './src/api/types';
import {
  CategoryCard,
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
  ValueTrapPanel,
} from './src/components/stock';

const errors: string[] = [];
const original = console.error;
console.error = (...args: unknown[]) => {
  errors.push(args.map(String).join(' '));
  original(...args);
};

const file = process.argv[2];
const out = process.argv[3];
const theme = process.argv[4] ?? 'dark';
const part = process.argv[5] ?? 'top';
const d = JSON.parse(readFileSync(file, 'utf8')) as StockDetail;

const top = (
  <>
    <div className="card"><StockHeader detail={d} /></div>
    <FairValueCard ticker={d.ticker} fairValue={d.fair_value} />
    <ScoreBreakdown ticker={d.ticker} categories={d.categories} composite={d.score} coverage={d.coverage} />
  </>
);

const bottom = (
  <>
    <div className="card">
      <div className="sd-cats">
        <CategoryCard category={d.categories[1]} ticker={d.ticker} open onToggle={() => undefined} />
      </div>
    </div>
    <div className="sd-panels">
      <RiskPanel risk={d.risk} ticker={d.ticker} />
      <ConfidencePanel confidence={d.confidence} ticker={d.ticker} />
    </div>
    <ValueTrapPanel valueTrap={d.value_trap} ticker={d.ticker} />
    <EvidenceLists detail={d} />
    <PriceHistoryCard ticker={d.ticker} priceHistory={d.price_history} />
    <FinancialTrends ticker={d.ticker} trends={d.financial_trends} />
    <PeerTable peers={d.peers} ticker={d.ticker} />
    <SourcesTable sources={d.sources} freshness={d.freshness} ticker={d.ticker} />
  </>
);

const tail = (
  <>
    <FinancialTrends ticker={d.ticker} trends={d.financial_trends} />
    <PeerTable peers={d.peers} ticker={d.ticker} />
    <SourcesTable sources={d.sources} freshness={d.freshness} ticker={d.ticker} />
  </>
);

const page = renderToStaticMarkup(
  <MemoryRouter>
    <main className="content stack">{part === 'top' ? top : part === 'tail' ? tail : bottom}</main>
  </MemoryRouter>,
);

const base = './src/';
const css = ['styles/tokens.css', 'styles/base.css', 'styles/components.css', 'styles/charts.css', 'components/stock/stock.css']
  .map((f) => readFileSync(base + f, 'utf8'))
  .join('\n');

writeFileSync(out, `<!doctype html><html lang="en" data-theme="${theme}"><head><meta charset="utf-8"><style>${css}</style></head><body>${page}</body></html>`);
console.log('wrote', out, 'console errors:', errors.length, 'NaN:', page.includes('NaN'), 'undefined-attr:', page.includes('="undefined"'));
