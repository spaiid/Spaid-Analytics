/**
 * Portfolio growth, drawdown, rolling excess and the metric tables.
 *
 * Two presentation decisions that carry meaning rather than taste:
 *
 * Net is the headline everywhere. Gross is drawn as a dashed line beside it and
 * tabulated in its own column, never in place of net, because the difference
 * between the two is the whole question of whether the turnover is affordable.
 *
 * Every variant is shown, in the order the protocol declared, with the primary
 * one marked. Sorting the table by return would turn a diagnostic into a
 * leaderboard, which is exactly how a backtest ends up reporting whichever
 * variant happened to win.
 */

import type {
  BenchmarkComparison,
  ContributionRow,
  EquityPoint,
  VariantPerformance,
} from '../../api/validation-types';
import { BarChart, LineChart } from '../charts';
import { Card, DataTable, Pill, StatTile, Tooltip, type Column } from '../ui';
import { compactCurrency, int, num, pct, signedPct } from '../../lib/format';

const BENCHMARK_ORDER = ['SPY', 'SPMO', 'SPY (risk-matched)'];

function orderBenchmarks(rows: BenchmarkComparison[]): BenchmarkComparison[] {
  return [...rows].sort((a, b) => {
    const ai = BENCHMARK_ORDER.indexOf(a.benchmark);
    const bi = BENCHMARK_ORDER.indexOf(b.benchmark);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });
}

/* --------------------------------------------------------------- charts */

export function GrowthChart({ points, benchmarks }: { points: EquityPoint[]; benchmarks: string[] }) {
  if (points.length === 0) {
    return (
      <Card title="Portfolio growth">
        <p className="note">No daily path was stored for this run.</p>
      </Card>
    );
  }
  const x = points.map((p) => p.date);
  const series = [
    { name: 'Strategy (net)', values: points.map((p) => p.strategy) },
    { name: 'Strategy (gross)', values: points.map((p) => p.strategy_gross), dashed: true, emphasis: false },
    ...benchmarks.map((name) => ({
      name,
      values: points.map((p) => (name in p.benchmarks ? p.benchmarks[name] : null)),
    })),
  ];
  return (
    <Card
      title="Portfolio growth"
      subtitle="One unit invested at the start of the period, with dividends received as cash. Net is after spread, slippage and commission; gross is the same portfolio with costs switched off."
      flush
    >
      <div style={{ padding: '0 18px 12px' }}>
        <LineChart
          x={x}
          series={series}
          ariaLabel="Growth of one unit, strategy against benchmarks"
          height={300}
          yFormat={(value) => num(value, 2)}
          xHeader="Date"
        />
      </div>
    </Card>
  );
}

export function DrawdownChart({ points }: { points: EquityPoint[] }) {
  if (points.length === 0) return null;
  return (
    <Card
      title="Drawdown"
      subtitle="Distance below the running peak of the net equity path. Depth and duration are different experiences, and both are in the metrics table."
      flush
    >
      <div style={{ padding: '0 18px 12px' }}>
        <LineChart
          x={points.map((p) => p.date)}
          series={[{ name: 'Drawdown', values: points.map((p) => p.drawdown) }]}
          ariaLabel="Strategy drawdown over time"
          height={200}
          yFormat={(value) => pct(value, 0)}
          baselineAt={0}
          directLabelEnds={false}
        />
      </div>
    </Card>
  );
}

/** Rolling twelve-month excess return against one benchmark. */
export function RollingExcessChart({
  points,
  benchmark,
}: {
  points: EquityPoint[];
  benchmark: string;
}) {
  const window = 252;
  if (points.length <= window) {
    return (
      <Card title={`Rolling excess return vs ${benchmark}`}>
        <p className="note">
          The period is shorter than one rolling year at this sampling, so there is nothing to roll.
        </p>
      </Card>
    );
  }

  const x: string[] = [];
  const values: Array<number | null> = [];
  for (let i = window; i < points.length; i += 1) {
    const now = points[i];
    const then = points[i - window];
    const benchNow = now.benchmarks[benchmark];
    const benchThen = then.benchmarks[benchmark];
    x.push(now.date);
    if (benchNow === undefined || benchThen === undefined || benchThen === 0 || then.strategy === 0) {
      values.push(null);
      continue;
    }
    values.push(now.strategy / then.strategy - benchNow / benchThen);
  }

  return (
    <Card
      title={`Rolling excess return vs ${benchmark}`}
      subtitle="Strategy minus benchmark over every trailing twelve months. Overlapping windows, so these are a description of the experience rather than independent observations."
      flush
    >
      <div style={{ padding: '0 18px 12px' }}>
        <LineChart
          x={x}
          series={[{ name: `Excess vs ${benchmark}`, values }]}
          ariaLabel={`Rolling twelve-month excess return against ${benchmark}`}
          height={220}
          yFormat={(value) => signedPct(value, 0)}
          baselineAt={0}
          directLabelEnds={false}
        />
      </div>
    </Card>
  );
}

/* ---------------------------------------------------------------- tables */

export function VariantTable({ variants }: { variants: VariantPerformance[] }) {
  const columns: Array<Column<VariantPerformance>> = [
    {
      key: 'label',
      header: 'Portfolio',
      width: '22%',
      render: (row) => (
        <span style={{ display: 'inline-flex', flexDirection: 'column', gap: 2 }}>
          <span className="row-tight">
            <span className="strong">{row.label}</span>
            {row.is_primary && <Pill tone="info">Primary</Pill>}
          </span>
          <span className="t-micro muted">{row.description}</span>
        </span>
      ),
    },
    {
      key: 'net_cagr',
      header: 'Net CAGR',
      numeric: true,
      render: (row) => (
        <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'flex-end' }}>
          <span className="tnum strong">{pct(row.net_cagr)}</span>
          {row.cagr_interval !== null && (
            <span className="t-micro muted nowrap">
              {pct(row.cagr_interval.low)} to {pct(row.cagr_interval.high)}
            </span>
          )}
        </span>
      ),
    },
    {
      key: 'gross_cagr',
      header: 'Gross CAGR',
      numeric: true,
      render: (row) => <span className="tnum secondary">{pct(row.gross_cagr)}</span>,
    },
    {
      key: 'volatility',
      header: 'Volatility',
      numeric: true,
      render: (row) => <span className="tnum">{pct(row.volatility)}</span>,
    },
    {
      key: 'sharpe',
      header: 'Sharpe',
      numeric: true,
      render: (row) => <span className="tnum">{num(row.sharpe, 2)}</span>,
    },
    {
      key: 'sortino',
      header: 'Sortino',
      numeric: true,
      render: (row) => <span className="tnum">{num(row.sortino, 2)}</span>,
    },
    {
      key: 'max_drawdown',
      header: 'Max drawdown',
      numeric: true,
      render: (row) => (
        <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'flex-end' }}>
          <span className="tnum">{pct(row.max_drawdown)}</span>
          <span className="t-micro muted nowrap">{int(row.drawdown_duration_days)} days under water</span>
        </span>
      ),
    },
    {
      key: 'calmar',
      header: 'Calmar',
      numeric: true,
      render: (row) => <span className="tnum">{num(row.calmar, 2)}</span>,
    },
    {
      key: 'positive_months',
      header: 'Positive months',
      numeric: true,
      render: (row) => <span className="tnum">{pct(row.positive_months, 0)}</span>,
    },
    {
      key: 'turnover',
      header: 'Turnover',
      numeric: true,
      title: 'One-way, annualised, as a fraction of equity',
      render: (row) => <span className="tnum">{num(row.turnover_annual_one_way, 1)}×</span>,
    },
    {
      key: 'costs',
      header: 'Costs paid',
      numeric: true,
      render: (row) => <span className="tnum secondary">{compactCurrency(row.total_costs)}</span>,
    },
    {
      key: 'holding',
      header: 'Holding',
      numeric: true,
      title: 'Mean months a name stays in the portfolio once bought',
      render: (row) => <span className="tnum">{num(row.mean_holding_months, 1)}m</span>,
    },
  ];

  return (
    <Card
      title="Predefined portfolio variants"
      subtitle={
        <>
          All three were declared before the test and all three are reported. The point of testing several sizes is
          diagnosis, not picking a winner — so the primary portfolio is the top-20 basket regardless of which one did
          best.
        </>
      }
      flush
    >
      <DataTable
        columns={columns}
        rows={variants}
        rowKey={(row) => row.label}
        ariaLabel="Performance by portfolio variant"
      />
      <p className="note" style={{ margin: '14px 18px 18px' }}>
        Confidence intervals under the net return are 95% moving-block bootstrap intervals: blocks of consecutive days,
        because daily returns are not independent and resampling single days would make the interval far too narrow.
      </p>
    </Card>
  );
}

export function BenchmarkTable({ variant }: { variant: VariantPerformance }) {
  const rows = orderBenchmarks(variant.benchmarks);
  const columns: Array<Column<BenchmarkComparison>> = [
    {
      key: 'benchmark',
      header: 'Benchmark',
      render: (row) => (
        <span style={{ display: 'inline-flex', flexDirection: 'column', gap: 2 }}>
          <span className="strong">{row.benchmark}</span>
          {row.benchmark === 'SPY (risk-matched)' && (
            <span className="t-micro muted">SPY scaled to the strategy&rsquo;s own volatility</span>
          )}
        </span>
      ),
    },
    { key: 'cagr', header: 'Benchmark CAGR', numeric: true, render: (row) => <span className="tnum">{pct(row.cagr)}</span> },
    {
      key: 'excess_cagr',
      header: 'Excess CAGR',
      numeric: true,
      render: (row) => <span className="tnum strong">{signedPct(row.excess_cagr)}</span>,
    },
    {
      key: 'alpha',
      header: 'Alpha',
      numeric: true,
      title: 'Annualised intercept of the daily regression',
      render: (row) => <span className="tnum">{signedPct(row.alpha_annual)}</span>,
    },
    { key: 'beta', header: 'Beta', numeric: true, render: (row) => <span className="tnum">{num(row.beta, 2)}</span> },
    {
      key: 'te',
      header: 'Tracking error',
      numeric: true,
      render: (row) => <span className="tnum">{pct(row.tracking_error)}</span>,
    },
    {
      key: 'ir',
      header: 'Information ratio',
      numeric: true,
      render: (row) => <span className="tnum">{num(row.information_ratio, 2)}</span>,
    },
    {
      key: 'capture',
      header: 'Capture up / down',
      numeric: true,
      render: (row) => (
        <span className="tnum">
          {pct(row.upside_capture, 0)} / {pct(row.downside_capture, 0)}
        </span>
      ),
    },
    {
      key: 'rolling',
      header: 'Beat over 12 / 36 / 60m',
      numeric: true,
      title: 'Share of rolling windows in which the strategy beat this benchmark',
      render: (row) => (
        <span className="tnum">
          {pct(row.rolling_12m_beat_share, 0)} / {pct(row.rolling_36m_beat_share, 0)} /{' '}
          {pct(row.rolling_60m_beat_share, 0)}
        </span>
      ),
    },
  ];

  return (
    <Card
      title={`Against the benchmarks — ${variant.label}`}
      subtitle={
        <>
          Net of costs, total return including dividends.{' '}
          <Tooltip content="A strategy that beats the market by carrying more risk has bought more market, not beaten it. The risk-matched row scales SPY to the strategy's own volatility so that explanation is removed.">
            <span>Why the risk-matched row is here</span>
          </Tooltip>
        </>
      }
      flush
    >
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(row) => row.benchmark}
        ariaLabel={`Benchmark comparison for ${variant.label}`}
      />
    </Card>
  );
}

export function HeadlineTiles({ variant }: { variant: VariantPerformance }) {
  const spy = variant.benchmarks.find((b) => b.benchmark === 'SPY');
  const spmo = variant.benchmarks.find((b) => b.benchmark === 'SPMO');
  return (
    <div className="grid grid-5">
      <StatTile label="Net CAGR" value={pct(variant.net_cagr)} sub={`gross ${pct(variant.gross_cagr)}`} small />
      <StatTile label="vs SPY" value={signedPct(spy?.excess_cagr ?? null)} sub={`SPY ${pct(spy?.cagr ?? null)}`} small />
      <StatTile
        label="vs SPMO"
        value={signedPct(spmo?.excess_cagr ?? null)}
        sub={`SPMO ${pct(spmo?.cagr ?? null)}`}
        small
      />
      <StatTile label="Max drawdown" value={pct(variant.max_drawdown)} sub={`Sharpe ${num(variant.sharpe, 2)}`} small />
      <StatTile
        label="Deflated Sharpe"
        value={pct(variant.deflated_sharpe_probability, 0)}
        sub={variant.deflated_sharpe_verdict ?? 'probability the edge is real'}
        small
      />
    </div>
  );
}

export function YearlyReturns({ variant }: { variant: VariantPerformance }) {
  if (variant.yearly_returns.length === 0) return null;
  return (
    <Card
      title="Calendar-year returns"
      subtitle="Net of costs. The best and worst years are the ones a holder would actually have had to sit through."
      flush
    >
      <div style={{ padding: '0 18px 12px' }}>
        <BarChart
          labels={variant.yearly_returns.map((row) => row.period)}
          values={variant.yearly_returns.map((row) => row.return)}
          ariaLabel="Net return by calendar year"
          diverging
          valueFormat={(value) => signedPct(value, 1)}
          categoryHeader="Year"
          barName="Net return"
          height={240}
        />
      </div>
    </Card>
  );
}

export function ContributionPanel({
  byStock,
  bySector,
}: {
  byStock: ContributionRow[];
  bySector: ContributionRow[];
}) {
  const top = byStock.slice(0, 12);
  const bottom = byStock.slice(-8).reverse();
  return (
    <div className="grid grid-2">
      <Card
        title="Contribution by holding"
        subtitle="Currency profit over the period, largest and smallest. A result carried by two names is a different object from a broad one."
        flush
      >
        <div style={{ padding: '0 18px 12px' }}>
          <BarChart
            labels={[...top, ...bottom].map((row) => row.name)}
            values={[...top, ...bottom].map((row) => row.profit)}
            ariaLabel="Profit contribution by holding"
            horizontal
            diverging
            valueFormat={(value) => compactCurrency(value)}
            categoryHeader="Holding"
            barName="Profit"
            height={Math.max(220, (top.length + bottom.length) * 26 + 40)}
          />
        </div>
      </Card>
      <Card title="Contribution by sector" subtitle="The same profit, grouped." flush>
        <div style={{ padding: '0 18px 12px' }}>
          <BarChart
            labels={bySector.map((row) => row.name)}
            values={bySector.map((row) => row.profit)}
            ariaLabel="Profit contribution by sector"
            horizontal
            diverging
            valueFormat={(value) => compactCurrency(value)}
            categoryHeader="Sector"
            barName="Profit"
            height={Math.max(220, bySector.length * 26 + 40)}
          />
        </div>
      </Card>
    </div>
  );
}
