/**
 * Does the ranking carry information, separately from whether it made money?
 *
 * These panels are the ones that matter most and they are deliberately placed
 * above the robustness battery on the page: a portfolio result is one path and
 * can be produced by two lucky holdings, while an information coefficient is
 * measured across the whole cross-section on every decision date.
 *
 * The client renders what it is given and adds no interpretation. Each
 * information-coefficient row arrives with an `assessment` sentence, and each
 * ablation row with a `reading`, both written server-side.
 */

import type {
  AblationRow,
  BucketRow,
  CategoryRow,
  GroupRow,
  IcRow,
  SignalSummary,
} from '../../api/validation-types';
import { BarChart, LineChart } from '../charts';
import { Card, DataTable, Pill, StatTile, type Column } from '../ui';
import { humanize, int, num, pct, signedPct } from '../../lib/format';

function icTone(mean: number | null, t: number | null): 'good' | 'warning' | 'serious' | 'neutral' {
  if (mean === null) return 'neutral';
  if (mean <= 0) return 'serious';
  if (t !== null && Math.abs(t) >= 2) return 'good';
  return 'warning';
}

export function IcTable({ rows }: { rows: IcRow[] }) {
  const columns: Array<Column<IcRow>> = [
    {
      key: 'horizon',
      header: 'Horizon',
      numeric: true,
      render: (row) => <span className="tnum strong">{row.horizon_months}m</span>,
    },
    {
      key: 'ic_mean',
      header: 'Mean IC',
      numeric: true,
      title: 'Spearman correlation between score and forward excess return, averaged over decision dates',
      render: (row) => <span className="tnum strong">{num(row.ic_mean, 4)}</span>,
    },
    {
      key: 'ic_std',
      header: 'IC volatility',
      numeric: true,
      render: (row) => <span className="tnum">{num(row.ic_std, 4)}</span>,
    },
    {
      key: 'ic_ir',
      header: 'IC / volatility',
      numeric: true,
      title: 'Mean over its own standard deviation: a steady small IC differs from a wild one',
      render: (row) => <span className="tnum">{num(row.ic_ir, 2)}</span>,
    },
    {
      key: 'hit_rate',
      header: 'Dates positive',
      numeric: true,
      render: (row) => <span className="tnum">{pct(row.hit_rate, 0)}</span>,
    },
    {
      key: 'ic_t',
      header: 't-statistic',
      numeric: true,
      title: 'Computed on non-overlapping blocks, because overlapping forward windows are not independent',
      render: (row) => (
        <span className="row-tight" style={{ justifyContent: 'flex-end' }}>
          <span className="tnum">{num(row.ic_t, 2)}</span>
          <Pill tone={icTone(row.ic_mean, row.ic_t)}>
            {row.ic_mean === null ? '—' : row.ic_mean <= 0 ? 'Negative' : Math.abs(row.ic_t ?? 0) >= 2 ? 'Significant' : 'Not significant'}
          </Pill>
        </span>
      ),
    },
    {
      key: 'n_dates',
      header: 'Dates',
      numeric: true,
      render: (row) => <span className="tnum muted">{int(row.n_dates)}</span>,
    },
    {
      key: 'assessment',
      header: 'Reading',
      width: '34%',
      render: (row) => (
        <span className="t-small secondary" style={{ display: 'block', maxWidth: '54ch' }}>
          {row.assessment}
        </span>
      ),
    },
  ];
  return (
    <Card
      title="Information coefficient"
      subtitle="The headline test of the ranking: within each decision date, how well did the score order what happened next? Measured against excess return over the market, so it is about selection rather than direction."
      flush
    >
      <DataTable columns={columns} rows={rows} rowKey={(row) => String(row.horizon_months)} ariaLabel="Information coefficient by horizon" />
    </Card>
  );
}

export function IcOverTime({ signal }: { signal: SignalSummary }) {
  if (signal.ic_series.length === 0) return null;
  return (
    <Card
      title={`Information coefficient over time (${signal.ic_series_horizon_months}-month forward)`}
      subtitle="One point per monthly decision date. A signal that works should spend more time above the line than below it; one that alternates is describing the regime, not the companies."
      flush
    >
      <div style={{ padding: '0 18px 12px' }}>
        <LineChart
          x={signal.ic_series.map((point) => point.date)}
          series={[{ name: 'Rank IC', values: signal.ic_series.map((point) => point.ic) }]}
          ariaLabel="Information coefficient by decision date"
          height={220}
          yFormat={(value) => num(value, 2)}
          baselineAt={0}
          directLabelEnds={false}
        />
      </div>
    </Card>
  );
}

function bucketLabel(bucket: string, total: number): string {
  const index = Number(bucket);
  if (!Number.isFinite(index)) return bucket;
  if (index === 0) return `${index + 1} (lowest)`;
  if (index === total - 1) return `${index + 1} (highest)`;
  return String(index + 1);
}

export function DecileChart({ signal }: { signal: SignalSummary }) {
  const rows = signal.deciles;
  if (rows.length === 0) {
    return (
      <Card title="Forward returns by score decile">
        <p className="note">No forward returns could be measured over this period.</p>
      </Card>
    );
  }
  return (
    <Card
      title="Forward returns by score decile"
      subtitle="Mean three-month excess return, with deciles assigned within each date. Assigning them across the pooled panel would sort the calendar rather than the scores."
      actions={
        <span className="row-tight">
          <Pill tone={(signal.decile_spread ?? 0) > 0 ? 'good' : 'serious'}>
            {`Top − bottom ${signedPct(signal.decile_spread, 2)}`}
          </Pill>
          <Pill tone={(signal.decile_monotonicity ?? 0) >= 0.5 ? 'good' : 'warning'}>
            {`Monotonicity ${num(signal.decile_monotonicity, 2)}`}
          </Pill>
        </span>
      }
      flush
    >
      <div style={{ padding: '0 18px 12px' }}>
        <BarChart
          labels={rows.map((row) => bucketLabel(row.bucket, rows.length))}
          values={rows.map((row) => row.mean_excess)}
          ariaLabel="Mean forward excess return by score decile"
          diverging
          valueFormat={(value) => signedPct(value, 2)}
          categoryHeader="Decile"
          barName="Mean 3m excess return"
          height={260}
        />
      </div>
      <p className="note" style={{ margin: '0 18px 18px' }}>
        Monotonicity is the rank correlation between decile number and mean return. A signal that separates only its
        extremes and is flat in between scores low here, which is the intended reading: two extreme buckets can be
        produced by a handful of outliers, a staircase cannot.
      </p>
    </Card>
  );
}

export function BandTable({ rows }: { rows: BucketRow[] }) {
  if (rows.length === 0) return null;
  const columns: Array<Column<BucketRow>> = [
    { key: 'band', header: 'Band', render: (row) => <span className="strong">{row.bucket}</span> },
    { key: 'mean_score', header: 'Mean score', numeric: true, render: (row) => <span className="tnum">{num(row.mean_score, 1)}</span> },
    {
      key: 'mean_excess',
      header: 'Mean 3m excess',
      numeric: true,
      render: (row) => <span className="tnum strong">{signedPct(row.mean_excess, 2)}</span>,
    },
    {
      key: 'median_excess',
      header: 'Median 3m excess',
      numeric: true,
      render: (row) => <span className="tnum">{signedPct(row.median_excess, 2)}</span>,
    },
    { key: 'hit_rate', header: 'Positive', numeric: true, render: (row) => <span className="tnum">{pct(row.hit_rate, 0)}</span> },
    { key: 'n', header: 'Observations', numeric: true, render: (row) => <span className="tnum muted">{int(row.n)}</span> },
  ];
  return (
    <Card
      title="Forward returns by score band"
      subtitle="The bands the interface actually displays, so this answers the user's question directly: when the screen said 'Very attractive', what happened next?"
      flush
    >
      <DataTable columns={columns} rows={rows} rowKey={(row) => row.bucket} ariaLabel="Forward returns by score band" />
    </Card>
  );
}

function GroupTable({
  rows,
  title,
  subtitle,
  groupHeader,
}: {
  rows: GroupRow[];
  title: string;
  subtitle: string;
  groupHeader: string;
}) {
  if (rows.length === 0) return null;
  const columns: Array<Column<GroupRow>> = [
    {
      key: 'group',
      header: groupHeader,
      sortValue: (row) => row.group,
      sortLabel: groupHeader.toLowerCase(),
      render: (row) => <span className="strong">{humanize(row.group)}</span>,
    },
    {
      key: 'ic_mean',
      header: 'Mean IC',
      numeric: true,
      sortValue: (row) => row.ic_mean,
      sortLabel: 'mean information coefficient',
      render: (row) => (
        <span className="row-tight" style={{ justifyContent: 'flex-end' }}>
          <span className="tnum strong">{num(row.ic_mean, 4)}</span>
          <Pill tone={(row.ic_mean ?? 0) > 0 ? 'good' : 'serious'}>{(row.ic_mean ?? 0) > 0 ? 'Positive' : 'Negative'}</Pill>
        </span>
      ),
    },
    {
      key: 'ic_t',
      header: 't-statistic',
      numeric: true,
      sortValue: (row) => row.ic_t,
      sortLabel: 't-statistic',
      render: (row) => <span className="tnum">{num(row.ic_t, 2)}</span>,
    },
    {
      key: 'hit_rate',
      header: 'Dates positive',
      numeric: true,
      sortValue: (row) => row.hit_rate,
      sortLabel: 'share of dates positive',
      render: (row) => <span className="tnum">{pct(row.hit_rate, 0)}</span>,
    },
    {
      key: 'n_obs',
      header: 'Observations',
      numeric: true,
      sortValue: (row) => row.n_obs,
      sortLabel: 'observation count',
      render: (row) => <span className="tnum muted">{int(row.n_obs)}</span>,
    },
  ];
  return (
    <Card title={title} subtitle={subtitle} flush>
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(row) => row.group}
        ariaLabel={title}
        defaultSort={{ key: 'ic_mean', direction: 'desc' }}
      />
    </Card>
  );
}

export function SectorCut({ rows }: { rows: GroupRow[] }) {
  return (
    <GroupTable
      rows={rows}
      title="By sector"
      subtitle="Computed inside each sector, which is the harder test: a score that works only because it sorts technology above utilities is not selecting companies."
      groupHeader="Sector"
    />
  );
}

export function SizeCut({ rows }: { rows: GroupRow[] }) {
  return (
    <GroupTable
      rows={rows}
      title="By company size"
      subtitle="Market-capitalisation quintiles assigned within each date."
      groupHeader="Size quintile"
    />
  );
}

export function YearCut({ rows }: { rows: GroupRow[] }) {
  return (
    <GroupTable
      rows={rows}
      title="By calendar year"
      subtitle="A relationship that holds in one year and reverses in the next is a regime, not a signal."
      groupHeader="Year"
    />
  );
}

export function RegimeCut({ rows }: { rows: GroupRow[] }) {
  return (
    <GroupTable
      rows={rows}
      title="By market regime"
      subtitle="Trend, volatility and interest-rate environments, each classified only from data available on the decision date."
      groupHeader="Regime"
    />
  );
}

export function CategoryTable({ rows }: { rows: CategoryRow[] }) {
  if (rows.length === 0) return null;
  const columns: Array<Column<CategoryRow>> = [
    { key: 'category', header: 'Category', render: (row) => <span className="strong">{humanize(row.category)}</span> },
    { key: 'horizon', header: 'Horizon', numeric: true, render: (row) => <span className="tnum">{row.horizon_months}m</span> },
    {
      key: 'ic_mean',
      header: 'Mean IC',
      numeric: true,
      render: (row) => <span className="tnum strong">{num(row.ic_mean, 4)}</span>,
    },
    { key: 'ic_t', header: 't-statistic', numeric: true, render: (row) => <span className="tnum">{num(row.ic_t, 2)}</span> },
    { key: 'hit_rate', header: 'Dates positive', numeric: true, render: (row) => <span className="tnum">{pct(row.hit_rate, 0)}</span> },
  ];
  return (
    <Card
      title="Each category on its own"
      subtitle="Quality, growth, momentum and valuation as predictors in their own right. The composite can look informative while one category carries it, and a category with a negative coefficient is evidence against its weight rather than noise to average away."
      flush
    >
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(row) => `${row.category}-${row.horizon_months}`}
        ariaLabel="Information coefficient by scoring category"
      />
    </Card>
  );
}

export function AblationTable({ rows }: { rows: AblationRow[] }) {
  if (rows.length === 0) return null;
  const columns: Array<Column<AblationRow>> = [
    { key: 'removed', header: 'Removed', render: (row) => <span className="strong">{humanize(row.removed)}</span> },
    { key: 'ic_mean', header: 'Mean IC', numeric: true, render: (row) => <span className="tnum">{num(row.ic_mean, 4)}</span> },
    {
      key: 'delta',
      header: 'Change',
      numeric: true,
      render: (row) => (
        <span className="tnum strong">{row.delta_ic === null ? '—' : num(row.delta_ic, 4)}</span>
      ),
    },
    { key: 'ic_t', header: 't-statistic', numeric: true, render: (row) => <span className="tnum">{num(row.ic_t, 2)}</span> },
    {
      key: 'reading',
      header: 'Reading',
      width: '46%',
      render: (row) => (
        <span className="t-small secondary" style={{ display: 'block', maxWidth: '60ch' }}>
          {row.reading}
        </span>
      ),
    },
  ];
  return (
    <Card
      title="Ablations"
      subtitle="The composite rebuilt without each category, using the same renormalisation the scoring spec applies to a missing metric. A category whose removal improves the signal is carrying weight it has not earned — a finding to report, not to act on while the baseline is still being established."
      flush
    >
      <DataTable columns={columns} rows={rows} rowKey={(row) => row.removed} ariaLabel="Ablation results by category" />
    </Card>
  );
}

export function PersistenceTiles({ signal }: { signal: SignalSummary }) {
  return (
    <div className="grid grid-3">
      <StatTile
        label="Score persistence"
        value={num(signal.score_autocorrelation, 2)}
        sub="rank correlation month to month"
        small
      />
      <StatTile
        label="Mean holding period"
        value={`${num(signal.mean_holding_months, 1)} months`}
        sub="once a name is bought"
        small
      />
      <StatTile
        label="Held one month only"
        value={pct(signal.share_single_month_holdings, 0)}
        sub="share of position spells"
        small
      />
    </div>
  );
}
