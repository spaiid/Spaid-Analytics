/**
 * The robustness battery, the overfitting probability and the trial registry.
 *
 * All three answer the same question from different directions: how much of
 * what you are looking at is search rather than signal? The registry is the
 * plainest form of it — every trial ever run, including the ones nobody liked,
 * because the count is an input to the deflated Sharpe ratio and curating it
 * would inflate whatever survived.
 */

import type {
  OverfittingSummary,
  RobustnessRow,
  RobustnessSummary,
  TrialRow,
} from '../../api/validation-types';
import { Card, DataTable, EmptyState, Pill, StatTile, type Column, type PillTone } from '../ui';
import { date, dateTime, humanize, int, pct, signedPct } from '../../lib/format';

const VERDICT_TONE: Record<string, PillTone> = {
  survived: 'good',
  weakened: 'warning',
  failed: 'critical',
  not_applicable: 'neutral',
  not_measurable: 'neutral',
};

export function RobustnessTable({
  rows,
  summary,
}: {
  rows: RobustnessRow[];
  summary: RobustnessSummary;
}) {
  if (rows.length === 0) {
    return (
      <Card title="Robustness tests">
        <p className="note">No robustness battery was run for this period.</p>
      </Card>
    );
  }

  const columns: Array<Column<RobustnessRow>> = [
    {
      key: 'label',
      header: 'Test',
      width: '26%',
      sortValue: (row) => row.label,
      sortLabel: 'test name',
      render: (row) => (
        <span style={{ display: 'inline-flex', flexDirection: 'column', gap: 2 }}>
          <span className="strong">{row.label}</span>
          <span className="t-micro muted" style={{ maxWidth: '44ch' }}>
            {row.question}
          </span>
        </span>
      ),
    },
    {
      key: 'variant',
      header: 'Setting',
      sortValue: (row) => row.variant,
      sortLabel: 'setting',
      render: (row) => <span className="mono t-small">{row.variant}</span>,
    },
    {
      key: 'cagr',
      header: 'Net CAGR',
      numeric: true,
      sortValue: (row) => row.cagr,
      sortLabel: 'net CAGR',
      render: (row) => <span className="tnum">{pct(row.cagr)}</span>,
    },
    {
      key: 'excess',
      header: 'Excess vs SPY',
      numeric: true,
      sortValue: (row) => row.excess_cagr_vs_spy,
      sortLabel: 'excess return',
      render: (row) => (
        <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'flex-end' }}>
          <span className="tnum strong">{signedPct(row.excess_cagr_vs_spy)}</span>
          <span className="t-micro muted nowrap">base {signedPct(row.base_excess_cagr_vs_spy)}</span>
        </span>
      ),
    },
    {
      key: 'verdict',
      header: 'Verdict',
      width: '26%',
      sortValue: (row) => row.verdict,
      sortLabel: 'verdict',
      render: (row) => (
        <span className="row-tight">
          <Pill tone={VERDICT_TONE[row.verdict] ?? 'neutral'}>{humanize(row.verdict)}</Pill>
          {row.verdict === 'not_applicable' && (
            <span className="t-micro muted">no excess return to preserve</span>
          )}
        </span>
      ),
    },
  ];

  const note =
    typeof summary.note === 'string' && summary.note !== ''
      ? summary.note
      : 'Each test changes one assumption and re-runs the primary portfolio. A valid signal should not disappear because the rebalance moved a day or because one unusually successful company was removed.';

  return (
    <Card
      title="Robustness tests"
      subtitle={note}
      actions={
        <span className="row-tight">
          <Pill tone="good">{`${int(summary.survived)} survived`}</Pill>
          <Pill tone="warning">{`${int(summary.weakened)} weakened`}</Pill>
          <Pill tone="critical">{`${int(summary.failed)} failed`}</Pill>
          {typeof summary.not_applicable === 'number' && summary.not_applicable > 0 && (
            <Pill tone="neutral">{`${int(summary.not_applicable)} not applicable`}</Pill>
          )}
        </span>
      }
      flush
    >
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(row) => `${row.test}-${row.variant}`}
        ariaLabel="Robustness test results"
      />
      <p className="note" style={{ margin: '14px 18px 18px' }}>
        Every test in the battery ran, in the order declared before the results existed. None was skipped because the
        previous one was disappointing, and the pass threshold was fixed in advance so a test cannot be reclassified as
        informational after it fails.
      </p>
    </Card>
  );
}

export function OverfittingPanel({ overfitting }: { overfitting: OverfittingSummary | null }) {
  if (overfitting === null) {
    return null;
  }
  if (!overfitting.feasible) {
    return (
      <Card title="Probability of backtest overfitting">
        <div className="row">
          <Pill tone="neutral">Not computable</Pill>
          <span className="secondary t-body">{overfitting.reason}</span>
        </div>
      </Card>
    );
  }
  const tone: PillTone = (overfitting.pbo ?? 1) >= 0.5 ? 'critical' : (overfitting.pbo ?? 1) >= 0.3 ? 'warning' : 'good';
  return (
    <Card
      title="Probability of backtest overfitting"
      subtitle="Combinatorially symmetric cross-validation: pick the best trial on half the periods, then see where it ranks on the other half. If the winner lands below median about half the time, the selection carried no information."
      actions={<Pill tone={tone}>{pct(overfitting.pbo, 0)}</Pill>}
    >
      <div className="grid grid-3">
        <StatTile label="Probability" value={pct(overfitting.pbo, 0)} sub={overfitting.verdict} small />
        <StatTile label="Trials compared" value={int(overfitting.n_trials)} sub="stored runs over this period" small />
        <StatTile label="Partitions" value={int(overfitting.n_partitions)} sub="balanced splits evaluated" small />
      </div>
    </Card>
  );
}

export function TrialRegistryTable({ trials, note }: { trials: TrialRow[]; note: string }) {
  if (trials.length === 0) {
    return (
      <Card title="Trial registry">
        <EmptyState
          title="No trials registered"
          message="Nothing has been run yet. Every backtest, including the ones that fail, is recorded here permanently."
        >
          <code>spaid validate --period development</code>
        </EmptyState>
      </Card>
    );
  }

  const columns: Array<Column<TrialRow>> = [
    {
      key: 'created_at',
      header: 'Run',
      sortValue: (row) => row.created_at,
      sortLabel: 'time',
      render: (row) => (
        <span style={{ display: 'inline-flex', flexDirection: 'column', gap: 2 }}>
          <span className="t-small secondary nowrap" title={dateTime(row.created_at)}>
            {date(row.created_at)}
          </span>
          <span className="t-micro muted mono truncate" style={{ maxWidth: 200 }}>
            {row.trial_id}
          </span>
        </span>
      ),
    },
    {
      key: 'purpose',
      header: 'Purpose',
      width: '28%',
      sortValue: (row) => row.purpose,
      sortLabel: 'purpose',
      render: (row) => (
        <span className="t-small secondary" style={{ display: 'block', maxWidth: '52ch' }}>
          {row.purpose}
          {row.notes !== '' && <span className="t-micro muted"> · {humanize(row.notes)}</span>}
        </span>
      ),
    },
    {
      key: 'period',
      header: 'Period',
      sortValue: (row) => row.period_label,
      sortLabel: 'period',
      render: (row) => (
        <span style={{ display: 'inline-flex', flexDirection: 'column', gap: 2 }}>
          <span className="t-small">{humanize(row.period_label)}</span>
          <span className="t-micro muted nowrap">
            {row.period_start === null ? '' : `${date(row.period_start)} – ${date(row.period_end)}`}
          </span>
        </span>
      ),
    },
    {
      key: 'net_cagr',
      header: 'Net CAGR',
      numeric: true,
      sortValue: (row) => row.net_cagr,
      sortLabel: 'net CAGR',
      render: (row) => <span className="tnum">{pct(row.net_cagr)}</span>,
    },
    {
      key: 'excess',
      header: 'vs SPY',
      numeric: true,
      sortValue: (row) => row.excess_vs_spy,
      sortLabel: 'excess return',
      render: (row) => <span className="tnum">{signedPct(row.excess_vs_spy)}</span>,
    },
    {
      key: 'provenance',
      header: 'Provenance',
      width: '22%',
      render: (row) => (
        <span style={{ display: 'inline-flex', flexDirection: 'column', gap: 1 }}>
          <span className="t-micro muted mono truncate">{row.config_checksum.slice(0, 12)} · config</span>
          <span className="t-micro muted mono truncate">{row.code_commit ?? '—'} · code</span>
          <span className="t-micro muted mono truncate">{row.data_version ?? '—'} · data</span>
          <span className="t-micro muted mono truncate">{row.universe_version ?? '—'} · universe</span>
        </span>
      ),
    },
    {
      key: 'outcome',
      header: 'Outcome',
      sortValue: (row) => row.outcome,
      sortLabel: 'outcome',
      render: (row) => (
        <Pill tone={row.outcome === 'completed' ? 'info' : row.outcome === 'failed' ? 'critical' : 'neutral'}>
          {humanize(row.outcome)}
        </Pill>
      ),
    },
  ];

  return (
    <Card title="Trial registry" subtitle={note} flush>
      <DataTable
        columns={columns}
        rows={trials}
        rowKey={(row) => row.trial_id}
        ariaLabel="Every registered backtest trial"
        defaultSort={{ key: 'created_at', direction: 'desc' }}
        sticky
      />
    </Card>
  );
}
