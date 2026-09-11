/**
 * HealthView — every dataset the store declares, and what the last quality
 * check said about it.
 *
 * The four statuses come from the API verbatim (`ok`, `stale`, `empty`,
 * `missing`); this view only chooses the icon and colour that travel with each
 * word — it never decides that a table is healthy. Loading, failure and a
 * genuinely empty register are three different renders.
 */

import { useEffect, useRef } from 'react';
import { apiPaths, useApi } from '../api/client';
import type { HealthResponse, HealthRow, HealthWarning } from '../api/types';
import {
  Card,
  DataTable,
  EmptyState,
  ErrorState,
  MissingValue,
  Pill,
  SkeletonTable,
  StatTile,
  Tooltip,
  type Column,
  type StatusTone,
} from '../components/ui';
import { ageHours, dateTime, humanize, int, pct } from '../lib/format';

/* ------------------------------------------------------------- status copy */

interface StatusCopy {
  tone: StatusTone;
  label: string;
  /** Severity order, worst last — used only to sort the column. */
  rank: number;
  explain: string;
}

const STATUS: Record<string, StatusCopy> = {
  ok: { tone: 'good', label: 'OK', rank: 0, explain: 'Present, non-empty and inside its freshness limit.' },
  stale: { tone: 'warning', label: 'Stale', rank: 1, explain: 'Older than the freshness limit set for this table.' },
  empty: { tone: 'serious', label: 'Empty', rank: 2, explain: 'The table exists but has no rows in it.' },
  missing: { tone: 'critical', label: 'Missing', rank: 3, explain: 'The table has never been written.' },
};

function statusCopy(status: string): StatusCopy {
  const known = STATUS[status];
  if (known) return known;
  return { tone: 'neutral', label: humanize(status), rank: -1, explain: 'Status reported by the API.' };
}

const WARNING_TONE: Record<HealthWarning['severity'], StatusTone> = {
  info: 'info',
  warning: 'warning',
  critical: 'critical',
};

/* ------------------------------------------------------------------ pieces */

function StatusCell({ row }: { row: HealthRow }) {
  const copy = statusCopy(row.status);
  return (
    <Pill tone={copy.tone} title={copy.explain}>
      {copy.label}
    </Pill>
  );
}

function CountCell({ value, label }: { value: number | null; label: string }) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return <span className="tnum">{int(value)}</span>;
  }
  return <MissingValue status="missing" label={label} detail="The table has not been written, so there is nothing to count." />;
}

function AgeCell({ row }: { row: HealthRow }) {
  if (typeof row.age_hours !== 'number' || !Number.isFinite(row.age_hours)) {
    return <MissingValue status="missing" label="Age" detail="No write timestamp was recorded for this table." />;
  }
  const limit =
    typeof row.max_age_hours === 'number' && Number.isFinite(row.max_age_hours)
      ? `limit ${ageHours(row.max_age_hours)}`
      : 'no limit set';
  return (
    <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'flex-end', lineHeight: 1.35 }}>
      <span className="tnum">{ageHours(row.age_hours)}</span>
      <span className="t-micro muted nowrap">{limit}</span>
    </span>
  );
}

function warningKey(warning: HealthWarning, index: number): string {
  return `${warning.severity}:${warning.table ?? '-'}:${index}`;
}

function WarningList({ warnings }: { warnings: HealthWarning[] }) {
  if (warnings.length === 0) {
    return (
      <div className="row">
        <Pill tone="good">No warnings</Pill>
        <span className="secondary t-body">The last quality check raised nothing on any dataset.</span>
      </div>
    );
  }
  return (
    <ul className="stack" style={{ gap: 10 }}>
      {warnings.map((warning, index) => (
        <li key={warningKey(warning, index)} className="row" style={{ alignItems: 'flex-start', flexWrap: 'nowrap' }}>
          <Pill tone={WARNING_TONE[warning.severity]}>{humanize(warning.severity)}</Pill>
          <span className="t-body secondary" style={{ minWidth: 0 }}>
            {warning.message}
            {warning.table !== null && warning.table !== '' && (
              <>
                {' '}
                <span className="mono muted t-micro">({warning.table})</span>
              </>
            )}
          </span>
        </li>
      ))}
    </ul>
  );
}

function countByStatus(rows: HealthRow[], status: string): number {
  return rows.filter((row) => row.status === status).length;
}

function HealthSummary({ data }: { data: HealthResponse }) {
  const rows = data.tables;
  return (
    <div className="grid grid-5">
      <StatTile label="Datasets" value={int(rows.length)} sub="declared by the store" small />
      <StatTile label="OK" value={int(countByStatus(rows, 'ok'))} sub="fresh and populated" small />
      <StatTile label="Stale" value={int(countByStatus(rows, 'stale'))} sub="past the freshness limit" small />
      <StatTile label="Empty" value={int(countByStatus(rows, 'empty'))} sub="written but no rows" small />
      <StatTile label="Missing" value={int(countByStatus(rows, 'missing'))} sub="never written" small />
    </div>
  );
}

/* -------------------------------------------------------------------- view */

export interface HealthViewProps {
  /** Starts a pipeline run from the empty state. */
  onRunPipeline?: () => void;
  /** True while a run is in flight, so the action cannot be double-fired. */
  pipelineBusy?: boolean;
  /** Changes when a pipeline run finishes; re-reads the health of the store. */
  refreshToken?: number | null;
}

export function HealthView({ onRunPipeline, pipelineBusy = false, refreshToken = null }: HealthViewProps) {
  const health = useApi<HealthResponse>(apiPaths.health());

  // Re-read after a pipeline run rewrites the store.
  const reloadRef = useRef(health.reload);
  reloadRef.current = health.reload;
  const seenTokenRef = useRef(refreshToken);
  useEffect(() => {
    if (refreshToken === null || refreshToken === seenTokenRef.current) return;
    seenTokenRef.current = refreshToken;
    reloadRef.current();
  }, [refreshToken]);

  if (health.state === 'loading') {
    return (
      <Card title="Data health" subtitle="Every dataset the pipeline writes, and how fresh it is.">
        <SkeletonTable rows={8} columns={6} label="Loading data health" />
      </Card>
    );
  }

  if (health.state === 'error') {
    return (
      <Card title="Data health">
        <ErrorState
          title="Could not read the data health"
          message={health.message}
          detail={apiPaths.health()}
          onRetry={health.reload}
        />
      </Card>
    );
  }

  const { tables, warnings, checked_at: checkedAt } = health.data;
  const showCoverage = tables.some((row) => typeof row.coverage === 'number' && Number.isFinite(row.coverage));

  const columns: Array<Column<HealthRow>> = [
    {
      key: 'name',
      header: 'Dataset',
      sortValue: (row) => row.name,
      sortLabel: 'dataset name',
      render: (row) => (
        <span style={{ display: 'inline-flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
          <span className="row-tight">
            <span className="strong mono">{row.name}</span>
            <span className="t-micro muted">{humanize(row.layer)}</span>
          </span>
          {row.detail !== null && row.detail !== '' && (
            <span className="t-micro muted truncate" style={{ maxWidth: 360 }} title={row.detail}>
              {row.detail}
            </span>
          )}
        </span>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      sortValue: (row) => statusCopy(row.status).rank,
      sortLabel: 'status',
      defaultDirection: 'desc',
      render: (row) => <StatusCell row={row} />,
    },
    {
      key: 'rows',
      header: 'Rows',
      numeric: true,
      sortValue: (row) => row.rows,
      sortLabel: 'row count',
      render: (row) => <CountCell value={row.rows} label="Row count" />,
    },
    {
      key: 'entities',
      header: 'Entities',
      numeric: true,
      sortValue: (row) => row.entities,
      sortLabel: 'entity count',
      title: 'Distinct companies or securities in the table',
      render: (row) => <CountCell value={row.entities} label="Entity count" />,
    },
    {
      key: 'span',
      header: 'Span',
      sortValue: (row) => row.span,
      sortLabel: 'date span',
      render: (row) =>
        row.span !== null && row.span !== '' ? (
          <span className="t-small secondary nowrap">{row.span}</span>
        ) : (
          <MissingValue status="missing" label="Date span" detail="No dated rows were reported for this table." />
        ),
    },
    {
      key: 'updated',
      header: 'Updated',
      sortValue: (row) => row.updated_at,
      sortLabel: 'last write time',
      render: (row) =>
        row.updated_at !== null && row.updated_at !== '' ? (
          <span className="t-small secondary nowrap">{dateTime(row.updated_at)}</span>
        ) : (
          <MissingValue status="missing" label="Last written" detail="This table has never been written." />
        ),
    },
    {
      key: 'age',
      header: 'Age',
      numeric: true,
      sortValue: (row) => row.age_hours,
      sortLabel: 'age',
      title: 'Time since the last write, against the freshness limit for this table',
      render: (row) => <AgeCell row={row} />,
    },
    {
      key: 'source',
      header: 'Source',
      sortValue: (row) => row.source,
      sortLabel: 'source',
      render: (row) =>
        row.source !== null && row.source !== '' ? (
          <span className="mono t-micro muted">{row.source}</span>
        ) : (
          <MissingValue status="missing" label="Source" detail="No provider was recorded for this table." />
        ),
    },
  ];

  if (showCoverage) {
    columns.splice(4, 0, {
      key: 'coverage',
      header: 'Coverage',
      numeric: true,
      sortValue: (row) => row.coverage,
      sortLabel: 'coverage',
      title: 'Share of the universe present in this table',
      render: (row) =>
        typeof row.coverage === 'number' && Number.isFinite(row.coverage) ? (
          <span className="tnum">{pct(row.coverage)}</span>
        ) : (
          <MissingValue status="missing" label="Coverage" detail="Coverage was not reported for this table." />
        ),
    });
  }

  return (
    <div className="stack">
      <HealthSummary data={health.data} />

      <Card
        title="Datasets"
        subtitle={
          <>
            Checked {dateTime(checkedAt)}. Status, size, span and age come straight from the store; the age limit is
            the freshness threshold configured for that table.
          </>
        }
        flush
        busy={health.refreshing}
        actions={
          <button type="button" className="btn btn-sm" onClick={health.reload}>
            Refresh
          </button>
        }
      >
        {tables.length === 0 ? (
          <div style={{ padding: '0 18px 18px' }}>
            <EmptyState
              title="No datasets registered"
              message="The store has not declared any tables yet, so there is nothing to report on. This is an empty store, not a failed read."
              action={
                onRunPipeline
                  ? { label: pipelineBusy ? 'Pipeline running…' : 'Run pipeline', onClick: onRunPipeline, disabled: pipelineBusy }
                  : undefined
              }
            >
              <code>python -m spaid pipeline run</code>
            </EmptyState>
          </div>
        ) : (
          <>
            <DataTable
              columns={columns}
              rows={tables}
              rowKey={(row) => row.name}
              ariaLabel="Data health by dataset"
              defaultSort={{ key: 'status', direction: 'desc' }}
              sticky
            />
            <p className="note" style={{ margin: '14px 18px 18px' }}>
              <strong>OK</strong> present, populated and inside its limit. <strong>Stale</strong> older than the limit
              set for that table. <strong>Empty</strong> written but with no rows. <strong>Missing</strong> never
              written. A dash with a mark means the store reported nothing for that field — it is not a zero.
            </p>
          </>
        )}
      </Card>

      <Card
        title="Quality warnings"
        subtitle={
          <>
            Raised by the checks that run after every pipeline pass.{' '}
            <Tooltip content="Warnings are produced by the API, not by this page. They persist until the next run clears them.">
              <span>How these are produced</span>
            </Tooltip>
          </>
        }
        busy={health.refreshing}
      >
        <WarningList warnings={warnings} />
      </Card>

      <p className="t-micro muted">
        Health read at {dateTime(checkedAt)} · {int(tables.length)} datasets · {int(warnings.length)} warnings
      </p>
    </div>
  );
}
