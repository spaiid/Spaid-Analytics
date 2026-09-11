/**
 * SourcesTable — where every number came from and how old it is.
 */

import type { DataFreshness, SourceRecord } from '../../api/types';
import { Card, DataTable, EmptyState, type Column } from '../ui';
import { DASH, ageHours, date } from '../../lib/format';

function columns(): Array<Column<SourceRecord>> {
  return [
    {
      key: 'field',
      header: 'Field',
      render: (record) => <span className="strong">{record.field}</span>,
      sortValue: (record) => record.field,
      sortLabel: 'field',
    },
    {
      key: 'source',
      header: 'Source',
      render: (record) => record.source,
      sortValue: (record) => record.source,
      sortLabel: 'source',
    },
    {
      key: 'as_of',
      header: 'As of',
      numeric: true,
      render: (record) => <span className="tnum">{date(record.as_of)}</span>,
      sortValue: (record) => record.as_of,
      sortLabel: 'as-of date',
    },
    {
      key: 'detail',
      header: 'Detail',
      render: (record) => <span className="muted">{record.detail ?? DASH}</span>,
    },
  ];
}

export interface SourcesTableProps {
  sources: SourceRecord[];
  freshness: DataFreshness;
  ticker: string;
}

export function SourcesTable({ sources, freshness, ticker }: SourcesTableProps) {
  const age =
    typeof freshness.age_hours === 'number' && Number.isFinite(freshness.age_hours)
      ? `${ageHours(freshness.age_hours)} old`
      : 'age unknown';

  return (
    <Card
      title="Sources and freshness"
      headingLevel={3}
      subtitle={`Store as of ${date(freshness.as_of)} · ${age}${freshness.detail ? ` · ${freshness.detail}` : ''}`}
    >
      {/*
        No StaleBanner here: StockDetailView already shows one for this same
        `freshness` at the top of the page, and repeating it inside a card at the
        bottom turns a warning into wallpaper. The subtitle above carries the age.
      */}
      {sources.length === 0 ? (
        <EmptyState
          title="No source records"
          message="Nothing was recorded about where this company's inputs came from."
        />
      ) : (
        <DataTable<SourceRecord>
          columns={columns()}
          rows={sources}
          rowKey={(record, index) => `${record.field}-${index}`}
          ariaLabel={`Data sources for ${ticker}`}
        />
      )}
    </Card>
  );
}
