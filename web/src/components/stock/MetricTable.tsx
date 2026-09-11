/**
 * MetricTable — every underlying metric behind a category score.
 *
 * Raw value, peer percentile, peer group and count, weight, contribution and
 * status, so the arithmetic can be checked line by line. Absence is never drawn
 * as a low value: a metric with no data shows the em dash and says why.
 */

import type { MetricDetail } from '../../api/types';
import { DataTable, MissingValue, Pill, Tooltip, type Column } from '../ui';
import { DASH, num, pct } from '../../lib/format';
import { isScored, metricMedianText, metricValueText, peerGroupText, statusLabel } from './metricFormat';

function columns(): Array<Column<MetricDetail>> {
  return [
    {
      key: 'label',
      header: 'Metric',
      render: (metric) => (
        <span>
          <span className="strong">
            {metric.description ? (
              <Tooltip content={metric.description}>{metric.label}</Tooltip>
            ) : (
              metric.label
            )}
          </span>
        </span>
      ),
      sortValue: (metric) => metric.label,
      sortLabel: 'metric name',
    },
    {
      key: 'value',
      header: 'Value',
      numeric: true,
      render: (metric) =>
        metric.raw_value === null && metric.display_value === null ? (
          <MissingValue status={metric.status} detail={metric.status_detail} label={metric.label} />
        ) : (
          <span className="tnum">{metricValueText(metric)}</span>
        ),
      sortValue: (metric) => metric.raw_value,
      sortLabel: 'raw value',
    },
    {
      key: 'median',
      header: 'Peer median',
      numeric: true,
      render: (metric) =>
        metric.peer_median === null && metric.peer_median_display === null ? (
          <span className="muted">{DASH}</span>
        ) : (
          <span className="tnum">{metricMedianText(metric)}</span>
        ),
      sortValue: (metric) => metric.peer_median,
      sortLabel: 'peer median',
    },
    {
      key: 'percentile',
      header: 'Percentile',
      numeric: true,
      title: 'Where this value sits inside its peer group, 0 to 100.',
      render: (metric) =>
        metric.percentile === null ? (
          <MissingValue status={metric.status} detail={metric.status_detail} label={`${metric.label} percentile`} />
        ) : (
          <span className="tnum">{pct(metric.percentile, 0)}</span>
        ),
      sortValue: (metric) => metric.percentile,
      sortLabel: 'peer percentile',
    },
    {
      key: 'peers',
      header: 'Peer group',
      render: (metric) => <span className="muted">{peerGroupText(metric)}</span>,
      sortValue: (metric) => metric.peer_count,
      sortLabel: 'peer count',
    },
    {
      key: 'weight',
      header: 'Weight',
      numeric: true,
      title: 'This metric’s share of its category, before any renormalisation for missing data.',
      render: (metric) => <span className="tnum">{pct(metric.weight, 0)}</span>,
      sortValue: (metric) => metric.weight,
      sortLabel: 'weight in category',
    },
    {
      key: 'contribution',
      header: 'Contribution',
      numeric: true,
      title: 'Points this metric added to its category score.',
      render: (metric) =>
        metric.contribution === null ? (
          <MissingValue
            status={metric.status}
            detail={
              metric.status_detail ??
              'This metric added nothing to the category score because it was not scored.'
            }
            label={`${metric.label} contribution`}
          />
        ) : (
          <span className="tnum">{num(metric.contribution, 1)}</span>
        ),
      sortValue: (metric) => metric.contribution,
      sortLabel: 'contribution',
    },
    {
      key: 'status',
      header: 'Status',
      render: (metric) => (
        <Pill tone="neutral" icon={!isScored(metric.status)} title={metric.status_detail ?? undefined}>
          {statusLabel(metric.status)}
        </Pill>
      ),
      sortValue: (metric) => statusLabel(metric.status),
      sortLabel: 'status',
    },
  ];
}

export interface MetricTableProps {
  metrics: MetricDetail[];
  /** Names the table for assistive tech, e.g. "Quality metrics for AAPL". */
  ariaLabel: string;
  className?: string;
}

export function MetricTable({ metrics, ariaLabel, className }: MetricTableProps) {
  return (
    <DataTable<MetricDetail>
      columns={columns()}
      rows={metrics}
      rowKey={(metric) => metric.key}
      ariaLabel={ariaLabel}
      className={className}
      empty={<span className="muted">No metrics in this group.</span>}
    />
  );
}
