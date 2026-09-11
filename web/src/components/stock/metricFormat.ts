/**
 * metricFormat.ts — presentation helpers for the stock detail view.
 *
 * Formatting only. Nothing here decides what a number means: every qualitative
 * reading (assessment, band, classification_label, confidence_label,
 * status_detail) arrives from the API and is rendered verbatim.
 *
 * The server already formats most values into `display_value` /
 * `peer_median_display`; the unit-aware fallbacks below exist only for the case
 * where it sent a raw number and no string.
 */

import type { MetricDetail, MetricStatus } from '../../api/types';
import { DASH, compactCurrency, multiple, num, pct } from '../../lib/format';

/** The unit vocabulary used by the API (`UNIT_BY_METRIC` server-side). */
export type MetricUnit = 'percent' | 'times' | 'currency' | 'score' | 'ratio' | string;

/** Format a raw number the way its unit asks for. Never decides anything. */
export function formatByUnit(value: number | null | undefined, unit: MetricUnit): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return DASH;
  switch (unit) {
    case 'percent':
      return pct(value, 1);
    case 'times':
      return multiple(value, 1);
    case 'currency':
      return compactCurrency(value);
    case 'score':
      return num(value, 2);
    default:
      return num(value, 2);
  }
}

function firstString(value: string | null | undefined): string | null {
  return typeof value === 'string' && value.trim() !== '' ? value : null;
}

/** The server's own rendering wins; the unit fallback covers raw-only payloads. */
export function metricValueText(metric: MetricDetail): string {
  return firstString(metric.display_value) ?? formatByUnit(metric.raw_value, metric.unit);
}

export function metricMedianText(metric: MetricDetail): string {
  return firstString(metric.peer_median_display) ?? formatByUnit(metric.peer_median, metric.unit);
}

/** True only for the API's "scored" status — anything else is an absence. */
export function isScored(status: MetricStatus | string | null | undefined): boolean {
  return status === 'scored';
}

/** A visible name for a metric status. Unknown values read as missing. */
export function statusLabel(status: MetricStatus | string | null | undefined): string {
  switch (status) {
    case 'scored':
      return 'Scored';
    case 'not_applicable':
      return 'Not applicable';
    case 'thin_peers':
      return 'Thin peer group';
    case 'missing':
    default:
      return 'Missing';
  }
}

/** "+1 higher is better" / "-1 lower is better", as the API encodes direction. */
export function directionText(direction: number | null | undefined): string {
  if (direction === 1) return 'Higher is better';
  if (direction === -1) return 'Lower is better';
  return 'Direction not specified';
}

/** "Peer group (n = 42)" — the basis a percentile was measured against. */
export function peerGroupText(metric: MetricDetail): string {
  const group = firstString(metric.peer_group) ?? firstString(metric.peer_basis);
  if (group === null) return DASH;
  if (typeof metric.peer_count === 'number' && Number.isFinite(metric.peer_count)) {
    return `${group} (n = ${Math.round(metric.peer_count)})`;
  }
  return group;
}
