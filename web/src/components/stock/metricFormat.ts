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

import type { MetricDetail, MetricStatusValue } from '../../api/types';
import { DASH, compactCurrency, dateShort, multiple, num, pct } from '../../lib/format';

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
export function isScored(status: MetricStatusValue | null | undefined): boolean {
  return status === 'scored';
}

/** A visible name for a metric status. Unknown values read as missing. */
export function statusLabel(status: MetricStatusValue | null | undefined): string {
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

/**
 * Axis label for a period end or a daily date: "Sep 23" when a series crosses a
 * calendar year (so two Decembers cannot look alike), "3 Sep" when it does not.
 * Pure presentation — the tooltip and the table twin keep the full date.
 */
export function axisDateFormatter(dates: string[]): (value: string) => string {
  const first = dates[0];
  const last = dates[dates.length - 1];
  const crossesYears =
    typeof first === 'string' && typeof last === 'string' && first.slice(0, 4) !== last.slice(0, 4);
  if (!crossesYears) return dateShort;
  return (value: string): string => {
    const parsed = new Date(/^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00` : value);
    if (Number.isNaN(parsed.getTime())) return DASH;
    return parsed.toLocaleDateString(undefined, { month: 'short', year: '2-digit' });
  };
}

/** "+1 higher is better" / "-1 lower is better", as the API encodes direction. */
export function directionText(direction: number | null | undefined): string {
  if (direction === 1) return 'Higher is better';
  if (direction === -1) return 'Lower is better';
  return 'Direction not specified';
}

/**
 * Confidence, normalised to a 0-1 fraction.
 *
 * schemas.py documents `Confidence.score` as 0-100, but the pipeline emits a
 * fraction: spaid/pipeline/confidence.py blends "the components into one 0-1
 * confidence" and clamps each component to 0-1 as well, and the live API agrees
 * (0.8156 for a company the API labels "High"). Rendering that as "0.8 out of
 * 100" would turn high confidence into almost none, so the scale is resolved
 * defensively: at most 1 reads as a fraction, above 1 as a 0-100 score. Both
 * readings agree at the bottom of the range, so neither scale can be flipped
 * into its opposite. Formatting only — the label beside the number is always
 * the API's own.
 */
export function confidenceFraction(value: number | null | undefined): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  const fraction = value > 1 ? value / 100 : value;
  return Math.min(1, Math.max(0, fraction));
}

/** Confidence as a percentage string, e.g. "82%". */
export function confidenceText(value: number | null | undefined, digits = 0): string {
  const fraction = confidenceFraction(value);
  return fraction === null ? DASH : pct(fraction, digits);
}

/** "Peer group (n = 42)" — the basis a percentile was measured against. */
export function peerGroupText(metric: MetricDetail): string {
  const raw = firstString(metric.peer_group) ?? firstString(metric.peer_basis);
  if (raw === null) return DASH;
  // The API uses "all" for the whole universe; spell that out.
  const group = raw === 'all' ? 'All companies' : raw;
  if (typeof metric.peer_count === 'number' && Number.isFinite(metric.peer_count)) {
    return `${group} (n = ${Math.round(metric.peer_count)})`;
  }
  return group;
}
