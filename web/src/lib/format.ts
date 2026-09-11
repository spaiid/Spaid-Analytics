/**
 * format.ts — presentation only.
 *
 * Formatting a number is allowed; deciding what it means is not. Nothing here
 * computes a metric, threshold, band or verdict — those arrive from the API.
 *
 * Every function returns the em dash for null/undefined/non-finite input, so a
 * missing value can never render as 0.
 */

export const DASH = '—';

function finite(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/** Plain number with fixed decimals. */
export function num(value: number | null | undefined, digits = 2): string {
  if (!finite(value)) return DASH;
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** Integer with thousands separators. */
export function int(value: number | null | undefined): string {
  if (!finite(value)) return DASH;
  return Math.round(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

/** Signed number, e.g. "+3.20". */
export function signed(value: number | null | undefined, digits = 2): string {
  if (!finite(value)) return DASH;
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
}

/** Takes a FRACTION: pct(0.12) -> "12.0%". */
export function pct(value: number | null | undefined, digits = 1): string {
  if (!finite(value)) return DASH;
  return `${(value * 100).toFixed(digits)}%`;
}

/** Takes a FRACTION: signedPct(0.12) -> "+12.0%". */
export function signedPct(value: number | null | undefined, digits = 1): string {
  if (!finite(value)) return DASH;
  const sign = value >= 0 ? '+' : '−';
  return `${sign}${Math.abs(value * 100).toFixed(digits)}%`;
}

/** K / M / B / T. */
export function compact(value: number | null | undefined, digits = 1): string {
  if (!finite(value)) return DASH;
  const abs = Math.abs(value);
  if (abs >= 1e12) return `${(value / 1e12).toFixed(digits)}T`;
  if (abs >= 1e9) return `${(value / 1e9).toFixed(digits)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(digits)}M`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(digits)}K`;
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

/** Currency with symbol, e.g. "$12.34". */
export function currency(value: number | null | undefined, digits = 2, code = 'USD'): string {
  if (!finite(value)) return DASH;
  return value.toLocaleString(undefined, {
    style: 'currency',
    currency: code,
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** Currency at K/M/B/T scale, e.g. "$1.2B" — for market caps. */
export function compactCurrency(value: number | null | undefined, code = 'USD'): string {
  if (!finite(value)) return DASH;
  const symbol = code === 'USD' ? '$' : `${code} `;
  const sign = value < 0 ? '−' : '';
  return `${sign}${symbol}${compact(Math.abs(value))}`;
}

/** A 0-100 score at one decimal. */
export function score(value: number | null | undefined): string {
  if (!finite(value)) return DASH;
  return value.toFixed(1);
}

/** Signed points, e.g. a score change: "+2.4". */
export function points(value: number | null | undefined, digits = 1): string {
  if (!finite(value)) return DASH;
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
}

/** Multiple, e.g. "12.4x". */
export function multiple(value: number | null | undefined, digits = 1): string {
  if (!finite(value)) return DASH;
  return `${value.toFixed(digits)}x`;
}

function toDate(value: string | Date | null | undefined): Date | null {
  if (value === null || value === undefined || value === '') return null;
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value;
  // Bare "YYYY-MM-DD" is parsed as UTC midnight; anchor it locally instead so
  // the rendered day never slips backwards in western time zones.
  const raw = /^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00` : value;
  const parsed = new Date(raw);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

/** "3 Mar 2025" in the user's locale. */
export function date(value: string | Date | null | undefined): string {
  const parsed = toDate(value);
  if (!parsed) return DASH;
  return parsed.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

/** "3 Mar" in the user's locale. */
export function dateShort(value: string | Date | null | undefined): string {
  const parsed = toDate(value);
  if (!parsed) return DASH;
  return parsed.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/** Date + time, for `checked_at`. */
export function dateTime(value: string | Date | null | undefined): string {
  const parsed = toDate(value);
  if (!parsed) return DASH;
  return parsed.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

/** Hours as a human span: "3h ago" scale values like `age_hours`. */
export function ageHours(value: number | null | undefined): string {
  if (!finite(value)) return DASH;
  if (value < 1) return `${Math.max(1, Math.round(value * 60))}m`;
  if (value < 48) return `${value.toFixed(value < 10 ? 1 : 0)}h`;
  const days = value / 24;
  if (days < 60) return `${days.toFixed(days < 10 ? 1 : 0)}d`;
  return `${(days / 30.44).toFixed(1)}mo`;
}

/** Any already-formatted string, with the em dash for empty input. */
export function dash(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return DASH;
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : DASH;
  return value.trim() === '' ? DASH : value;
}

/** "12 of 480" style counts. */
export function ofTotal(value: number | null | undefined, total: number | null | undefined): string {
  return `${int(value)} of ${int(total)}`;
}

/** Title Case for API enum-ish strings: "deep_value" -> "Deep value". */
export function humanize(value: string | null | undefined): string {
  if (value === null || value === undefined || value.trim() === '') return DASH;
  const spaced = value.replace(/[_-]+/g, ' ').trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
