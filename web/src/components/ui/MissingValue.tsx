import type { MetricStatus } from '../../api/types';
import { DASH } from '../../lib/format';

export type MissingKind = MetricStatus | string;

interface Copy {
  mark: string;
  short: string;
  explain: string;
  className: string;
}

const COPY: Record<'missing' | 'not_applicable' | 'thin_peers' | 'unknown', Copy> = {
  missing: {
    mark: '?',
    short: 'Missing',
    explain: 'Missing: the input for this value was not available, so it was left out of the score.',
    className: 'missing-missing',
  },
  not_applicable: {
    mark: 'n/a',
    short: 'Not applicable',
    explain: 'Not applicable: this measure does not apply to this business model, so it was never expected.',
    className: 'missing-na',
  },
  thin_peers: {
    mark: 'thin',
    short: 'Thin peer group',
    explain: 'Thin peer group: too few comparable companies to rank this value with confidence.',
    className: 'missing-thin',
  },
  unknown: {
    mark: '?',
    short: 'No value',
    explain: 'No value was reported for this field.',
    className: 'missing-missing',
  },
};

function resolve(status: MissingKind | null | undefined): Copy {
  switch (status) {
    case 'missing':
      return COPY.missing;
    case 'not_applicable':
      return COPY.not_applicable;
    case 'thin_peers':
      return COPY.thin_peers;
    default:
      return COPY.unknown;
  }
}

export interface MissingValueProps {
  /** MetricDetail.status, or one of the same four strings. */
  status?: MissingKind | null;
  /** MetricDetail.status_detail — the server's explanation, shown verbatim. */
  detail?: string | null;
  /** What is missing, e.g. "Gross margin" — used in the spoken text. */
  label?: string;
  className?: string;
}

/**
 * The em dash, plus an affordance that says WHY. Missing, not-applicable and
 * thin-peers read differently from each other and none of them can be mistaken
 * for a zero or a middling value.
 */
export function MissingValue({ status = 'missing', detail, label, className }: MissingValueProps) {
  const copy = resolve(status);
  const sentence = [label ? `${label}: ${copy.explain}` : copy.explain, detail ?? null]
    .filter((part): part is string => Boolean(part))
    .join(' ');

  return (
    <span className={['missing', copy.className, className].filter(Boolean).join(' ')} title={sentence}>
      <span className="missing-dash" aria-hidden="true">
        {DASH}
      </span>
      <span className="missing-mark" aria-hidden="true">
        {copy.mark}
      </span>
      <span className="sr-only">{sentence}</span>
    </span>
  );
}

/**
 * Render `value` when it is present, otherwise a MissingValue that explains the
 * absence. Keeps "0" and "—" visually distinct at every call site.
 */
export interface ValueOrMissingProps extends MissingValueProps {
  value: number | string | null | undefined;
  /** Formatter for a present numeric value. */
  format?: (value: number) => string;
}

export function ValueOrMissing({ value, format, ...missing }: ValueOrMissingProps) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return <>{format ? format(value) : String(value)}</>;
  }
  if (typeof value === 'string' && value.trim() !== '' && value !== DASH) {
    return <>{value}</>;
  }
  return <MissingValue {...missing} />;
}
