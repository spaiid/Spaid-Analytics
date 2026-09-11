import { DASH } from '../../lib/format';

export type MeterTone = 'default' | 'good' | 'warning' | 'serious' | 'critical' | 'neutral';

export interface MeterProps {
  /** null renders a hatched, unmistakably empty track — never a zero fill. */
  value: number | null | undefined;
  min?: number;
  max?: number;
  /** Accessible name, e.g. "Composite score". Required. */
  label: string;
  /** Spoken value, e.g. "72.4 out of 100". Defaults to the numeric value. */
  valueText?: string;
  tone?: MeterTone;
  size?: 'sm' | 'lg';
  className?: string;
}

/**
 * A bounded bar. `role="meter"` (not progressbar) because the value is a
 * measurement, not task progress.
 */
export function Meter({
  value,
  min = 0,
  max = 100,
  label,
  valueText,
  tone = 'default',
  size = 'sm',
  className,
}: MeterProps) {
  const known = typeof value === 'number' && Number.isFinite(value);
  const span = max - min || 1;
  const clamped = known ? Math.min(max, Math.max(min, value)) : min;
  const fraction = known ? (clamped - min) / span : 0;

  const classes = [
    'meter',
    size === 'lg' ? 'meter-lg' : null,
    tone === 'default' ? null : `meter-${tone}`,
    known ? null : 'is-empty',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div
      className={classes}
      role="meter"
      aria-label={label}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={known ? clamped : undefined}
      aria-valuetext={known ? (valueText ?? `${clamped.toFixed(1)} of ${max}`) : `${DASH} not scored`}
    >
      {known && <div className="meter-fill" style={{ width: `${fraction * 100}%` }} />}
    </div>
  );
}

export interface ScoreMeterProps {
  /** 0-100 from the API. */
  score: number | null | undefined;
  label: string;
  tone?: MeterTone;
  size?: 'sm' | 'lg';
  /** Show the numeral next to the bar (right aligned, tabular). */
  showValue?: boolean;
  className?: string;
}

/** Score numeral + meter, the pairing used in the ranked table. */
export function ScoreMeter({ score, label, tone = 'default', size = 'sm', showValue = true, className }: ScoreMeterProps) {
  const known = typeof score === 'number' && Number.isFinite(score);
  return (
    <div className={['score-cell', className].filter(Boolean).join(' ')}>
      {showValue && <span className="score-num">{known ? score.toFixed(1) : DASH}</span>}
      <Meter value={score ?? null} label={label} tone={tone} size={size} />
    </div>
  );
}
