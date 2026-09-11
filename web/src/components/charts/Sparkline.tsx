import { isNum, useElementWidth } from './chartUtils';

export interface SparklineProps {
  values: Array<number | null>;
  /** Required: sparklines carry no axes, so the label does the explaining. */
  ariaLabel: string;
  height?: number;
  color?: string;
  /** Paint the most recent quarter of the run in the accent hue. */
  emphasisLast?: boolean;
  className?: string;
}

/**
 * Inline trend line. The run sits in the de-emphasis hue and only the current
 * stretch takes the accent, so a row of sparklines reads as texture, not noise.
 */
export function Sparkline({
  values,
  ariaLabel,
  height = 30,
  color = 'var(--series-1)',
  emphasisLast = true,
  className,
}: SparklineProps) {
  const [hostRef, measured] = useElementWidth<HTMLDivElement>(120);
  const width = Math.max(60, measured);
  const finite = values.filter(isNum);

  const content = (() => {
    if (finite.length < 2) return null;
    const lo = Math.min(...finite);
    const hi = Math.max(...finite);
    const span = hi - lo || 1;
    const sx = (index: number) => (index / Math.max(1, values.length - 1)) * (width - 4) + 2;
    const sy = (value: number) => height - 3 - ((value - lo) / span) * (height - 6);

    let base = '';
    values.forEach((value, index) => {
      if (isNum(value)) base += `${base === '' ? 'M' : 'L'}${sx(index)},${sy(value)}`;
    });

    let recent = '';
    let lastPoint: { x: number; y: number } | null = null;
    if (emphasisLast && values.length > 1) {
      const count = Math.max(2, Math.floor(values.length * 0.25));
      values.slice(-count).forEach((value, offset) => {
        const index = values.length - count + offset;
        if (isNum(value)) recent += `${recent === '' ? 'M' : 'L'}${sx(index)},${sy(value)}`;
      });
      const last = values[values.length - 1];
      if (isNum(last)) lastPoint = { x: sx(values.length - 1), y: sy(last) };
    }

    return (
      <>
        <path d={base} fill="none" stroke="var(--de-emphasis)" strokeWidth={1.5} strokeLinejoin="round" />
        {recent !== '' && (
          <path d={recent} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        )}
        {lastPoint && <circle cx={lastPoint.x} cy={lastPoint.y} r={2.5} fill={color} />}
      </>
    );
  })();

  return (
    <div className={['chart', className].filter(Boolean).join(' ')} ref={hostRef} style={{ height }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        height={height}
        preserveAspectRatio="none"
        role="img"
        aria-label={content === null ? `${ariaLabel} (not enough data)` : ariaLabel}
      >
        {content}
      </svg>
    </div>
  );
}
