import type { ReactNode } from 'react';
import { ArrowDownIcon, ArrowUpIcon, FlatIcon } from './Icons';
import { Tooltip } from './Tooltip';

export type DeltaDirection = 'up' | 'down' | 'flat';

export interface StatDelta {
  /** Already formatted, e.g. "+2.4" or "+1.8%". */
  text: ReactNode;
  direction: DeltaDirection;
  /** Spoken prefix, e.g. "up over 1 week". */
  srLabel?: string;
}

export interface StatTileProps {
  label: ReactNode;
  /** Already formatted by src/lib/format — this component never computes. */
  value: ReactNode;
  /** Secondary line under the value, e.g. a confidence label. */
  sub?: ReactNode;
  delta?: StatDelta | null;
  footer?: ReactNode;
  /** An explanation attached to the label. */
  hint?: string;
  small?: boolean;
  /** Drop the surface so the tile can sit inside another card. */
  bare?: boolean;
  /** Extra content under the value — a Meter, a Sparkline, a Pill. */
  children?: ReactNode;
  className?: string;
}

const DELTA_ICON: Record<DeltaDirection, typeof ArrowUpIcon> = {
  up: ArrowUpIcon,
  down: ArrowDownIcon,
  flat: FlatIcon,
};

/** A labelled figure. Direction carries an arrow as well as a colour. */
export function StatTile({
  label,
  value,
  sub,
  delta,
  footer,
  hint,
  small = false,
  bare = false,
  children,
  className,
}: StatTileProps) {
  const DeltaIcon = delta ? DELTA_ICON[delta.direction] : null;
  const classes = ['stat', bare ? 'is-bare' : null, className].filter(Boolean).join(' ');

  return (
    <div className={classes}>
      <div className="stat-label">
        {hint ? <Tooltip content={hint}>{label}</Tooltip> : label}
      </div>
      <div className={['stat-value', small ? 'is-small' : null].filter(Boolean).join(' ')}>{value}</div>
      {sub !== undefined && <div className="stat-foot">{sub}</div>}
      {delta && DeltaIcon && (
        <div className={`stat-delta ${delta.direction}`}>
          <DeltaIcon size={11} />
          <span>
            {delta.srLabel && <span className="sr-only">{delta.srLabel} </span>}
            {delta.text}
          </span>
        </div>
      )}
      {children}
      {footer !== undefined && <div className="stat-foot">{footer}</div>}
    </div>
  );
}
