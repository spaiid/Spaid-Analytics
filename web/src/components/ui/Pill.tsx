import type { ReactNode } from 'react';
import type { Assessment, Level } from '../../api/types';
import { StatusIcon, toneFromLevel, type StatusTone } from './Icons';

export type PillTone = StatusTone;

export interface PillProps {
  /** Colour + icon. Defaults to neutral. */
  tone?: PillTone;
  children: ReactNode;
  /** Pass `false` to drop the icon (only for non-status labels). */
  icon?: boolean;
  size?: 'sm' | 'lg';
  title?: string;
  className?: string;
}

/**
 * A status chip. Colour ALWAYS travels with an icon and a text label, so the
 * meaning survives greyscale, CVD and a bad monitor.
 */
export function Pill({ tone = 'neutral', children, icon = true, size = 'sm', title, className }: PillProps) {
  const classes = ['pill', `pill-${tone}`, size === 'lg' ? 'pill-lg' : null, className]
    .filter(Boolean)
    .join(' ');
  return (
    <span className={classes} title={title}>
      {icon && <StatusIcon tone={tone} className="pill-icon" size={size === 'lg' ? 13 : 11} />}
      <span>{children}</span>
    </span>
  );
}

export { toneFromLevel };
export type { Level };

export interface AssessmentPillProps {
  assessment: Assessment | null | undefined;
  /** Shown when the API sent no assessment. */
  fallback?: ReactNode;
  size?: 'sm' | 'lg';
  className?: string;
}

/**
 * Renders the server's qualitative reading verbatim. The client never decides
 * what a number means — it only paints the level the API already chose.
 */
export function AssessmentPill({ assessment, fallback, size = 'sm', className }: AssessmentPillProps) {
  if (!assessment) {
    if (fallback === undefined) return null;
    return (
      <Pill tone="neutral" size={size} className={className}>
        {fallback}
      </Pill>
    );
  }
  return (
    <Pill tone={toneFromLevel(assessment.level)} size={size} className={className}>
      {assessment.text}
    </Pill>
  );
}
