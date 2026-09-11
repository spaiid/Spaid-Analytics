/**
 * Icons.tsx — small inline SVG glyphs.
 *
 * Status is never colour alone: each tone has a distinct SHAPE so the meaning
 * survives greyscale and every kind of colour-vision deficiency.
 */

import type { CSSProperties } from 'react';
import type { Level } from '../../api/types';

export type StatusTone = 'good' | 'warning' | 'serious' | 'critical' | 'neutral' | 'info';

/** Map the API's `Assessment.level` onto a tone. */
export function toneFromLevel(level: Level | null | undefined): StatusTone {
  switch (level) {
    case 'good':
      return 'good';
    case 'warning':
      return 'warning';
    case 'critical':
      return 'critical';
    case 'neutral':
      return 'neutral';
    default:
      return 'neutral';
  }
}

export interface IconProps {
  size?: number;
  className?: string;
  style?: CSSProperties;
}

interface GlyphProps extends IconProps {
  children: React.ReactNode;
}

function Glyph({ size = 12, className, style, children }: GlyphProps) {
  return (
    <svg
      viewBox="0 0 12 12"
      width={size}
      height={size}
      className={className}
      style={style}
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

const STROKE = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.4,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
};

/** Circle + tick. */
export function GoodIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <circle cx="6" cy="6" r="5" {...STROKE} />
      <path d="M3.7 6.2 L5.3 7.8 L8.4 4.3" {...STROKE} />
    </Glyph>
  );
}

/** Triangle + bang. */
export function WarningIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <path d="M6 1 L11.3 10.6 L0.7 10.6 Z" {...STROKE} />
      <path d="M6 4.6 L6 7.2" {...STROKE} />
      <circle cx="6" cy="9" r="0.75" fill="currentColor" stroke="none" />
    </Glyph>
  );
}

/** Diamond + bang — between warning and critical. */
export function SeriousIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <path d="M6 0.8 L11.2 6 L6 11.2 L0.8 6 Z" {...STROKE} />
      <path d="M6 3.5 L6 6.4" {...STROKE} />
      <circle cx="6" cy="8.3" r="0.75" fill="currentColor" stroke="none" />
    </Glyph>
  );
}

/** Octagon + cross. */
export function CriticalIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <path d="M4.1 0.8 H7.9 L11.2 4.1 V7.9 L7.9 11.2 H4.1 L0.8 7.9 V4.1 Z" {...STROKE} />
      <path d="M4.4 4.4 L7.6 7.6 M7.6 4.4 L4.4 7.6" {...STROKE} />
    </Glyph>
  );
}

/** Circle + dash — nothing to flag. */
export function NeutralIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <circle cx="6" cy="6" r="5" {...STROKE} />
      <path d="M3.8 6 L8.2 6" {...STROKE} />
    </Glyph>
  );
}

/** Circle + i. */
export function InfoIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <circle cx="6" cy="6" r="5" {...STROKE} />
      <path d="M6 5.4 L6 8.6" {...STROKE} />
      <circle cx="6" cy="3.5" r="0.75" fill="currentColor" stroke="none" />
    </Glyph>
  );
}

/** Question mark — used for missing data. */
export function QuestionIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <circle cx="6" cy="6" r="5" {...STROKE} />
      <path d="M4.5 4.5 A1.6 1.6 0 1 1 6 7.1 L6 7.9" {...STROKE} />
      <circle cx="6" cy="9.3" r="0.7" fill="currentColor" stroke="none" />
    </Glyph>
  );
}

export function ArrowUpIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <path d="M6 10 L6 2.4 M2.8 5.6 L6 2.4 L9.2 5.6" {...STROKE} />
    </Glyph>
  );
}

export function ArrowDownIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <path d="M6 2 L6 9.6 M2.8 6.4 L6 9.6 L9.2 6.4" {...STROKE} />
    </Glyph>
  );
}

export function FlatIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <path d="M2 6 L10 6" {...STROKE} />
    </Glyph>
  );
}

export function ChevronRightIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <path d="M4.4 2.4 L8 6 L4.4 9.6" {...STROKE} />
    </Glyph>
  );
}

export function ChevronDownIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <path d="M2.4 4.4 L6 8 L9.6 4.4" {...STROKE} />
    </Glyph>
  );
}

export function RefreshIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <path d="M10.2 6 A4.2 4.2 0 1 1 8.6 2.7" {...STROKE} />
      <path d="M10.6 1.2 L10.6 3.4 L8.4 3.4" {...STROKE} />
    </Glyph>
  );
}

export function ClockIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <circle cx="6" cy="6" r="5" {...STROKE} />
      <path d="M6 3.2 L6 6.2 L8.2 7.4" {...STROKE} />
    </Glyph>
  );
}

export function SunIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <circle cx="6" cy="6" r="2.6" {...STROKE} />
      <path d="M6 0.8 V2 M6 10 V11.2 M0.8 6 H2 M10 6 H11.2 M2.3 2.3 L3.2 3.2 M8.8 8.8 L9.7 9.7 M9.7 2.3 L8.8 3.2 M3.2 8.8 L2.3 9.7" {...STROKE} />
    </Glyph>
  );
}

export function MoonIcon(props: IconProps) {
  return (
    <Glyph {...props}>
      <path d="M10 7.4 A4.6 4.6 0 1 1 4.6 2 A3.8 3.8 0 0 0 10 7.4 Z" {...STROKE} />
    </Glyph>
  );
}

const TONE_ICONS: Record<StatusTone, (props: IconProps) => React.ReactElement> = {
  good: GoodIcon,
  warning: WarningIcon,
  serious: SeriousIcon,
  critical: CriticalIcon,
  neutral: NeutralIcon,
  info: InfoIcon,
};

export interface StatusIconProps extends IconProps {
  tone: StatusTone;
}

/** The icon that belongs to a tone. Always decorative — pair it with text. */
export function StatusIcon({ tone, ...rest }: StatusIconProps) {
  const Component = TONE_ICONS[tone];
  return <Component {...rest} />;
}
