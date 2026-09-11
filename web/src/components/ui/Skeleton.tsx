import type { CSSProperties } from 'react';

export interface SkeletonProps {
  width?: number | string;
  height?: number | string;
  radius?: number | string;
  className?: string;
  style?: CSSProperties;
}

/** A loading placeholder. Decorative: mark the container `aria-busy`. */
export function Skeleton({ width = '100%', height = 14, radius, className, style }: SkeletonProps) {
  return (
    <div
      className={['skeleton', className].filter(Boolean).join(' ')}
      aria-hidden="true"
      style={{ width, height, borderRadius: radius, ...style }}
    />
  );
}

export interface SkeletonTextProps {
  lines?: number;
  className?: string;
}

export function SkeletonText({ lines = 3, className }: SkeletonTextProps) {
  return (
    <div className={['stack', className].filter(Boolean).join(' ')} style={{ gap: 8 }} aria-hidden="true">
      {Array.from({ length: lines }, (_, index) => (
        <Skeleton key={index} height={12} width={index === lines - 1 ? '60%' : '100%'} />
      ))}
    </div>
  );
}

export interface SkeletonTableProps {
  rows?: number;
  columns?: number;
  /** Accessible status text announced while the table loads. */
  label?: string;
  className?: string;
}

export function SkeletonTable({ rows = 6, columns = 5, label = 'Loading…', className }: SkeletonTableProps) {
  return (
    <div className={className} role="status" aria-live="polite" aria-busy="true">
      <span className="sr-only">{label}</span>
      <div className="stack" style={{ gap: 9 }} aria-hidden="true">
        <Skeleton height={11} width="34%" />
        {Array.from({ length: rows }, (_, rowIndex) => (
          <div key={rowIndex} className="row" style={{ gap: 12, flexWrap: 'nowrap' }}>
            {Array.from({ length: columns }, (_, columnIndex) => (
              <Skeleton key={columnIndex} height={13} width={columnIndex === 0 ? '22%' : '100%'} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
