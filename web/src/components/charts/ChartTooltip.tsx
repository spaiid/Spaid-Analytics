import { useLayoutEffect, useRef, useState } from 'react';

export interface ChartTooltipRow {
  name: string;
  /** Already formatted — values lead, labels follow. */
  value: string;
  color?: string;
}

export interface ChartTooltipState {
  /** Position in chart (viewBox === CSS px) coordinates. */
  x: number;
  y: number;
  title?: string;
  rows: ChartTooltipRow[];
}

export interface ChartTooltipProps {
  state: ChartTooltipState | null;
  /** Width of the chart host, used to flip the bubble at the right edge. */
  containerWidth: number;
}

/**
 * The hover card. Positioned with the same rule as the ported JS: 14px to the
 * right of the anchor, flipped left when it would overflow, clamped to 4px.
 * Purely decorative — the same numbers are always in the table twin.
 */
export function ChartTooltip({ state, containerWidth }: ChartTooltipProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;
    const width = node.offsetWidth;
    const height = node.offsetHeight;
    setBox((previous) =>
      Math.abs(previous.width - width) > 0.5 || Math.abs(previous.height - height) > 0.5
        ? { width, height }
        : previous,
    );
  }, [state]);

  if (!state) return null;

  let left = state.x + 14;
  if (box.width > 0 && left + box.width > containerWidth - 4) left = state.x - box.width - 14;
  if (left < 4) left = 4;
  const top = Math.max(4, state.y - box.height / 2);

  return (
    <div
      ref={ref}
      className="chart-tooltip is-visible"
      aria-hidden="true"
      style={{ left, top, opacity: box.height === 0 ? 0 : undefined }}
    >
      {state.title !== undefined && <div className="tooltip-title">{state.title}</div>}
      {state.rows.map((row) => (
        <div className="tooltip-row" key={`${row.name}-${row.value}`}>
          {row.color && <span className="tooltip-key" style={{ background: row.color }} />}
          <span className="tooltip-name">{row.name}</span>
          <span className="tooltip-val">{row.value}</span>
        </div>
      ))}
    </div>
  );
}
