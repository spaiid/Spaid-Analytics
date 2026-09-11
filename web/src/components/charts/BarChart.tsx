import { useState, type ReactNode } from 'react';
import { DASH, num } from '../../lib/format';
import { ChartFrame, ChartSvg, NoData } from './ChartFrame';
import { ChartTooltip, type ChartTooltipState } from './ChartTooltip';
import { BAR, barPath, isNum, niceTicks } from './chartUtils';

export interface BarChartProps {
  labels: string[];
  values: Array<number | null>;
  ariaLabel: string;
  /** Rows instead of columns. Row height is fixed at 26px (hit target >= 24px). */
  horizontal?: boolean;
  height?: number;
  /** One series -> one colour for every bar; never a value ramp on nominal categories. */
  color?: string;
  /** Overrides `color` and the diverging rule, e.g. to colour by sector slot. */
  colorFor?: (index: number, value: number) => string;
  /** Split positive/negative across the diverging poles and centre the axis on 0. */
  diverging?: boolean;
  valueFormat?: (value: number) => string;
  directLabels?: boolean;
  /** Series name used in the tooltip and the table twin header. */
  barName?: string;
  categoryHeader?: string;
  onBarActivate?: (index: number, label: string) => void;
  title?: ReactNode;
  subtitle?: ReactNode;
  className?: string;
}

/** Bars with a 4px rounded data-end and a square baseline end, max 24px thick. */
export function BarChart({
  labels,
  values,
  ariaLabel,
  horizontal = false,
  height = 260,
  color = 'var(--series-1)',
  colorFor,
  diverging = false,
  valueFormat = (value) => num(value, 2),
  directLabels = true,
  barName = 'Value',
  categoryHeader = 'Category',
  onBarActivate,
  title,
  subtitle,
  className,
}: BarChartProps) {
  const [hover, setHover] = useState<number | null>(null);

  const table = {
    headers: [categoryHeader, barName],
    rows: labels.map((label, index) => {
      const value = values[index];
      return [label, isNum(value) ? valueFormat(value) : DASH];
    }),
  };

  const barColor = (index: number, value: number): string => {
    if (colorFor) return colorFor(index, value);
    if (diverging) return value >= 0 ? 'var(--div-pos)' : 'var(--div-neg)';
    return color;
  };

  return (
    <ChartFrame ariaLabel={ariaLabel} title={title} subtitle={subtitle} table={table} className={className}>
      {(width) => {
        const finite = values.filter(isNum);
        if (labels.length === 0 || finite.length === 0) {
          return (
            <ChartSvg width={width} height={height} ariaLabel={`${ariaLabel} (no data)`}>
              <NoData width={width} height={height} />
            </ChartSvg>
          );
        }

        const vMax = Math.max(0, ...finite);
        const vMin = Math.min(0, ...finite);
        const lo = diverging ? Math.min(vMin, -Math.abs(vMax)) : Math.min(0, vMin);

        if (horizontal) {
          const rowH = 26;
          const svgH = Math.max(height, labels.length * rowH + 34);
          const padL = 118;
          const padR = 58;
          const padT = 8;
          const padB = 26;
          const plotW = width - padL - padR;
          const plotH = svgH - padT - padB;
          const hi = diverging ? Math.max(vMax, Math.abs(vMin)) : Math.max(0, vMax);
          const span = hi - lo || 1;
          const sx = (value: number) => padL + ((value - lo) / span) * plotW;
          const zero = sx(0);
          const band = plotH / labels.length;
          const barH = Math.min(BAR.MAX_WIDTH, band - BAR.GAP * 2);

          const hoverValue = hover === null ? null : values[hover];
          const tooltip: ChartTooltipState | null =
            hover !== null && isNum(hoverValue)
              ? {
                  x: sx(hoverValue),
                  y: padT + hover * band + band / 2,
                  title: labels[hover] ?? '',
                  rows: [{ color: barColor(hover, hoverValue), name: barName, value: valueFormat(hoverValue) }],
                }
              : null;

          return (
            <>
              <ChartSvg width={width} height={svgH} ariaLabel={ariaLabel}>
                {niceTicks(lo, hi, 4).map((tick) => (
                  <g key={`t-${tick}`}>
                    <line className="grid-line" x1={sx(tick)} x2={sx(tick)} y1={padT} y2={padT + plotH} />
                    <text className="tick-text" x={sx(tick)} y={svgH - 8} textAnchor="middle">
                      {valueFormat(tick)}
                    </text>
                  </g>
                ))}
                <line className="axis-line" x1={zero} x2={zero} y1={padT} y2={padT + plotH} />

                {labels.map((label, index) => {
                  const value = values[index];
                  if (!isNum(value)) return null;
                  const y = padT + index * band + (band - barH) / 2;
                  const fill = barColor(index, value);
                  const x0 = Math.min(zero, sx(value));
                  const barW = Math.abs(sx(value) - zero);
                  const interactive = Boolean(onBarActivate);
                  return (
                    <g key={label + String(index)}>
                      <path
                        className="mark"
                        d={barPath(x0, y, barW, barH, BAR.RADIUS, value >= 0 ? 'right' : 'left')}
                        fill={fill}
                      />
                      <text className="tick-text" x={padL - 10} y={y + barH / 2 + 4} textAnchor="end">
                        {label}
                      </text>
                      {directLabels && (
                        <text
                          className="value-label"
                          x={value >= 0 ? sx(value) + 7 : sx(value) - 7}
                          y={y + barH / 2 + 4}
                          textAnchor={value >= 0 ? 'start' : 'end'}
                        >
                          {valueFormat(value)}
                        </text>
                      )}
                      {/* hit target spans the whole band (>=24px), not just the painted bar */}
                      <rect
                        className="bar-hit"
                        x={padL}
                        y={padT + index * band}
                        width={Math.max(0, plotW)}
                        height={band}
                        role={interactive ? 'button' : undefined}
                        tabIndex={interactive ? 0 : undefined}
                        aria-label={interactive ? `${label}: ${valueFormat(value)}` : undefined}
                        onPointerMove={() => setHover(index)}
                        onPointerLeave={() => setHover(null)}
                        onFocus={() => setHover(index)}
                        onBlur={() => setHover(null)}
                        onClick={interactive ? () => onBarActivate?.(index, label) : undefined}
                        onKeyDown={
                          interactive
                            ? (event) => {
                                if (event.key === 'Enter' || event.key === ' ') {
                                  event.preventDefault();
                                  onBarActivate?.(index, label);
                                }
                              }
                            : undefined
                        }
                      />
                    </g>
                  );
                })}
              </ChartSvg>
              <ChartTooltip state={tooltip} containerWidth={width} />
            </>
          );
        }

        const padL = 52;
        const padR = 14;
        const padT = 12;
        const padB = 38;
        const plotW = width - padL - padR;
        const plotH = height - padT - padB;
        const hi = diverging ? Math.max(vMax, Math.abs(vMin)) : Math.max(0, vMax) * 1.06;
        const span = hi - lo || 1;
        const sy = (value: number) => padT + plotH - ((value - lo) / span) * plotH;
        const zero = sy(0);
        const band = plotW / labels.length;
        const barW = Math.min(BAR.MAX_WIDTH, band - BAR.GAP * 2);

        const hoverValue = hover === null ? null : values[hover];
        const tooltip: ChartTooltipState | null =
          hover !== null && isNum(hoverValue)
            ? {
                x: padL + hover * band + band / 2,
                y: Math.min(zero, sy(hoverValue)),
                title: labels[hover] ?? '',
                rows: [{ color: barColor(hover, hoverValue), name: barName, value: valueFormat(hoverValue) }],
              }
            : null;

        return (
          <>
            <ChartSvg width={width} height={height} ariaLabel={ariaLabel}>
              {niceTicks(lo, hi, 5).map((tick) => (
                <g key={`t-${tick}`}>
                  <line className="grid-line" x1={padL} x2={padL + plotW} y1={sy(tick)} y2={sy(tick)} />
                  <text className="tick-text" x={padL - 9} y={sy(tick) + 3.5} textAnchor="end">
                    {valueFormat(tick)}
                  </text>
                </g>
              ))}
              <line className="axis-line" x1={padL} x2={padL + plotW} y1={zero} y2={zero} />

              {labels.map((label, index) => {
                const value = values[index];
                if (!isNum(value)) return null;
                const x = padL + index * band + (band - barW) / 2;
                const fill = barColor(index, value);
                const y0 = Math.min(zero, sy(value));
                const barH = Math.abs(sy(value) - zero);
                const interactive = Boolean(onBarActivate);
                return (
                  <g key={label + String(index)}>
                    <path
                      className="mark"
                      d={barPath(x, y0, barW, barH, BAR.RADIUS, value >= 0 ? 'up' : 'down')}
                      fill={fill}
                    />
                    {labels.length <= 16 && (
                      <text className="tick-text" x={x + barW / 2} y={height - 20} textAnchor="middle">
                        {label}
                      </text>
                    )}
                    {directLabels && labels.length <= 14 && (
                      <text
                        className="value-label"
                        x={x + barW / 2}
                        y={value >= 0 ? y0 - 6 : y0 + barH + 12}
                        textAnchor="middle"
                      >
                        {valueFormat(value)}
                      </text>
                    )}
                    <rect
                      className="bar-hit"
                      x={padL + index * band}
                      y={padT}
                      width={band}
                      height={Math.max(0, plotH)}
                      role={interactive ? 'button' : undefined}
                      tabIndex={interactive ? 0 : undefined}
                      aria-label={interactive ? `${label}: ${valueFormat(value)}` : undefined}
                      onPointerMove={() => setHover(index)}
                      onPointerLeave={() => setHover(null)}
                      onFocus={() => setHover(index)}
                      onBlur={() => setHover(null)}
                      onClick={interactive ? () => onBarActivate?.(index, label) : undefined}
                      onKeyDown={
                        interactive
                          ? (event) => {
                              if (event.key === 'Enter' || event.key === ' ') {
                                event.preventDefault();
                                onBarActivate?.(index, label);
                              }
                            }
                          : undefined
                      }
                    />
                  </g>
                );
              })}
            </ChartSvg>
            <ChartTooltip state={tooltip} containerWidth={width} />
          </>
        );
      }}
    </ChartFrame>
  );
}
