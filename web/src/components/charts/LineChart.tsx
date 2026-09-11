import { useRef, useState, type ReactNode } from 'react';
import { DASH, date as fmtDate, dateShort, num } from '../../lib/format';
import { ChartFrame, ChartSvg, NoData, type LegendEntry } from './ChartFrame';
import { ChartTooltip, type ChartTooltipState } from './ChartTooltip';
import { clamp, indexFromPointer, isNum, linePath, niceTicks, seriesColor } from './chartUtils';

export interface LineSeries {
  name: string;
  values: Array<number | null>;
  /** Defaults to the categorical slot colour. */
  color?: string;
  dashed?: boolean;
  /** false renders the series in the de-emphasis hue at 1.5px. */
  emphasis?: boolean;
}

export interface LineChartProps {
  /** Shared x categories, normally ISO dates. */
  x: string[];
  series: LineSeries[];
  /** Required: what the chart shows, for screen readers. */
  ariaLabel: string;
  height?: number;
  title?: ReactNode;
  subtitle?: ReactNode;
  yFormat?: (value: number) => string;
  xFormat?: (value: string) => string;
  /** Long-form x label used in the tooltip and the table twin. */
  xLongFormat?: (value: string) => string;
  tooltipTitle?: (index: number) => string;
  /** Label the last point of each series inline (<=4 series). */
  directLabelEnds?: boolean;
  /** Draw a reference line, e.g. 0 for a diverging series. */
  baselineAt?: number | null;
  /** Header for the first column of the table twin. */
  xHeader?: string;
  className?: string;
}

/** Multi-series line chart. 2px stroke, round caps, hairline grid. */
export function LineChart({
  x,
  series,
  ariaLabel,
  height = 260,
  title,
  subtitle,
  yFormat = (value) => num(value, 2),
  xFormat = dateShort,
  xLongFormat = fmtDate,
  tooltipTitle,
  directLabelEnds = true,
  baselineAt = null,
  xHeader = 'Date',
  className,
}: LineChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const resolved = series.map((entry, index) => ({
    ...entry,
    color: entry.color ?? seriesColor(index),
  }));

  const legend: LegendEntry[] = resolved.map((entry) => ({ name: entry.name, color: entry.color, shape: 'line' }));

  const table = {
    headers: [xHeader, ...resolved.map((entry) => entry.name)],
    rows: x.map((label, index) => [
      xLongFormat(label),
      ...resolved.map((entry) => {
        const value = entry.values[index];
        return isNum(value) ? yFormat(value) : DASH;
      }),
    ]),
  };

  return (
    <ChartFrame ariaLabel={ariaLabel} title={title} subtitle={subtitle} legend={legend} table={table} className={className}>
      {(width) => {
        const padL = 54;
        const padR = directLabelEnds && resolved.length <= 4 ? 66 : 16;
        const padT = 12;
        const padB = 28;
        const plotW = width - padL - padR;
        const plotH = height - padT - padB;

        const flat = resolved.flatMap((entry) => entry.values).filter(isNum);
        if (x.length === 0 || flat.length === 0) {
          return (
            <ChartSvg width={width} height={height} ariaLabel={`${ariaLabel} (no data)`} svgRef={svgRef}>
              <NoData width={width} height={height} />
            </ChartSvg>
          );
        }

        let yMin = Math.min(...flat);
        let yMax = Math.max(...flat);
        if (baselineAt !== null) {
          yMin = Math.min(yMin, baselineAt);
          yMax = Math.max(yMax, baselineAt);
        }
        const pad = (yMax - yMin) * 0.08 || Math.abs(yMax || 1) * 0.1;
        yMin -= pad;
        yMax += pad;

        const sx = (index: number) => padL + (x.length === 1 ? plotW / 2 : (index / (x.length - 1)) * plotW);
        const sy = (value: number) => padT + plotH - ((value - yMin) / (yMax - yMin || 1)) * plotH;

        const yTicks = niceTicks(yMin, yMax, 5);
        const tickCount = Math.min(6, x.length);
        const xTickIndexes: number[] = [];
        for (let k = 0; k < tickCount; k += 1) {
          const index = Math.round((k / Math.max(1, tickCount - 1)) * (x.length - 1));
          if (!xTickIndexes.includes(index)) xTickIndexes.push(index);
        }

        // Selective direct labels: the end point only, nudged just enough to
        // clear a neighbour, with a leader line carrying the connection.
        const ends = directLabelEnds && resolved.length <= 4
          ? resolved
              .map((entry) => {
                for (let index = entry.values.length - 1; index >= 0; index -= 1) {
                  const value = entry.values[index];
                  if (isNum(value)) return { entry, index, value };
                }
                return null;
              })
              .filter((item): item is { entry: (typeof resolved)[number]; index: number; value: number } => item !== null)
              .sort((a, b) => sy(a.value) - sy(b.value))
          : [];

        const MIN_GAP = 13;
        const placed: number[] = [];
        const endLabels = ends.map((end) => {
          let y = sy(end.value);
          const previous = placed[placed.length - 1];
          if (placed.length > 0 && previous !== undefined && y - previous < MIN_GAP) y = previous + MIN_GAP;
          placed.push(y);
          return { ...end, labelY: y, anchorY: sy(end.value), anchorX: sx(end.index) };
        });

        const hoverIndex = hover === null ? null : clamp(hover, 0, x.length - 1);
        const hoverRows =
          hoverIndex === null
            ? []
            : resolved
                .map((entry) => {
                  const value = entry.values[hoverIndex];
                  return isNum(value)
                    ? { color: entry.color, name: entry.name, value: yFormat(value) }
                    : null;
                })
                .filter((row): row is { color: string; name: string; value: string } => row !== null);

        const tooltip: ChartTooltipState | null =
          hoverIndex !== null && hoverRows.length > 0
            ? {
                x: sx(hoverIndex),
                y: padT + plotH / 2,
                title: tooltipTitle ? tooltipTitle(hoverIndex) : xLongFormat(x[hoverIndex] ?? ''),
                rows: hoverRows,
              }
            : null;

        return (
          <>
            <ChartSvg width={width} height={height} ariaLabel={ariaLabel} svgRef={svgRef}>
              {yTicks.map((tick) => (
                <g key={`y-${tick}`}>
                  <line className="grid-line" x1={padL} x2={padL + plotW} y1={sy(tick)} y2={sy(tick)} />
                  <text className="tick-text" x={padL - 9} y={sy(tick) + 3.5} textAnchor="end">
                    {yFormat(tick)}
                  </text>
                </g>
              ))}

              {baselineAt !== null && (
                <line className="zero-line" x1={padL} x2={padL + plotW} y1={sy(baselineAt)} y2={sy(baselineAt)} />
              )}

              {xTickIndexes.map((index, position) => (
                <text
                  key={`x-${index}`}
                  className="tick-text"
                  x={sx(index)}
                  y={height - 8}
                  textAnchor={position === 0 ? 'start' : position === xTickIndexes.length - 1 ? 'end' : 'middle'}
                >
                  {xFormat(x[index] ?? '')}
                </text>
              ))}

              <line className="axis-line" x1={padL} x2={padL + plotW} y1={padT + plotH} y2={padT + plotH} />

              {resolved.map((entry) => {
                const d = linePath(entry.values, sx, sy);
                if (d === '') return null;
                return (
                  <path
                    key={entry.name}
                    className="series-line"
                    d={d}
                    stroke={entry.emphasis === false ? 'var(--de-emphasis)' : entry.color}
                    strokeWidth={entry.emphasis === false ? 1.5 : 2}
                    strokeDasharray={entry.dashed ? '5 4' : undefined}
                  />
                );
              })}

              {endLabels.map((end) => (
                <g key={`end-${end.entry.name}`}>
                  <circle className="hover-dot" cx={end.anchorX} cy={end.anchorY} r={4} fill={end.entry.color} />
                  {Math.abs(end.labelY - end.anchorY) > 1.5 && (
                    <line
                      x1={end.anchorX + 5}
                      y1={end.anchorY}
                      x2={end.anchorX + 11}
                      y2={end.labelY}
                      stroke="var(--axis)"
                      strokeWidth={1}
                    />
                  )}
                  <text className="end-label" x={end.anchorX + 13} y={end.labelY + 3.5}>
                    {yFormat(end.value)}
                  </text>
                </g>
              ))}

              {hoverIndex !== null && (
                <>
                  <line className="crosshair" x1={sx(hoverIndex)} x2={sx(hoverIndex)} y1={padT} y2={padT + plotH} />
                  {resolved.map((entry) => {
                    const value = entry.values[hoverIndex];
                    if (!isNum(value)) return null;
                    return (
                      <circle
                        key={`dot-${entry.name}`}
                        className="hover-dot"
                        cx={sx(hoverIndex)}
                        cy={sy(value)}
                        r={4.5}
                        fill={entry.color}
                      />
                    );
                  })}
                </>
              )}

              <rect
                className="hit"
                x={padL}
                y={padT}
                width={Math.max(0, plotW)}
                height={Math.max(0, plotH)}
                onPointerMove={(event) => {
                  const svg = svgRef.current;
                  if (!svg) return;
                  setHover(indexFromPointer(event.clientX, svg, width, padL, plotW, x.length));
                }}
                onPointerLeave={() => setHover(null)}
              />
            </ChartSvg>
            <ChartTooltip state={tooltip} containerWidth={width} />
          </>
        );
      }}
    </ChartFrame>
  );
}
