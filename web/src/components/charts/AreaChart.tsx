import { useRef, useState, type ReactNode } from 'react';
import { DASH, date as fmtDate, dateShort, pct } from '../../lib/format';
import { ChartFrame, ChartSvg, NoData } from './ChartFrame';
import { ChartTooltip, type ChartTooltipState } from './ChartTooltip';
import { indexFromPointer, isNum, linePath, niceTicks } from './chartUtils';

export interface AreaChartProps {
  x: string[];
  values: Array<number | null>;
  ariaLabel: string;
  name?: string;
  height?: number;
  color?: string;
  yFormat?: (value: number) => string;
  xFormat?: (value: string) => string;
  xLongFormat?: (value: string) => string;
  xHeader?: string;
  title?: ReactNode;
  subtitle?: ReactNode;
  className?: string;
}

/** Single-series area. Fill sits at ~10% opacity under a 2px stroke. */
export function AreaChart({
  x,
  values,
  ariaLabel,
  name = 'Value',
  height = 200,
  color = 'var(--series-1)',
  yFormat = (value) => pct(value, 1),
  xFormat = dateShort,
  xLongFormat = fmtDate,
  xHeader = 'Date',
  title,
  subtitle,
  className,
}: AreaChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const table = {
    headers: [xHeader, name],
    rows: x.map((label, index) => {
      const value = values[index];
      return [xLongFormat(label), isNum(value) ? yFormat(value) : DASH];
    }),
  };

  return (
    <ChartFrame ariaLabel={ariaLabel} title={title} subtitle={subtitle} table={table} className={className}>
      {(width) => {
        const padL = 54;
        const padR = 16;
        const padT = 12;
        const padB = 28;
        const plotW = width - padL - padR;
        const plotH = height - padT - padB;

        const finite = values.filter(isNum);
        if (finite.length === 0) {
          return (
            <ChartSvg width={width} height={height} ariaLabel={`${ariaLabel} (no data)`} svgRef={svgRef}>
              <NoData width={width} height={height} />
            </ChartSvg>
          );
        }

        const yMax = Math.max(0, ...finite);
        const yMin = Math.min(0, ...finite);
        const span = yMax - yMin || 1;
        const sx = (index: number) => padL + (values.length === 1 ? plotW / 2 : (index / (values.length - 1)) * plotW);
        const sy = (value: number) => padT + plotH - ((value - yMin) / span) * plotH;

        const d = linePath(values, sx, sy);
        const zeroY = sy(Math.max(yMin, Math.min(0, yMax)));
        const areaPath = d === '' ? '' : `${d} L${sx(values.length - 1)},${zeroY} L${sx(0)},${zeroY} Z`;

        const tickCount = Math.min(6, x.length);
        const xTickIndexes: number[] = [];
        for (let k = 0; k < tickCount; k += 1) {
          const index = Math.round((k / Math.max(1, tickCount - 1)) * (x.length - 1));
          if (!xTickIndexes.includes(index)) xTickIndexes.push(index);
        }

        const hoverValue = hover === null ? null : values[hover];
        const tooltip: ChartTooltipState | null =
          hover !== null && isNum(hoverValue)
            ? {
                x: sx(hover),
                y: padT + plotH / 2,
                title: xLongFormat(x[hover] ?? ''),
                rows: [{ color, name, value: yFormat(hoverValue) }],
              }
            : null;

        return (
          <>
            <ChartSvg width={width} height={height} ariaLabel={ariaLabel} svgRef={svgRef}>
              {niceTicks(yMin, yMax, 4).map((tick) => (
                <g key={`y-${tick}`}>
                  <line className="grid-line" x1={padL} x2={padL + plotW} y1={sy(tick)} y2={sy(tick)} />
                  <text className="tick-text" x={padL - 9} y={sy(tick) + 3.5} textAnchor="end">
                    {yFormat(tick)}
                  </text>
                </g>
              ))}

              {areaPath !== '' && <path d={areaPath} fill={color} opacity={0.1} />}
              {d !== '' && <path className="series-line" d={d} stroke={color} />}

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

              <line className="axis-line" x1={padL} x2={padL + plotW} y1={sy(0)} y2={sy(0)} />

              {hover !== null && isNum(hoverValue) && (
                <>
                  <line className="crosshair" x1={sx(hover)} x2={sx(hover)} y1={padT} y2={padT + plotH} />
                  <circle className="hover-dot" cx={sx(hover)} cy={sy(hoverValue)} r={4.5} fill={color} />
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
                  setHover(indexFromPointer(event.clientX, svg, width, padL, plotW, values.length));
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
