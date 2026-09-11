import { useState, type ReactNode } from 'react';
import { DASH, currency } from '../../lib/format';
import { ChartFrame, ChartSvg, NoData } from './ChartFrame';
import { ChartTooltip, type ChartTooltipState } from './ChartTooltip';
import { clamp, isNum, niceTicks } from './chartUtils';

interface Marker {
  key: string;
  name: string;
  value: number;
  color: string;
  shape: 'down' | 'circle' | 'up';
}

export interface RangeBarProps {
  /** The traded price — drawn as ink, never as a series colour. */
  price: number | null;
  bear?: number | null;
  base?: number | null;
  bull?: number | null;
  rangeLow?: number | null;
  rangeHigh?: number | null;
  midpoint?: number | null;
  ariaLabel: string;
  valueFormat?: (value: number) => string;
  height?: number;
  title?: ReactNode;
  subtitle?: ReactNode;
  className?: string;
}

const BEAR_COLOR = 'var(--series-2)';
const BASE_COLOR = 'var(--series-1)';
const BULL_COLOR = 'var(--series-3)';

function glyph(shape: Marker['shape'], x: number, y: number): string {
  const r = 5;
  switch (shape) {
    case 'up':
      return `M${x},${y - r} L${x + r},${y + r * 0.8} L${x - r},${y + r * 0.8} Z`;
    case 'down':
      return `M${x},${y + r} L${x + r},${y - r * 0.8} L${x - r},${y - r * 0.8} Z`;
    default:
      return `M${x - r},${y} a${r},${r} 0 1,0 ${r * 2},0 a${r},${r} 0 1,0 ${-r * 2},0`;
  }
}

/**
 * The fair-value range: bear / base / bull on a horizontal scale, the overall
 * range as a band, the midpoint, and where the current price sits inside it.
 * Every mark has a shape and a direct label as well as a colour.
 */
export function RangeBar({
  price,
  bear,
  base,
  bull,
  rangeLow,
  rangeHigh,
  midpoint,
  ariaLabel,
  valueFormat = (value) => currency(value, 2),
  height = 132,
  title,
  subtitle,
  className,
}: RangeBarProps) {
  const [hover, setHover] = useState<string | null>(null);

  const markers: Marker[] = [
    isNum(bear) ? { key: 'bear', name: 'Bear', value: bear, color: BEAR_COLOR, shape: 'down' as const } : null,
    isNum(base) ? { key: 'base', name: 'Base', value: base, color: BASE_COLOR, shape: 'circle' as const } : null,
    isNum(bull) ? { key: 'bull', name: 'Bull', value: bull, color: BULL_COLOR, shape: 'up' as const } : null,
  ].filter((marker): marker is Marker => marker !== null);

  const table = {
    headers: ['Point', 'Value per share'],
    rows: [
      ['Current price', isNum(price) ? valueFormat(price) : DASH],
      ['Bear', isNum(bear) ? valueFormat(bear) : DASH],
      ['Base', isNum(base) ? valueFormat(base) : DASH],
      ['Bull', isNum(bull) ? valueFormat(bull) : DASH],
      ['Range low', isNum(rangeLow) ? valueFormat(rangeLow) : DASH],
      ['Range high', isNum(rangeHigh) ? valueFormat(rangeHigh) : DASH],
      ['Fair value', isNum(midpoint) ? valueFormat(midpoint) : DASH],
    ],
  };

  const legend = [
    { name: 'Bear', color: BEAR_COLOR, shape: 'rect' as const },
    { name: 'Base', color: BASE_COLOR, shape: 'rect' as const },
    { name: 'Bull', color: BULL_COLOR, shape: 'rect' as const },
  ].filter((entry) => markers.some((marker) => marker.name === entry.name));

  return (
    <ChartFrame
      ariaLabel={ariaLabel}
      title={title}
      subtitle={subtitle}
      table={table}
      legend={legend}
      className={['rangebar', className].filter(Boolean).join(' ')}
    >
      {(width) => {
        const domainValues = [price, bear, base, bull, rangeLow, rangeHigh, midpoint].filter(isNum);
        if (domainValues.length === 0) {
          return (
            <ChartSvg width={width} height={height} ariaLabel={`${ariaLabel} (no data)`}>
              <NoData width={width} height={height} />
            </ChartSvg>
          );
        }

        const rawLow = Math.min(...domainValues);
        const rawHigh = Math.max(...domainValues);
        const pad = (rawHigh - rawLow) * 0.12 || Math.abs(rawHigh || 1) * 0.12;
        const lo = rawLow - pad;
        const hi = rawHigh + pad;

        const padL = 16;
        const padR = 16;
        const plotW = Math.max(1, width - padL - padR);
        const sx = (value: number) => padL + ((clamp(value, lo, hi) - lo) / (hi - lo || 1)) * plotW;

        const bandTop = 44;
        const bandHeight = 20;
        const bandBottom = bandTop + bandHeight;
        const axisY = bandBottom + 14;

        const bandLow = isNum(rangeLow) ? rangeLow : markers.length > 0 ? Math.min(...markers.map((m) => m.value)) : null;
        const bandHigh = isNum(rangeHigh) ? rangeHigh : markers.length > 0 ? Math.max(...markers.map((m) => m.value)) : null;

        // Direct labels for the markers, nudged apart so they never collide.
        const MIN_GAP = 62;
        const sorted = [...markers].sort((a, b) => a.value - b.value);
        const labelPositions = new Map<string, number>();
        let previous = -Infinity;
        for (const marker of sorted) {
          let x = sx(marker.value);
          if (x - previous < MIN_GAP) x = previous + MIN_GAP;
          x = clamp(x, padL + 14, width - padR - 14);
          labelPositions.set(marker.key, x);
          previous = x;
        }

        const hovered = hover === null ? null : markers.find((marker) => marker.key === hover) ?? null;
        const tooltip: ChartTooltipState | null = hovered
          ? {
              x: sx(hovered.value),
              y: bandTop,
              title: hovered.name,
              rows: [{ color: hovered.color, name: 'Value per share', value: valueFormat(hovered.value) }],
            }
          : null;

        return (
          <>
            <ChartSvg width={width} height={height} ariaLabel={ariaLabel}>
              {niceTicks(lo, hi, 4).map((tick) => (
                <g key={`tick-${tick}`}>
                  <line className="grid-line" x1={sx(tick)} x2={sx(tick)} y1={bandTop - 6} y2={bandBottom + 6} />
                  <text className="tick-text" x={sx(tick)} y={axisY + 10} textAnchor="middle">
                    {valueFormat(tick)}
                  </text>
                </g>
              ))}

              {isNum(bandLow) && isNum(bandHigh) && bandHigh > bandLow && (
                <>
                  <rect
                    className="band"
                    x={sx(bandLow)}
                    y={bandTop}
                    width={Math.max(1, sx(bandHigh) - sx(bandLow))}
                    height={bandHeight}
                    rx={4}
                    fill="var(--series-1)"
                  />
                  <rect
                    className="band-stroke"
                    x={sx(bandLow)}
                    y={bandTop}
                    width={Math.max(1, sx(bandHigh) - sx(bandLow))}
                    height={bandHeight}
                    rx={4}
                  />
                </>
              )}

              <line className="axis-line" x1={padL} x2={padL + plotW} y1={bandBottom + 6} y2={bandBottom + 6} />

              {isNum(midpoint) && (
                <g>
                  <line className="mid-line" x1={sx(midpoint)} x2={sx(midpoint)} y1={bandTop - 4} y2={bandBottom + 4} />
                  <text className="marker-name" x={sx(midpoint)} y={bandTop - 8} textAnchor="middle">
                    Midpoint
                  </text>
                </g>
              )}

              {markers.map((marker) => {
                const cx = sx(marker.value);
                const labelX = labelPositions.get(marker.key) ?? cx;
                return (
                  <g key={marker.key}>
                    <path className="mark" d={glyph(marker.shape, cx, bandBottom + 16)} fill={marker.color} />
                    {Math.abs(labelX - cx) > 1.5 && (
                      <line x1={cx} y1={bandBottom + 22} x2={labelX} y2={bandBottom + 28} stroke="var(--axis)" strokeWidth={1} />
                    )}
                    <text className="marker-label" x={labelX} y={bandBottom + 38} textAnchor="middle">
                      {valueFormat(marker.value)}
                    </text>
                    <text className="marker-name" x={labelX} y={bandBottom + 50} textAnchor="middle">
                      {marker.name}
                    </text>
                    <rect
                      className="bar-hit"
                      x={cx - 14}
                      y={bandTop}
                      width={28}
                      height={bandBottom + 24 - bandTop}
                      onPointerMove={() => setHover(marker.key)}
                      onPointerLeave={() => setHover(null)}
                    />
                  </g>
                );
              })}

              {isNum(price) && (
                <g>
                  <line className="price-line" x1={sx(price)} x2={sx(price)} y1={bandTop - 22} y2={bandBottom + 6} />
                  <path
                    d={`M${sx(price) - 5},${bandTop - 22} L${sx(price) + 5},${bandTop - 22} L${sx(price)},${bandTop - 14} Z`}
                    fill="var(--text-primary)"
                  />
                  <text
                    className="end-label"
                    x={clamp(sx(price), padL + 30, width - padR - 30)}
                    y={bandTop - 28}
                    textAnchor="middle"
                  >
                    {`Price ${valueFormat(price)}`}
                  </text>
                </g>
              )}
            </ChartSvg>
            <ChartTooltip state={tooltip} containerWidth={width} />
          </>
        );
      }}
    </ChartFrame>
  );
}
