import { useId, useState, type ReactNode } from 'react';
import { useElementWidth } from './chartUtils';

export interface LegendEntry {
  name: string;
  color: string;
  /** 'line' for strokes, 'rect' for filled marks. */
  shape?: 'line' | 'rect';
  muted?: boolean;
}

export interface ChartTableData {
  headers: string[];
  /** Pre-formatted cells. First column is the category/date. */
  rows: string[][];
}

export interface ChartFrameProps {
  /**
   * Summarises what the chart shows, e.g.
   * "Line chart of AAPL close price, 3 Jan 2024 to 8 Sep 2025, ranging 164 to 232".
   * Also names the table twin.
   */
  ariaLabel: string;
  title?: ReactNode;
  subtitle?: ReactNode;
  /** Extra header controls, placed before the chart/table toggle. */
  actions?: ReactNode;
  legend?: LegendEntry[];
  table: ChartTableData;
  /** Draw the SVG at the measured width. */
  children: (width: number) => ReactNode;
  minWidth?: number;
  className?: string;
}

/**
 * Wraps every chart: header, the chart/table toggle, the legend (only when
 * there are 2+ series) and the table-view twin that every chart must ship.
 */
export function ChartFrame({
  ariaLabel,
  title,
  subtitle,
  actions,
  legend,
  table,
  children,
  minWidth = 320,
  className,
}: ChartFrameProps) {
  const [hostRef, measured] = useElementWidth<HTMLDivElement>(640);
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const groupId = useId();
  const width = Math.max(minWidth, measured);
  const showLegend = Boolean(legend && legend.length >= 2);

  return (
    <div className={['chart-frame', className].filter(Boolean).join(' ')}>
      <div className="chart-head">
        {title !== undefined && <h3 className="chart-title">{title}</h3>}
        {subtitle !== undefined && <p className="chart-sub">{subtitle}</p>}
        {actions}
        <div className="chart-toggle" role="group" aria-labelledby={groupId}>
          <span className="sr-only" id={groupId}>
            View {ariaLabel} as
          </span>
          <button type="button" aria-pressed={view === 'chart'} onClick={() => setView('chart')}>
            Chart
          </button>
          <button type="button" aria-pressed={view === 'table'} onClick={() => setView('table')}>
            Table
          </button>
        </div>
      </div>

      <div className="chart" ref={hostRef}>
        {view === 'chart' ? (
          <>
            {children(width)}
            {showLegend && legend && (
              <ul className="legend">
                {legend.map((entry) => (
                  <li className={['legend-item', entry.muted ? 'is-muted' : null].filter(Boolean).join(' ')} key={entry.name}>
                    <span
                      className={['legend-key', entry.shape === 'rect' ? 'rect' : null].filter(Boolean).join(' ')}
                      style={{ background: entry.color }}
                    />
                    <span>{entry.name}</span>
                  </li>
                ))}
              </ul>
            )}
          </>
        ) : (
          <div className="chart-table table-wrap">
            <table className="data is-sticky">
              <caption className="sr-only">{ariaLabel}</caption>
              <thead>
                <tr>
                  {table.headers.map((header, index) => (
                    <th key={header + String(index)} scope="col" className={index > 0 ? 'num' : undefined}>
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, rowIndex) => (
                  <tr key={`${row[0] ?? ''}-${rowIndex}`}>
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex} className={cellIndex > 0 ? 'num' : undefined}>
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export interface ChartSvgProps {
  width: number;
  height: number;
  /** Every chart SVG is described in a sentence, whatever its role. */
  ariaLabel: string;
  /**
   * True when the chart contains focusable marks (bars a keyboard user can tab
   * to). `role="img"` prunes an element's descendants from the accessibility
   * tree, so a focusable bar inside one receives focus that a screen reader
   * never announces. Such a chart is a `group` instead: same label, descendants
   * preserved.
   */
  hasInteractiveMarks?: boolean;
  children: ReactNode;
  className?: string;
  svgRef?: React.Ref<SVGSVGElement>;
}

/** The <svg> element every chart draws into, always with an accessible name. */
export function ChartSvg({
  width,
  height,
  ariaLabel,
  hasInteractiveMarks = false,
  children,
  className,
  svgRef,
}: ChartSvgProps) {
  return (
    <svg
      ref={svgRef}
      className={className}
      viewBox={`0 0 ${width} ${height}`}
      height={height}
      role={hasInteractiveMarks ? 'group' : 'img'}
      aria-label={ariaLabel}
    >
      {children}
    </svg>
  );
}

export interface NoDataProps {
  width: number;
  height: number;
  message?: string;
}

/** The in-chart empty state. Distinct from a failed fetch (see ErrorState). */
export function NoData({ width, height, message = 'No data' }: NoDataProps) {
  return (
    <text x={width / 2} y={height / 2} textAnchor="middle" className="empty-text">
      {message}
    </text>
  );
}
