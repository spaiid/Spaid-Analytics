/**
 * FinancialTrends — one small multiple per reported trend.
 *
 * The x axis is the period end (the quarter the numbers describe); the date the
 * filing became usable is carried in the tooltip-free table twin as context.
 */

import type { FinancialTrend } from '../../api/types';
import { AreaChart, seriesColor } from '../charts';
import { Card, EmptyState } from '../ui';
import { compactCurrency, date as fmtDate, num, pct } from '../../lib/format';

function formatterFor(unit: string): (value: number) => string {
  switch (unit) {
    case 'currency':
      return (value) => compactCurrency(value);
    case 'percent':
      return (value) => pct(value, 1);
    default:
      return (value) => num(value, 2);
  }
}

export interface FinancialTrendsProps {
  ticker: string;
  trends: FinancialTrend[];
}

export function FinancialTrends({ ticker, trends }: FinancialTrendsProps) {
  if (trends.length === 0) {
    return (
      <Card title="Financial trends" headingLevel={3}>
        <EmptyState
          title="No reported financials"
          message="No fundamentals are stored for this company, so there is nothing to trend. Treat this as a gap in the data, not as a weak set of results."
        />
      </Card>
    );
  }

  return (
    <Card
      title="Financial trends"
      headingLevel={3}
      subtitle="As reported, by period end. Each series is point-in-time: a value appears only from the date its filing became public."
    >
      <div className="sd-trends">
        {trends.map((trend, index) => {
          const x = trend.points.map((point) => point.period_end);
          const values = trend.points.map((point) => point.value);
          const known = values.filter((value) => Number.isFinite(value));
          const format = formatterFor(trend.unit);
          const first = x[0];
          const last = x[x.length - 1];
          const label =
            known.length === 0
              ? `${trend.label} for ${ticker}: no values reported.`
              : `Area chart of ${ticker} ${trend.label.toLowerCase()} by period end, ${fmtDate(first)} to ${fmtDate(
                  last,
                )}, ranging ${format(Math.min(...known))} to ${format(Math.max(...known))}.`;

          return (
            <AreaChart
              key={trend.concept}
              x={x}
              values={values}
              name={trend.label}
              ariaLabel={label}
              title={trend.label}
              subtitle={`${trend.points.length} periods · latest ${fmtDate(last)}`}
              color={seriesColor(index)}
              yFormat={format}
              xHeader="Period end"
              height={180}
            />
          );
        })}
      </div>
    </Card>
  );
}
