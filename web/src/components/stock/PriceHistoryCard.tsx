/**
 * PriceHistoryCard — the daily close, drawn as a line.
 *
 * Price history is an untyped dict server-side, so every array is treated as
 * possibly absent and possibly a different length from its siblings.
 */

import type { PriceHistory } from '../../api/types';
import { LineChart } from '../charts';
import { Card, EmptyState } from '../ui';
import { currency, date as fmtDate } from '../../lib/format';

function valueAt(values: number[] | undefined, index: number): number | null {
  if (!values) return null;
  const value: number | undefined = values[index];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export interface PriceHistoryCardProps {
  ticker: string;
  priceHistory: PriceHistory;
}

export function PriceHistoryCard({ ticker, priceHistory }: PriceHistoryCardProps) {
  const dates = priceHistory.dates ?? [];
  const closes = dates.map((_, index) => valueAt(priceHistory.close, index));
  const known = closes.filter((value): value is number => value !== null);

  if (dates.length === 0 || known.length === 0) {
    return (
      <Card title="Price history" headingLevel={3}>
        <EmptyState
          title="No price history"
          message="No daily bars are stored for this ticker, so the chart has nothing to draw. This is missing data, not a flat price."
        />
      </Card>
    );
  }

  const first = dates[0];
  const last = dates[dates.length - 1];
  const low = Math.min(...known);
  const high = Math.max(...known);
  const ariaLabel = `Line chart of ${ticker} daily closing price from ${fmtDate(first)} to ${fmtDate(
    last,
  )}, ranging ${currency(low)} to ${currency(high)}.`;

  return (
    <Card title="Price history" headingLevel={3}>
      <LineChart
        x={dates}
        series={[{ name: 'Close', values: closes }]}
        ariaLabel={ariaLabel}
        subtitle={`${known.length} daily closes · ${fmtDate(first)} to ${fmtDate(last)}`}
        yFormat={(value) => currency(value, 2)}
        height={280}
      />
    </Card>
  );
}
