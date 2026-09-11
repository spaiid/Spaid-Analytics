import { useMemo } from 'react';
import type { OpportunityRow } from '../../api/types';
import * as fmt from '../../lib/format';
import { BarChart } from '../charts';
import { isNum } from '../charts/chartUtils';
import { Card } from '../ui';
import { binScores, meanScoreBySector } from './derive';

const BIN_WIDTH = 5;

/**
 * Counts are whole companies. When the tallest bar is small the tick generator
 * lands on half steps, and "1, 1, 2, 2" up the axis is a lie — label only the
 * whole ones. Bar values and the table twin are always integers, so they are
 * unaffected.
 */
const countLabel = (value: number): string => (Number.isInteger(value) ? fmt.int(value) : '');

/** Signed points, except at the axis centre where "+0.0" reads as a value. */
const deltaLabel = (value: number): string => (value === 0 ? fmt.num(0, 1) : fmt.points(value));

function NotEnough({ message }: { message: string }) {
  return <p className="muted t-small">{message}</p>;
}

export interface ScoreDistributionChartProps {
  rows: OpportunityRow[];
}

/**
 * How the composite scores of the rows on screen are spread out. The scores are
 * the API's; only the bin counts are computed here, which is presentation.
 */
export function ScoreDistributionChart({ rows }: ScoreDistributionChartProps) {
  const bins = useMemo(() => {
    const scores: number[] = [];
    for (const row of rows) if (isNum(row.score)) scores.push(row.score);
    return binScores(scores, BIN_WIDTH);
  }, [rows]);

  if (bins.length === 0) {
    return <NotEnough message="No scored companies in the current selection, so there is nothing to plot." />;
  }

  const total = bins.reduce((sum, bin) => sum + bin.count, 0);
  const busiest = bins.reduce((best, bin) => (bin.count > best.count ? bin : best), bins[0]);
  const first = bins[0];
  const last = bins[bins.length - 1];

  const ariaLabel =
    `Histogram of composite scores for the ${fmt.int(total)} scored companies shown, ` +
    `in ${BIN_WIDTH}-point bins from ${fmt.int(first.from)} to ${fmt.int(last.to)}. ` +
    `The largest bin is ${fmt.int(busiest.from)} to ${fmt.int(busiest.to)} with ${fmt.int(busiest.count)} companies.`;

  return (
    <BarChart
      labels={bins.map((bin) => bin.label)}
      values={bins.map((bin) => bin.count)}
      ariaLabel={ariaLabel}
      title="Score distribution"
      subtitle={`${fmt.int(total)} scored companies shown, in ${BIN_WIDTH}-point bins.`}
      valueFormat={countLabel}
      barName="Companies"
      categoryHeader="Score range"
      height={236}
    />
  );
}

export interface SectorScoreChartProps {
  rows: OpportunityRow[];
  /** Clicking or pressing Enter on a bar filters the table to that sector. */
  onSelectSector: (sector: string) => void;
}

/**
 * Mean composite score per sector, drawn as the difference from the mean of
 * every scored row on screen — a diverging scale needs a real zero, and "the
 * average of what you are looking at" is the only one available. Nothing here
 * judges a sector: the poles are just above and below that average.
 */
export function SectorScoreChart({ rows, onSelectSector }: SectorScoreChartProps) {
  const summary = useMemo(() => meanScoreBySector(rows), [rows]);
  const { sectors, overallMean, scored, withoutSector } = summary;

  if (overallMean === null || sectors.length === 0) {
    return <NotEnough message="No scored companies in the current selection, so there is nothing to compare." />;
  }
  if (sectors.length === 1) {
    return (
      <NotEnough
        message={`Only one sector (${sectors[0].sector}) is in the current selection, so there is nothing to compare it with. Its mean score is ${fmt.score(sectors[0].mean)}.`}
      />
    );
  }

  const labels = sectors.map((entry) => entry.sector);
  const values = sectors.map((entry) => entry.mean - overallMean);
  const best = sectors[0];
  const worst = sectors[sectors.length - 1];

  const ariaLabel =
    `Diverging bar chart of mean composite score by sector, as the difference from the ` +
    `${fmt.score(overallMean)} mean across the ${fmt.int(scored)} scored companies shown. ` +
    `${fmt.int(sectors.length)} sectors, from ${best.sector} at ${fmt.points(best.mean - overallMean)} points ` +
    `to ${worst.sector} at ${fmt.points(worst.mean - overallMean)} points.`;

  const subtitle =
    `Difference from the ${fmt.score(overallMean)} mean of the ${fmt.int(scored)} scored companies shown.` +
    (withoutSector > 0
      ? ` ${fmt.int(withoutSector)} without a sector ${withoutSector === 1 ? 'is' : 'are'} not plotted.`
      : '');

  return (
    <BarChart
      labels={labels}
      values={values}
      ariaLabel={ariaLabel}
      title="Mean score by sector"
      subtitle={subtitle}
      horizontal
      diverging
      valueFormat={deltaLabel}
      barName="Difference from the overall mean, in points"
      categoryHeader="Sector"
      onBarActivate={(_index, label) => onSelectSector(label)}
      height={200}
    />
  );
}

export interface OpportunityChartsProps {
  rows: OpportunityRow[];
  onSelectSector: (sector: string) => void;
}

/** The two charts under the table, side by side down to 1180px. */
export function OpportunityCharts({ rows, onSelectSector }: OpportunityChartsProps) {
  return (
    <div className="opp-charts">
      <Card>
        <ScoreDistributionChart rows={rows} />
      </Card>
      <Card>
        <SectorScoreChart rows={rows} onSelectSector={onSelectSector} />
      </Card>
    </div>
  );
}
