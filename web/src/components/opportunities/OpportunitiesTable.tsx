import { useMemo, type ReactNode } from 'react';
import type { OpportunityRow } from '../../api/types';
import * as fmt from '../../lib/format';
import { isNum } from '../charts/chartUtils';
import { DataTable, MissingValue, Pill, ScoreMeter, ValueOrMissing } from '../ui';
import type { Column, SortState } from '../ui';
import { bandTone, valuationTone } from './labels';
import { ScoreDelta } from './ScoreDelta';

/** Rank ascending: the order the API itself ranked the universe in. */
export const DEFAULT_OPPORTUNITY_SORT: SortState = { key: 'rank', direction: 'asc' };

export interface OpportunitiesTableProps {
  rows: OpportunityRow[];
  /** Band -> position in the API's own best-first band list, for sorting. */
  bandOrder: Map<string, number>;
  onOpen: (ticker: string) => void;
  /** Ticker of the row to mark as current, if any. */
  activeTicker?: string | null;
  /**
   * Controlled sort. The view holds it so that filtering down to nothing and
   * back does not silently throw the reader's chosen order away.
   */
  sort?: SortState | null;
  onSortChange?: (sort: SortState) => void;
  empty?: ReactNode;
}

/** A sub-score cell: the number, or an explained dash. Never a zero. */
function CategoryScore({ value, label }: { value: number | null; label: string }) {
  return <ValueOrMissing value={value} format={fmt.score} status="missing" label={label} />;
}

/**
 * The ranked list. Every column that holds a number is sortable through a real
 * button with aria-sort; rows are focusable and open the stock detail on Enter,
 * Space or click.
 */
export function OpportunitiesTable({
  rows,
  bandOrder,
  onOpen,
  activeTicker,
  sort,
  onSortChange,
  empty,
}: OpportunitiesTableProps) {
  const columns = useMemo<Array<Column<OpportunityRow>>>(
    () => [
      {
        key: 'rank',
        header: '#',
        title: 'Rank in the universe, best first. Only scored companies are ranked.',
        sortLabel: 'rank',
        numeric: true,
        align: 'left',
        width: '52px',
        defaultDirection: 'asc',
        sortValue: (row) => row.rank,
        render: (row) =>
          isNum(row.rank) ? (
            <span className="rank-badge">{fmt.int(row.rank)}</span>
          ) : (
            <MissingValue status="missing" label="Rank" detail="Only scored companies are ranked." />
          ),
      },
      {
        key: 'ticker',
        header: 'Company',
        // The spoken name keeps the visible word, so voice control can target it.
        sortLabel: 'company ticker',
        width: '216px',
        sortValue: (row) => row.ticker,
        render: (row) => (
          <div className="opp-ticker">
            <div className="ticker-cell">
              <span className="ticker-sym">{row.ticker}</span>
              <span className="ticker-name" title={row.name}>
                {row.name}
              </span>
            </div>
            {row.top_reason ? (
              <div className="opp-reason" title={row.top_reason}>
                {row.top_reason}
              </div>
            ) : null}
          </div>
        ),
      },
      {
        key: 'sector',
        header: 'Sector',
        sortLabel: 'sector',
        width: '118px',
        sortValue: (row) => row.sector,
        render: (row) => (
          <span className="secondary">
            <ValueOrMissing value={row.sector} status="missing" label="Sector" />
          </span>
        ),
      },
      {
        key: 'score',
        header: 'Score',
        title: 'Composite score, 0-100, computed by the API.',
        sortLabel: 'composite score',
        numeric: true,
        width: '124px',
        sortValue: (row) => row.score,
        render: (row) =>
          isNum(row.score) ? (
            <ScoreMeter score={row.score} label={`Composite score for ${row.ticker}`} />
          ) : (
            <MissingValue status="missing" label={`Composite score for ${row.ticker}`} />
          ),
      },
      {
        key: 'band',
        header: 'Band',
        title: 'The reading the API attaches to the composite score.',
        sortLabel: 'band',
        width: '136px',
        defaultDirection: 'asc',
        // Sorted by the server's own band order, not alphabetically.
        sortValue: (row) => (row.band ? bandOrder.get(row.band) ?? null : null),
        render: (row) =>
          row.band ? (
            <Pill tone={bandTone(row.band)}>{row.band}</Pill>
          ) : (
            <MissingValue status="missing" label="Band" />
          ),
      },
      {
        key: 'quality',
        header: 'Quality',
        sortLabel: 'quality score',
        numeric: true,
        width: '78px',
        sortValue: (row) => row.quality,
        render: (row) => <CategoryScore value={row.quality} label={`Quality score for ${row.ticker}`} />,
      },
      {
        key: 'growth',
        header: 'Growth',
        sortLabel: 'growth score',
        numeric: true,
        width: '78px',
        sortValue: (row) => row.growth,
        render: (row) => <CategoryScore value={row.growth} label={`Growth score for ${row.ticker}`} />,
      },
      {
        key: 'momentum',
        header: 'Momentum',
        sortLabel: 'momentum score',
        numeric: true,
        width: '92px',
        sortValue: (row) => row.momentum,
        render: (row) => <CategoryScore value={row.momentum} label={`Momentum score for ${row.ticker}`} />,
      },
      {
        key: 'valuation',
        header: 'Valuation',
        sortLabel: 'valuation score',
        numeric: true,
        width: '88px',
        sortValue: (row) => row.valuation,
        render: (row) => <CategoryScore value={row.valuation} label={`Valuation score for ${row.ticker}`} />,
      },
      {
        key: 'valuation_class',
        header: 'Fair-value view',
        title: "The API's classification of the price against its fair-value range.",
        width: '162px',
        render: (row) => {
          // The API supplies a compact form for this column and the full
          // wording for the detail view; the client picks, never shortens.
          const label =
            row.valuation_label_short ??
            row.valuation_label ??
            (row.valuation_class ? fmt.humanize(row.valuation_class) : null);
          return label ? (
            <Pill tone={valuationTone(row.valuation_class, row.valuation_label)}>{label}</Pill>
          ) : (
            <MissingValue status="missing" label="Fair-value view" />
          );
        },
      },
      {
        key: 'upside',
        header: 'Upside',
        title: 'Upside to the fair-value midpoint, as the API computed it.',
        sortLabel: 'upside to the fair-value midpoint',
        numeric: true,
        width: '86px',
        sortValue: (row) => row.upside,
        render: (row) => (
          <ValueOrMissing
            value={row.upside}
            format={fmt.signedPct}
            status="missing"
            label={`Upside for ${row.ticker}`}
            detail={row.valuation_label ? `The API's reading is: ${row.valuation_label}.` : null}
          />
        ),
      },
      {
        key: 'confidence',
        header: 'Confidence',
        title: 'How much the API trusts this valuation. Shown beside every score, never behind it.',
        sortLabel: 'confidence',
        numeric: true,
        width: '96px',
        sortValue: (row) => row.confidence,
        render: (row) => (
          <div className="cell-stack">
            {/* Same weight as the score numeral: confidence is never the junior partner. */}
            <span className="tnum strong">
              <ValueOrMissing
                value={row.confidence}
                format={fmt.pct}
                status="missing"
                label={`Confidence for ${row.ticker}`}
              />
            </span>
            {row.confidence_label ? <span className="cell-sub">{row.confidence_label}</span> : null}
          </div>
        ),
      },
      {
        key: 'risk_score',
        header: 'Risk',
        title: 'Risk score, 0-100, from the API. Its reading is on the stock detail.',
        sortLabel: 'risk score',
        numeric: true,
        width: '74px',
        sortValue: (row) => row.risk_score,
        render: (row) => (
          <ValueOrMissing
            value={row.risk_score}
            format={fmt.score}
            status="missing"
            label={`Risk score for ${row.ticker}`}
          />
        ),
      },
      {
        key: 'score_change_1w',
        header: '1w',
        title: 'Change in the composite score over the past week, in points.',
        sortLabel: '1w score change',
        numeric: true,
        width: '78px',
        sortValue: (row) => row.score_change_1w,
        render: (row) => <ScoreDelta value={row.score_change_1w} windowLabel="1 week" />,
      },
      {
        key: 'score_change_1m',
        header: '1m',
        title: 'Change in the composite score over the past month, in points.',
        sortLabel: '1m score change',
        numeric: true,
        width: '78px',
        sortValue: (row) => row.score_change_1m,
        render: (row) => <ScoreDelta value={row.score_change_1m} windowLabel="1 month" />,
      },
      {
        key: 'days_to_earnings',
        header: 'To earnings',
        title: 'Days until the next scheduled earnings date.',
        sortLabel: 'days to earnings',
        numeric: true,
        width: '88px',
        defaultDirection: 'asc',
        sortValue: (row) => row.days_to_earnings,
        render: (row) => (
          <ValueOrMissing
            value={row.days_to_earnings}
            format={(value) => `${fmt.int(value)}d`}
            status="missing"
            label={`Days to earnings for ${row.ticker}`}
          />
        ),
      },
    ],
    [bandOrder],
  );

  return (
    <DataTable<OpportunityRow>
      className="opp-table"
      columns={columns}
      rows={rows}
      rowKey={(row) => row.company_id}
      ariaLabel="Ranked opportunities"
      sort={sort}
      onSortChange={onSortChange}
      defaultSort={DEFAULT_OPPORTUNITY_SORT}
      onRowActivate={(row) => onOpen(row.ticker)}
      rowLabel={(row) => `Open ${row.ticker}, ${row.name}`}
      isRowActive={(row) => Boolean(activeTicker) && row.ticker === activeTicker}
      sticky
      empty={empty}
    />
  );
}
