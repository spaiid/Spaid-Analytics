/**
 * PeerTable — the comparison set the percentiles were measured against.
 *
 * The subject company is marked with a label as well as a background tint, and
 * every row is keyboard operable so the peer set is navigable without a mouse.
 */

import type { PeerRow } from '../../api/types';
import { useNavigate } from '../../lib/router';
import { Card, DataTable, EmptyState, Meter, Pill, ValueOrMissing, type Column } from '../ui';
import { compactCurrency, multiple, pct, score as fmtScore } from '../../lib/format';

function scoreCell(value: number | null, label: string) {
  return (
    <div className="score-cell">
      <span className="score-num">
        <ValueOrMissing value={value} format={fmtScore} label={label} />
      </span>
      <Meter value={value} label={label} />
    </div>
  );
}

function columns(): Array<Column<PeerRow>> {
  return [
    {
      key: 'ticker',
      header: 'Company',
      render: (peer) => (
        <span className="ticker-cell">
          <span className="ticker-sym">{peer.ticker}</span>
          <span className="ticker-name">{peer.name}</span>
          {peer.is_self && <Pill tone="info">This company</Pill>}
        </span>
      ),
      sortValue: (peer) => peer.ticker,
      sortLabel: 'ticker',
    },
    {
      key: 'score',
      header: 'Score',
      numeric: true,
      width: '150px',
      render: (peer) => scoreCell(peer.score, `Composite score for ${peer.ticker}`),
      sortValue: (peer) => peer.score,
      sortLabel: 'composite score',
    },
    {
      key: 'quality',
      header: 'Quality',
      numeric: true,
      render: (peer) => <ValueOrMissing value={peer.quality} format={fmtScore} label={`Quality score for ${peer.ticker}`} />,
      sortValue: (peer) => peer.quality,
      sortLabel: 'quality score',
    },
    {
      key: 'growth',
      header: 'Growth',
      numeric: true,
      render: (peer) => <ValueOrMissing value={peer.growth} format={fmtScore} label={`Growth score for ${peer.ticker}`} />,
      sortValue: (peer) => peer.growth,
      sortLabel: 'growth score',
    },
    {
      key: 'momentum',
      header: 'Momentum',
      numeric: true,
      render: (peer) => (
        <ValueOrMissing value={peer.momentum} format={fmtScore} label={`Momentum score for ${peer.ticker}`} />
      ),
      sortValue: (peer) => peer.momentum,
      sortLabel: 'momentum score',
    },
    {
      key: 'valuation',
      header: 'Valuation',
      numeric: true,
      render: (peer) => (
        <ValueOrMissing value={peer.valuation} format={fmtScore} label={`Valuation score for ${peer.ticker}`} />
      ),
      sortValue: (peer) => peer.valuation,
      sortLabel: 'valuation score',
    },
    {
      key: 'ev_ebit',
      header: 'EV/EBIT',
      numeric: true,
      render: (peer) => <ValueOrMissing value={peer.ev_ebit} format={(value) => multiple(value, 1)} label="EV to EBIT" />,
      sortValue: (peer) => peer.ev_ebit,
      sortLabel: 'EV to EBIT',
    },
    {
      key: 'fcf_yield',
      header: 'FCF yield',
      numeric: true,
      render: (peer) => <ValueOrMissing value={peer.fcf_yield} format={(value) => pct(value, 1)} label="Free cash flow yield" />,
      sortValue: (peer) => peer.fcf_yield,
      sortLabel: 'free cash flow yield',
    },
    {
      key: 'revenue_growth_1y',
      header: 'Revenue 1y',
      numeric: true,
      render: (peer) => (
        <ValueOrMissing value={peer.revenue_growth_1y} format={(value) => pct(value, 1)} label="Revenue growth, 1 year" />
      ),
      sortValue: (peer) => peer.revenue_growth_1y,
      sortLabel: 'revenue growth over 1 year',
    },
    {
      key: 'operating_margin',
      header: 'Op margin',
      numeric: true,
      render: (peer) => (
        <ValueOrMissing value={peer.operating_margin} format={(value) => pct(value, 1)} label="Operating margin" />
      ),
      sortValue: (peer) => peer.operating_margin,
      sortLabel: 'operating margin',
    },
    {
      key: 'market_cap',
      header: 'Market cap',
      numeric: true,
      render: (peer) => (
        <ValueOrMissing value={peer.market_cap} format={(value) => compactCurrency(value)} label="Market cap" />
      ),
      sortValue: (peer) => peer.market_cap,
      sortLabel: 'market cap',
    },
  ];
}

export interface PeerTableProps {
  peers: PeerRow[];
  ticker: string;
}

export function PeerTable({ peers, ticker }: PeerTableProps) {
  const navigate = useNavigate();

  if (peers.length === 0) {
    return (
      <Card title="Peers" headingLevel={3}>
        <EmptyState
          title="No peer set"
          message="No comparable companies were recorded for this ticker, so the percentiles above rest on a peer group this view cannot show."
        />
      </Card>
    );
  }

  return (
    <Card
      title="Peers"
      headingLevel={3}
      subtitle="The comparison set behind the percentiles. Select a row to open that company."
      flush
    >
      <DataTable<PeerRow>
        columns={columns()}
        rows={peers}
        rowKey={(peer) => peer.company_id}
        ariaLabel={`Peer comparison for ${ticker}`}
        onRowActivate={(peer) => navigate({ name: 'stock', ticker: peer.ticker })}
        rowLabel={(peer) => (peer.is_self ? `${peer.ticker}, this company` : `Open ${peer.ticker}, ${peer.name}`)}
        rowClassName={(peer) => (peer.is_self ? 'is-self' : undefined)}
        defaultSort={{ key: 'score', direction: 'desc' }}
        sticky
      />
    </Card>
  );
}
