/**
 * StockDetailView — the decision view for one company.
 *
 * Order: identity and score, then the fair-value range, then the complete
 * explanation of how the score was reached, then risk / traps / confidence,
 * then the evidence, the charts, the peer set and the sources.
 *
 * Rules this view holds to:
 *  - it computes no verdict: every band, classification, assessment and label
 *    is rendered exactly as the API sent it;
 *  - missing, not-applicable and thin-peer values look different from each
 *    other and none of them looks like a zero;
 *  - a failed fetch is never drawn as an empty store;
 *  - confidence is shown wherever a score or a fair value is shown.
 */

import { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

import type { StockDetail } from '../api/types';
import { apiPaths, useApi } from '../api/client';
import { hrefFor } from '../lib/router';
import { Card, ChevronRightIcon, Skeleton, SkeletonTable, StaleBanner } from '../components/ui';
import {
  ConfidencePanel,
  EvidenceLists,
  FairValueCard,
  FinancialTrends,
  PeerTable,
  PriceHistoryCard,
  RiskPanel,
  ScoreBreakdown,
  SourcesTable,
  StockHeader,
  StockLoadError,
  ValueTrapPanel,
} from '../components/stock';
import '../components/stock/stock.css';

function BackLink() {
  return (
    <Link className="sd-back" to={hrefFor({ name: 'opportunities' })}>
      <ChevronRightIcon size={11} style={{ transform: 'rotate(180deg)' }} />
      <span>Back to Opportunities</span>
    </Link>
  );
}

function LoadingState({ ticker }: { ticker: string }) {
  return (
    <div className="stack">
      <Card>
        <div className="sd-head">
          <div className="sd-ident">
            <Skeleton width="140px" height="34px" />
            <Skeleton width="220px" height="15px" />
            <Skeleton width="260px" height="18px" />
          </div>
          <div className="sd-tiles">
            <Skeleton height="96px" radius="var(--radius)" />
            <Skeleton height="96px" radius="var(--radius)" />
            <Skeleton height="96px" radius="var(--radius)" />
            <Skeleton height="96px" radius="var(--radius)" />
          </div>
        </div>
      </Card>
      <Card title="Fair value">
        <Skeleton height="132px" radius="var(--radius)" />
      </Card>
      <Card title="How the score was calculated">
        <SkeletonTable rows={6} columns={6} label={`Loading the score breakdown for ${ticker}`} />
      </Card>
    </div>
  );
}

export interface StockDetailViewProps {
  /** Uppercase ticker from the route. */
  ticker: string;
  /** Changes when a pipeline run finishes; re-reads this company. */
  refreshToken?: number | null;
}

export function StockDetailView({ ticker, refreshToken = null }: StockDetailViewProps) {
  const result = useApi<StockDetail>(apiPaths.stock(ticker));

  // Re-read after a pipeline run rewrites the store (same contract as the
  // other views: the shell passes the run's finish timestamp).
  const reloadRef = useRef(result.reload);
  reloadRef.current = result.reload;
  const seenTokenRef = useRef(refreshToken);
  useEffect(() => {
    if (refreshToken === null || refreshToken === seenTokenRef.current) return;
    seenTokenRef.current = refreshToken;
    reloadRef.current();
  }, [refreshToken]);

  if (result.state === 'loading') {
    return (
      <>
        <p className="sr-only" aria-live="polite">
          Loading {ticker}.
        </p>
        <LoadingState ticker={ticker} />
      </>
    );
  }

  if (result.state === 'error') {
    return (
      <div className="stack">
        <p className="sr-only" aria-live="polite">
          {ticker} could not be loaded. {result.message}
        </p>
        <BackLink />
        <StockLoadError
          ticker={ticker}
          message={result.message}
          path={apiPaths.stock(ticker)}
          onRetry={result.reload}
        />
      </div>
    );
  }

  const detail = result.data;

  return (
    <div className={['stack', result.refreshing ? 'is-refreshing' : null].filter(Boolean).join(' ')}>
      <p className="sr-only" aria-live="polite">
        Loaded {detail.ticker}, {detail.name}.
      </p>

      <BackLink />

      <StaleBanner freshness={detail.freshness} onRefresh={result.reload} refreshLabel="Reload" />

      <Card>
        <StockHeader detail={detail} />
      </Card>

      <FairValueCard ticker={detail.ticker} fairValue={detail.fair_value} />

      <ScoreBreakdown
        ticker={detail.ticker}
        categories={detail.categories}
        composite={detail.score}
        coverage={detail.coverage}
      />

      <div className="sd-panels">
        <RiskPanel risk={detail.risk} ticker={detail.ticker} />
        <ValueTrapPanel valueTrap={detail.value_trap} ticker={detail.ticker} />
        <ConfidencePanel confidence={detail.confidence} ticker={detail.ticker} />
      </div>

      <EvidenceLists detail={detail} />

      <PriceHistoryCard ticker={detail.ticker} priceHistory={detail.price_history} />

      <FinancialTrends ticker={detail.ticker} trends={detail.financial_trends} />

      <PeerTable peers={detail.peers} ticker={detail.ticker} />

      <SourcesTable
        sources={detail.sources}
        freshness={detail.freshness}
        ticker={detail.ticker}
        onRefresh={result.reload}
      />
    </div>
  );
}

export default StockDetailView;
