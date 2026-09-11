/**
 * OpportunitiesView — the ranked list.
 *
 * The API sends the whole ranked set in one response, so sector / band /
 * minimum-score / text filters are applied here in the browser and answer
 * instantly. What the browser never does is decide what a number means: the
 * band, the fair-value view and the confidence label are all server text.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiPaths, useApi } from '../api/client';
import type { OpportunitiesResponse, OpportunityRow } from '../api/types';
import * as fmt from '../lib/format';
import { useNavigate } from '../lib/router';
import { Card, EmptyState, ErrorState, Skeleton, SkeletonTable } from '../components/ui';
import {
  ALL,
  EMPTY_FILTERS,
  OpportunitiesTable,
  OpportunityCharts,
  OpportunityFilters,
  OpportunityHeader,
  applyFilters,
  bandOrderMap,
  countByBand,
  hasActiveFilters,
  useDebouncedValue,
  type OpportunityFilterState,
} from '../components/opportunities';
import '../components/opportunities/opportunities.css';

const NO_ROWS: OpportunityRow[] = [];
const NO_STRINGS: string[] = [];

function LoadingView() {
  return (
    <div className="stack">
      <Card title="Opportunities" busy>
        <div className="grid grid-4" aria-hidden="true">
          {[0, 1, 2, 3].map((index) => (
            <div className="stat" key={index}>
              <Skeleton height={11} width="52%" />
              <Skeleton height={26} width="68%" style={{ marginTop: 10 }} />
              <Skeleton height={11} width="80%" style={{ marginTop: 10 }} />
            </div>
          ))}
        </div>
      </Card>
      <Card>
        <SkeletonTable rows={10} columns={7} label="Loading the ranked list…" />
      </Card>
    </div>
  );
}

export interface OpportunitiesViewProps {
  /** Starts a pipeline run from the empty state. Supplied by the shell. */
  onRunPipeline?: () => void;
  /** True while a run is in flight, so the action cannot be double-fired. */
  pipelineBusy?: boolean;
  /** Changes when a pipeline run finishes; re-reads the ranked list. */
  refreshToken?: number | null;
}

export function OpportunitiesView({
  onRunPipeline,
  pipelineBusy = false,
  refreshToken = null,
}: OpportunitiesViewProps) {
  const navigate = useNavigate();
  const result = useApi<OpportunitiesResponse>(apiPaths.opportunities());
  const [filters, setFilters] = useState<OpportunityFilterState>(EMPTY_FILTERS);

  // Re-read after a pipeline run rewrites the store.
  const reloadRef = useRef(result.reload);
  reloadRef.current = result.reload;
  const seenTokenRef = useRef(refreshToken);
  useEffect(() => {
    if (refreshToken === null || refreshToken === seenTokenRef.current) return;
    seenTokenRef.current = refreshToken;
    reloadRef.current();
  }, [refreshToken]);

  // Only the *applied* search lags; the input itself stays instant.
  const settledSearch = useDebouncedValue(filters.search, 250);
  const applied = useMemo<OpportunityFilterState>(
    () => ({ ...filters, search: settledSearch }),
    [filters, settledSearch],
  );

  const data = result.state === 'data' ? result.data : null;
  const rows = data ? data.rows : NO_ROWS;
  const bands = data ? data.bands : NO_STRINGS;
  const sectors = data ? data.sectors : NO_STRINGS;

  const { rows: visible, unscoredHidden } = useMemo(
    () => applyFilters(rows, applied),
    [rows, applied],
  );
  const bandCounts = useMemo(() => countByBand(rows, bands), [rows, bands]);
  const bandOrder = useMemo(() => bandOrderMap(bands), [bands]);
  const showUnbanded = useMemo(() => rows.some((row) => !row.band), [rows]);

  const openStock = useCallback(
    (ticker: string) => navigate({ name: 'stock', ticker }),
    [navigate],
  );
  const selectBand = useCallback((band: string) => {
    setFilters((current) => ({ ...current, band }));
  }, []);
  const selectSector = useCallback((sector: string) => {
    setFilters((current) => ({ ...current, sector: current.sector === sector ? ALL : sector }));
  }, []);
  const reset = useCallback(() => setFilters(EMPTY_FILTERS), []);

  if (result.state === 'loading') return <LoadingView />;

  if (result.state === 'error') {
    return (
      <div className="stack">
        <Card title="Opportunities">
          <ErrorState
            title="Could not load the ranked list"
            message={result.message}
            detail={apiPaths.opportunities()}
            onRetry={result.reload}
          />
        </Card>
      </div>
    );
  }

  const response = result.data;
  const filtersActive = hasActiveFilters(applied);

  // A genuinely empty store — never the same state as a failed request above.
  if (rows.length === 0) {
    return (
      <div className="stack">
        <Card title="Opportunities">
          <EmptyState
            title={
              response.universe_size === 0
                ? 'Nothing in the store yet'
                : 'No scores computed yet'
            }
            message={
              response.universe_size === 0
                ? 'No companies have been ingested. Run the pipeline to fetch the universe and score it.'
                : `${fmt.int(response.universe_size)} companies are in the universe but none of them has been scored. Run the pipeline to compute scores.`
            }
            action={
              onRunPipeline
                ? { label: 'Run pipeline', onClick: onRunPipeline, disabled: pipelineBusy }
                : { label: 'Check again', onClick: result.reload, disabled: result.refreshing }
            }
          >
            <p className="muted t-small">
              {onRunPipeline ? (
                'Or run it from a terminal:'
              ) : (
                <>
                  Use <strong>Run pipeline</strong> in the top bar, or run it from a terminal:
                </>
              )}
            </p>
            <code>spaid run</code>
          </EmptyState>
        </Card>
      </div>
    );
  }

  return (
    <div className="stack">
      <OpportunityHeader
        data={response}
        shown={visible.length}
        bandCounts={bandCounts}
        selectedBand={applied.band}
        onSelectBand={selectBand}
      />

      <Card
        title="Ranked list"
        subtitle="Sorted by rank. Sort by any column header, open a row for the full explanation, and scroll the table sideways for risk, score changes and earnings."
        busy={result.refreshing}
      >
        <OpportunityFilters
          sectors={sectors}
          bands={bands}
          showUnbanded={showUnbanded}
          value={filters}
          onChange={setFilters}
          onReset={reset}
          shown={visible.length}
          total={rows.length}
          unscoredHidden={unscoredHidden}
        />

        {visible.length === 0 ? (
          <EmptyState
            title="No companies match these filters"
            message={
              filtersActive
                ? 'Every one of the ranked companies was filtered out. Widen a filter or clear them all.'
                : 'The ranked list came back empty.'
            }
            action={filtersActive ? { label: 'Clear filters', onClick: reset } : undefined}
          />
        ) : (
          <OpportunitiesTable rows={visible} bandOrder={bandOrder} onOpen={openStock} />
        )}

        {response.notes.length > 0 && (
          <div className="opp-notes">
            {response.notes.map((note, index) => (
              <p className="note" key={`${index}-${note}`}>
                {note}
              </p>
            ))}
          </div>
        )}
      </Card>

      <OpportunityCharts rows={visible} onSelectSector={selectSector} />
    </div>
  );
}
