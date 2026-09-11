/**
 * App.tsx — the application shell.
 *
 * Owns: the router, the sidebar/top-bar layout, the theme, the pipeline
 * control with its live region, and the two shell-level banners (stale data,
 * pipeline outcome). Views own their own data; the shell only reads
 * `/api/status` for the as-of date, the spec version and the universe counts.
 *
 * Routing uses real URL paths rather than hashes. The API serves this bundle
 * with a catch-all fallback, so /stock/NVDA reaches the server, returns
 * index.html and is then resolved by the client router — which keeps the URL
 * shareable and meaningful in server logs.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
  matchPath,
  useLocation,
  useNavigate,
  useParams,
} from 'react-router-dom';

import { apiPaths, useApi } from './api/client';
import type { StatusResponse } from './api/types';
import { Banner, StaleBanner } from './components/ui';
import { Sidebar } from './components/layout/Sidebar';
import { TopBar } from './components/layout/TopBar';
import { usePipeline } from './hooks/usePipeline';
import { dateTime, humanize } from './lib/format';
import { useTheme } from './lib/theme';
// Loaded after the four base stylesheets in main.tsx, so it settles ties.
import './styles/shell.css';

import { HealthView } from './views/HealthView';
import { NotFoundView, PlaceholderView } from './views/PlaceholderView';
import { OpportunitiesView } from './views/OpportunitiesView';
import { StockDetailView } from './views/StockDetailView';


/* ---------------------------------------------------------------- routing */

const TITLES: ReadonlyArray<readonly [string, string]> = [
  ['/opportunities', 'Opportunities'],
  ['/health', 'Data health'],
  ['/today', 'Today'],
  ['/portfolio', 'Portfolio'],
  ['/performance', 'Performance'],
  ['/journal', 'Research journal'],
];

function titleFor(pathname: string, ticker: string | null): string {
  if (ticker !== null) return `${ticker} · Stock detail`;
  if (pathname === '/' || pathname === '') return 'Opportunities';
  for (const [pattern, title] of TITLES) {
    if (matchPath(pattern, pathname)) return title;
  }
  return 'Not found';
}

function readTicker(pathname: string): string | null {
  const match = matchPath('/stock/:ticker', pathname);
  const raw = match === null ? undefined : match.params.ticker;
  if (raw === undefined || raw.trim() === '') return null;
  return decodeURIComponent(raw).trim().toUpperCase();
}

/** Reads the ticker out of the URL and hands it to the detail view. */
function StockRoute() {
  const params = useParams();
  const raw = params.ticker ?? '';
  const ticker = decodeURIComponent(raw).trim().toUpperCase();
  if (ticker === '') return <Navigate to="/opportunities" replace />;
  return <StockDetailView ticker={ticker} />;
}

function NotFoundRoute() {
  const location = useLocation();
  return <NotFoundView path={location.pathname} />;
}

/* ------------------------------------------------------------ progress bar */

function progressPercent(progress: number | null): number | null {
  if (typeof progress !== 'number' || !Number.isFinite(progress)) return null;
  return Math.max(0, Math.min(100, Math.round(progress * 100)));
}

interface PipelineProgressProps {
  progress: number | null;
  stage: string | null;
}

/**
 * Task progress, so `role="progressbar"` rather than the Meter primitive's
 * `role="meter"`. When the server reports no progress the bar is left
 * indeterminate instead of showing a misleading zero.
 */
function PipelineProgress({ progress, stage }: PipelineProgressProps) {
  const percent = progressPercent(progress);
  return (
    <span className="row-tight" style={{ minWidth: 0 }}>
      <span className="t-small secondary nowrap">
        {stage === null || stage.trim() === '' ? 'Working' : humanize(stage)}
      </span>
      <span
        className={percent === null ? 'meter is-empty' : 'meter'}
        role="progressbar"
        aria-label="Pipeline progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent === null ? undefined : percent}
        aria-valuetext={percent === null ? 'Progress not reported' : `${percent}%`}
        style={{ width: 160, display: 'inline-block' }}
      >
        {percent !== null && <span className="meter-fill" style={{ width: `${percent}%`, display: 'block' }} />}
      </span>
      <span className="t-small muted tnum nowrap">{percent === null ? '' : `${percent}%`}</span>
    </span>
  );
}

/* ------------------------------------------------------------------ shell */

function AppShell() {
  const { theme, toggle } = useTheme();
  const location = useLocation();
  const status = useApi<StatusResponse>(apiPaths.status());

  // A finished run rewrites the store, so re-read the shell's own status.
  const reloadStatusRef = useRef(status.reload);
  reloadStatusRef.current = status.reload;
  const onPipelineFinish = useCallback(() => {
    reloadStatusRef.current();
  }, []);
  const pipeline = usePipeline({ onFinish: onPipelineFinish });

  const routeTicker = readTicker(location.pathname);
  const [lastTicker, setLastTicker] = useState<string | null>(null);
  useEffect(() => {
    if (routeTicker !== null) setLastTicker(routeTicker);
  }, [routeTicker]);

  const title = titleFor(location.pathname, routeTicker);
  useEffect(() => {
    document.title = `${title} · Spaid Analytics`;
  }, [title]);

  const mainRef = useRef<HTMLElement>(null);
  const skipToContent = useCallback(() => {
    mainRef.current?.focus();
  }, []);

  const data = status.state === 'data' ? status.data : null;
  const busy = pipeline.running || pipeline.starting;
  const showFinished = !pipeline.running && pipeline.finishedAt !== null && pipeline.succeeded;

  return (
    <div className="app">
      <Sidebar
        lastTicker={lastTicker}
        asOf={data === null ? null : data.as_of}
        scoredCount={data === null ? null : data.scored_count}
        universeSize={data === null ? null : data.universe_size}
      />

      <div className="main">
        <TopBar
          title={title}
          asOf={data === null ? null : data.as_of}
          specVersion={data === null ? null : data.scoring_version}
          loading={status.state === 'loading'}
          failed={status.state === 'error'}
          theme={theme}
          onToggleTheme={toggle}
          pipeline={pipeline}
          onSkipToContent={skipToContent}
        />

        <main className="content stack" id="main" ref={mainRef} tabIndex={-1}>
          {status.state === 'error' && (
            <Banner
              tone="critical"
              title="Could not read the app status"
              actions={
                <button type="button" className="btn btn-sm" onClick={status.reload}>
                  Retry
                </button>
              }
            >
              {status.message} The as-of date and universe counts are unavailable; each view still loads its own data.
            </Banner>
          )}

          {/* Never dismissible: out-of-date numbers stay visibly out of date. */}
          <StaleBanner
            freshness={data === null ? null : data.freshness}
            onRefresh={busy ? undefined : pipeline.start}
            refreshLabel="Run pipeline"
          />

          {data !== null && data.message !== null && data.message.trim() !== '' && (
            <Banner tone="info" title="From the API">
              {data.message}
            </Banner>
          )}

          {busy && (
            <Banner tone="info" title="Pipeline running">
              <PipelineProgress
                progress={pipeline.progress}
                stage={pipeline.starting && !pipeline.running ? 'Starting' : pipeline.stage}
              />
            </Banner>
          )}

          {pipeline.error !== null && (
            <Banner tone="critical" title="The last pipeline run failed" onDismiss={pipeline.dismiss}>
              {pipeline.error}
            </Banner>
          )}

          {showFinished && (
            <Banner tone="good" title="Pipeline run finished" onDismiss={pipeline.dismiss}>
              Finished at {dateTime(new Date(pipeline.finishedAt ?? Date.now()))}. The views on this page have been
              re-read.
            </Banner>
          )}

          <Routes>
            <Route path="/" element={<Navigate to="/opportunities" replace />} />
            <Route path="/opportunities" element={<OpportunitiesView />} />
            <Route path="/stock" element={<Navigate to="/opportunities" replace />} />
            <Route path="/stock/:ticker" element={<StockRoute />} />
            <Route
              path="/health"
              element={
                <HealthView onRunPipeline={pipeline.start} pipelineBusy={busy} refreshToken={pipeline.finishedAt} />
              }
            />
            <Route
              path="/today"
              element={
                <PlaceholderView
                  title="Today"
                  description="The daily brief: what changed since the last run, which names crossed a band, and what needs a decision today."
                  contents={[
                    'Names whose composite score or band moved overnight, with the size of the move',
                    'Fair-value crossings: anything that moved into or out of its buy range',
                    'Earnings, filings and other dated events inside the next two weeks',
                    'Data that did not refresh, so you know what today’s numbers are missing',
                  ]}
                />
              }
            />
            <Route
              path="/portfolio"
              element={
                <PlaceholderView
                  title="Portfolio"
                  description="Current holdings, what each position is worth against its thesis, and the sizing rules behind it."
                  contents={[
                    'Positions with cost, weight, and current score and band',
                    'Exposure by sector, factor and business model',
                    'Sizing: the rule that set each weight, and the confidence behind it',
                    'Names that no longer meet the thesis they were bought on',
                  ]}
                />
              }
            />
            <Route
              path="/performance"
              element={
                <PlaceholderView
                  title="Performance"
                  description="How the decisions actually did: returns against the benchmark, and which parts of the process earned them."
                  contents={[
                    'Realised and unrealised return over each holding period',
                    'Attribution by category score, sector and entry band',
                    'Hit rate by band and by confidence bucket',
                    'The score as it stood at purchase against the outcome since',
                  ]}
                />
              }
            />
            <Route
              path="/journal"
              element={
                <PlaceholderView
                  title="Research journal"
                  description="A dated log of every thesis: the evidence at the time, what was expected, and how it turned out."
                  contents={[
                    'One entry per name, with the score, fair value and caveats captured on the day',
                    'The catalysts that were expected, and whether they arrived',
                    'Notes tied to the metric that prompted them',
                    'A review queue for theses whose evidence has since changed',
                  ]}
                />
              }
            />
            <Route path="*" element={<NotFoundRoute />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}

export default App;
