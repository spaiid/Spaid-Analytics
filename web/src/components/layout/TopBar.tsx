/**
 * TopBar — where the app says how old its data is and what it is doing.
 *
 * It holds the one live region for pipeline progress: the visible strip is
 * `aria-hidden` and a screen-reader sentence next to it carries the same words,
 * so the announcement is a sentence rather than "Fetching prices · 40%".
 * Nothing else in the shell announces the running state, so it is never read
 * out twice.
 */

import { ClockIcon, MoonIcon, RefreshIcon, Skeleton, SunIcon, Tooltip } from '../ui';
import { dash, date, humanize, pct } from '../../lib/format';
import type { Theme } from '../../lib/theme';
import type { PipelineController } from '../../hooks/usePipeline';

export interface TopBarProps {
  /** The current view's name — the page's <h1>. */
  title: string;
  asOf: string | null;
  specVersion: string | null;
  /** True while /api/status has never resolved. */
  loading: boolean;
  /** Set when /api/status failed; the banner below carries the detail. */
  failed: boolean;
  theme: Theme;
  onToggleTheme: () => void;
  pipeline: PipelineController;
  /** Focus target for the skip link. */
  onSkipToContent: () => void;
}

function stageText(pipeline: PipelineController): string {
  const stage = pipeline.stage === null || pipeline.stage.trim() === '' ? 'Working' : humanize(pipeline.stage);
  const progress =
    typeof pipeline.progress === 'number' && Number.isFinite(pipeline.progress)
      ? ` · ${pct(pipeline.progress, 0)}`
      : '';
  return `${stage}${progress}`;
}

function spokenStatus(pipeline: PipelineController): string {
  if (pipeline.starting && !pipeline.running) return 'Starting the pipeline.';
  if (pipeline.running) return `Pipeline running. ${stageText(pipeline).replace(' · ', ', ')}.`;
  if (pipeline.finishedAt !== null && pipeline.succeeded) return 'Pipeline run finished.';
  // A failed run is announced by the alert banner in the content area, so it is
  // deliberately not repeated here.
  return '';
}

export function TopBar({
  title,
  asOf,
  specVersion,
  loading,
  failed,
  theme,
  onToggleTheme,
  pipeline,
  onSkipToContent,
}: TopBarProps) {
  const busy = pipeline.running || pipeline.starting;
  const nextTheme: Theme = theme === 'dark' ? 'light' : 'dark';

  return (
    <header className="topbar">
      <button type="button" className="skip-link" onClick={onSkipToContent}>
        Skip to main content
      </button>

      <h1>{title}</h1>

      <div className="topbar-spacer" />

      {loading ? (
        <Skeleton width={116} height={12} />
      ) : failed ? (
        <span className="t-small muted nowrap">As of {dash(null)}</span>
      ) : (
        <span className="row-tight t-small secondary nowrap">
          <ClockIcon size={12} />
          <Tooltip content="The date of the newest inputs behind every number on screen.">
            <span>As of {date(asOf)}</span>
          </Tooltip>
        </span>
      )}

      <span className="mono muted t-micro nowrap" title="Active scoring specification version">
        spec {dash(specVersion)}
      </span>

      {/* Visible strip + its spoken twin. Only the twin is a live region. */}
      <span className="row-tight nowrap" style={{ minWidth: 0 }}>
        {busy && <span className="spinner" aria-hidden="true" />}
        {busy && (
          <span className="t-small secondary truncate" aria-hidden="true" style={{ maxWidth: 220 }}>
            {pipeline.starting && !pipeline.running ? 'Starting' : stageText(pipeline)}
          </span>
        )}
        <span className="sr-only" role="status" aria-live="polite" aria-atomic="true">
          {spokenStatus(pipeline)}
        </span>
      </span>

      <button
        type="button"
        className="btn btn-sm btn-primary"
        onClick={pipeline.start}
        aria-disabled={busy || undefined}
      >
        <RefreshIcon size={12} />
        <span>{busy ? 'Running…' : 'Run pipeline'}</span>
      </button>

      <button
        type="button"
        className="btn btn-sm"
        onClick={onToggleTheme}
        aria-label={`Switch to the ${nextTheme} theme`}
      >
        {theme === 'dark' ? <SunIcon size={12} /> : <MoonIcon size={12} />}
        <span>{nextTheme === 'light' ? 'Light' : 'Dark'}</span>
      </button>
    </header>
  );
}
