import type { ReactNode } from 'react';
import type { DataFreshness } from '../../api/types';
import { ageHours, date } from '../../lib/format';
import { StatusIcon, type StatusTone } from './Icons';

export interface BannerProps {
  tone?: StatusTone;
  title?: ReactNode;
  children?: ReactNode;
  /** Buttons or links on the right. */
  actions?: ReactNode;
  onDismiss?: () => void;
  /**
   * Announce changes to assistive tech. Use for anything that updates in the
   * background (pipeline status, refresh outcome).
   */
  live?: boolean;
  /** `alert` for failures, `status` for everything else. */
  role?: 'status' | 'alert';
  className?: string;
}

/** A persistent strip of context: stale data, pipeline state, warnings. */
export function Banner({
  tone = 'info',
  title,
  children,
  actions,
  onDismiss,
  live = false,
  role,
  className,
}: BannerProps) {
  const resolvedRole = role ?? (tone === 'critical' ? 'alert' : 'status');
  return (
    <div
      className={['banner', `banner-${tone}`, className].filter(Boolean).join(' ')}
      role={live || resolvedRole === 'alert' ? resolvedRole : undefined}
      aria-live={live ? (resolvedRole === 'alert' ? 'assertive' : 'polite') : undefined}
    >
      <StatusIcon tone={tone} size={14} className="banner-icon" />
      <div className="banner-body">
        {title !== undefined && <div className="banner-title">{title}</div>}
        {children !== undefined && <div>{children}</div>}
      </div>
      {(actions !== undefined || onDismiss) && (
        <div className="banner-actions">
          {actions}
          {onDismiss && (
            <button type="button" className="btn btn-sm" onClick={onDismiss}>
              Dismiss
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export interface StaleBannerProps {
  freshness: DataFreshness | null | undefined;
  /** Usually a pipeline run or a reload. */
  onRefresh?: () => void;
  refreshLabel?: string;
  className?: string;
}

/**
 * Shown whenever the API flags the data as stale. It is never dismissible:
 * out-of-date numbers must stay visibly out of date.
 */
export function StaleBanner({ freshness, onRefresh, refreshLabel = 'Run pipeline', className }: StaleBannerProps) {
  if (!freshness || !freshness.is_stale) return null;

  const parts: string[] = [];
  if (freshness.as_of) parts.push(`Data is as of ${date(freshness.as_of)}.`);
  if (typeof freshness.age_hours === 'number' && Number.isFinite(freshness.age_hours)) {
    parts.push(`That is ${ageHours(freshness.age_hours)} old.`);
  }
  if (freshness.detail) parts.push(freshness.detail);

  return (
    <Banner
      tone="warning"
      title="Stale data"
      live
      className={className}
      actions={
        onRefresh ? (
          <button type="button" className="btn btn-sm" onClick={onRefresh}>
            {refreshLabel}
          </button>
        ) : undefined
      }
    >
      {parts.length > 0 ? parts.join(' ') : 'The store has not been refreshed recently.'}
    </Banner>
  );
}
