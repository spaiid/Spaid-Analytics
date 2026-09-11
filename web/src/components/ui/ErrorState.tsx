import type { ReactNode } from 'react';
import { CriticalIcon, RefreshIcon } from './Icons';

export interface ErrorStateProps {
  /** Defaults to "Could not load this". */
  title?: ReactNode;
  /** The server's `detail`, or the transport failure. Always shown. */
  message: string;
  /** Optional extra context, e.g. the path that failed. */
  detail?: string | null;
  onRetry?: () => void;
  retryLabel?: string;
  /** Inline variant for a small panel. */
  compact?: boolean;
  className?: string;
}

/**
 * A failed fetch. Distinguishable from emptiness by border, icon, wording and
 * the retry action — never reuse EmptyState for an error.
 */
export function ErrorState({
  title = 'Could not load this',
  message,
  detail,
  onRetry,
  retryLabel = 'Retry',
  compact = false,
  className,
}: ErrorStateProps) {
  return (
    <div
      className={['error-state', compact ? 'is-compact' : null, className].filter(Boolean).join(' ')}
      role="alert"
      style={compact ? { padding: '14px 16px' } : undefined}
    >
      <div className="error-title">
        <CriticalIcon size={14} style={{ color: 'var(--critical)' }} />
        <span>{title}</span>
      </div>
      <p className="error-message">{message}</p>
      {detail ? <p className="error-detail">{detail}</p> : null}
      {onRetry && (
        <button type="button" className="btn" onClick={onRetry}>
          <RefreshIcon size={12} />
          <span>{retryLabel}</span>
        </button>
      )}
    </div>
  );
}
