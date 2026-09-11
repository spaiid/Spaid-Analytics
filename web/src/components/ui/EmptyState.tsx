import type { ReactNode } from 'react';

export interface EmptyStateAction {
  label: string;
  onClick: () => void;
  /** Disable while a pipeline run is already in flight. */
  disabled?: boolean;
}

export interface EmptyStateProps {
  /** What is empty, e.g. "No scored names yet". */
  title: ReactNode;
  /** Why it is empty and what to do about it. */
  message?: ReactNode;
  action?: EmptyStateAction;
  /** Extra content, e.g. a <code> hint with the CLI command. */
  children?: ReactNode;
  className?: string;
}

/**
 * A genuinely empty store. This must never be used for a failed request —
 * see ErrorState, which looks and reads differently on purpose.
 */
export function EmptyState({ title, message, action, children, className }: EmptyStateProps) {
  return (
    <div className={['empty', className].filter(Boolean).join(' ')}>
      <div className="empty-title">{title}</div>
      {message !== undefined && <div>{message}</div>}
      {children}
      {action && (
        <button type="button" className="btn btn-primary" onClick={action.onClick} disabled={action.disabled}>
          {action.label}
        </button>
      )}
    </div>
  );
}
