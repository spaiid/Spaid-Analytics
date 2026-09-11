/**
 * StockLoadError — a failed request for one company.
 *
 * Deliberately unlike an empty state: an alert role, the server's own message,
 * a retry, and a way back to the ranked list. Nothing here implies the company
 * is unscored — only that this request did not succeed.
 */

import { Link } from 'react-router-dom';

import { hrefFor } from '../../lib/router';
import { Card, ErrorState } from '../ui';

export interface StockLoadErrorProps {
  ticker: string;
  /** The API's `detail`, or the transport failure. Shown verbatim. */
  message: string;
  /** The path that failed, for the detail line. */
  path?: string;
  onRetry?: () => void;
}

export function StockLoadError({ ticker, message, path, onRetry }: StockLoadErrorProps) {
  return (
    <Card>
      <ErrorState
        title={`Could not load ${ticker}`}
        message={message}
        detail={path ?? null}
        onRetry={onRetry}
      />
      <div className="sd-error-actions">
        <Link className="btn" to={hrefFor({ name: 'opportunities' })}>
          Back to Opportunities
        </Link>
      </div>
      <p className="sd-note">
        This is a failed request, not an empty store: nothing here says the company is unscored. If the ticker is
        not in the current universe, the ranked list will not contain it either.
      </p>
    </Card>
  );
}
