/**
 * PlaceholderView — a route that exists but is not built yet.
 *
 * This is deliberately NOT an empty state and NOT an error state: nothing here
 * failed and nothing is missing from the store. It says what will live here, so
 * an unfinished view can never be mistaken for a broken or empty one.
 */

import { Link } from 'react-router-dom';
import { Card, Pill } from '../components/ui';

export interface PlaceholderViewProps {
  title: string;
  /** One sentence: what this view is for. */
  description: string;
  /** The things that will be on this page, in the order they will appear. */
  contents?: string[];
  /** What to use in the meantime. */
  insteadHref?: string;
  insteadLabel?: string;
}

export function PlaceholderView({
  title,
  description,
  contents = [],
  insteadHref = '/opportunities',
  insteadLabel = 'Opportunities',
}: PlaceholderViewProps) {
  return (
    <Card
      title={title}
      actions={<Pill tone="info">Not built yet</Pill>}
      subtitle="This view is planned for a later milestone. Nothing has failed and no data is missing."
    >
      <div className="stack">
        <p className="secondary" style={{ maxWidth: '68ch' }}>
          {description}
        </p>

        {contents.length > 0 && (
          <div>
            <h3 className="expand-title">What will live here</h3>
            <ul className="stack" style={{ gap: 7 }}>
              {contents.map((item) => (
                <li key={item} className="t-body secondary" style={{ display: 'flex', gap: 9 }}>
                  <span aria-hidden="true" className="muted">
                    •
                  </span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <p className="note">
          In the meantime, <Link to={insteadHref} style={{ textDecoration: 'underline' }}>{insteadLabel}</Link> is
          the built view closest to this one.
        </p>
      </div>
    </Card>
  );
}

export interface NotFoundViewProps {
  path: string;
}

/** A URL that matches no route. Distinct from both "empty" and "failed". */
export function NotFoundView({ path }: NotFoundViewProps) {
  return (
    <Card title="No such view" actions={<Pill tone="neutral">Unknown route</Pill>}>
      <div className="stack">
        <p className="secondary">
          Nothing is routed at <code className="mono">{path}</code>. The address may be mistyped, or it may point at
          a view that has since moved.
        </p>
        <p>
          <Link to="/opportunities" className="btn btn-sm">
            Go to Opportunities
          </Link>
        </p>
      </div>
    </Card>
  );
}
