import type { ReactNode } from 'react';
import { useId } from 'react';

export interface CardProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  /** Right-aligned controls in the header row (chart/table toggle, links). */
  actions?: ReactNode;
  children: ReactNode;
  /** Remove the padding so a table can bleed to the card edge. */
  flush?: boolean;
  headingLevel?: 2 | 3 | 4;
  className?: string;
  id?: string;
  /** Marks the whole card as a background refresh in progress. */
  busy?: boolean;
}

/**
 * A titled panel. When a title is given the card becomes a <section> labelled
 * by its heading, so screen-reader users can navigate by region.
 */
export function Card({
  title,
  subtitle,
  actions,
  children,
  flush = false,
  headingLevel = 2,
  className,
  id,
  busy = false,
}: CardProps) {
  const headingId = useId();
  const Heading = `h${headingLevel}` as 'h2' | 'h3' | 'h4';
  const classes = ['card', flush ? 'is-flush' : null, busy ? 'is-refreshing' : null, className]
    .filter(Boolean)
    .join(' ');

  return (
    <section
      className={classes}
      id={id}
      aria-labelledby={title === undefined ? undefined : headingId}
      aria-busy={busy || undefined}
    >
      {(title !== undefined || actions !== undefined) && (
        <div className="card-head">
          {title !== undefined && (
            <Heading className="card-title" id={headingId}>
              {title}
            </Heading>
          )}
          {actions !== undefined && <div className="card-actions">{actions}</div>}
        </div>
      )}
      {subtitle !== undefined && <p className="card-sub">{subtitle}</p>}
      <div className="card-body">{children}</div>
    </section>
  );
}
