/**
 * CategoryCard — one of the four categories, expandable into its metrics.
 *
 * Collapsed it answers "how much did this category move the composite"; opened
 * it shows every metric behind that number, split so that "not applicable"
 * can never be mistaken for "missing".
 */

import { useId } from 'react';
import type { CategoryDetail, MetricDetail } from '../../api/types';
import { AssessmentPill, ChevronDownIcon, ChevronRightIcon, Meter, ValueOrMissing } from '../ui';
import { DASH, num, pct, score as fmtScore } from '../../lib/format';
import { MetricTable } from './MetricTable';

function partition(metrics: MetricDetail[]): {
  scored: MetricDetail[];
  absent: MetricDetail[];
  notApplicable: MetricDetail[];
} {
  const scored: MetricDetail[] = [];
  const absent: MetricDetail[] = [];
  const notApplicable: MetricDetail[] = [];
  for (const metric of metrics) {
    if (metric.status === 'scored') scored.push(metric);
    else if (metric.status === 'not_applicable') notApplicable.push(metric);
    else absent.push(metric);
  }
  return { scored, absent, notApplicable };
}

export interface CategoryCardProps {
  category: CategoryDetail;
  ticker: string;
  open: boolean;
  onToggle: () => void;
}

export function CategoryCard({ category, ticker, open, onToggle }: CategoryCardProps) {
  const panelId = useId();
  const { scored, absent, notApplicable } = partition(category.metrics);

  const contributionSum = scored.reduce(
    (total, metric) => (metric.contribution === null ? total : total + metric.contribution),
    0,
  );

  const Chevron = open ? ChevronDownIcon : ChevronRightIcon;

  return (
    <section className={['sd-cat', open ? 'is-open' : null].filter(Boolean).join(' ')}>
      {/*
        The header is a <div>, not a <button>: it carries a role="meter" bar and
        an assessment pill, and neither is valid inside a button's phrasing
        content — nesting them there would also swallow both into the button's
        accessible name. The toggle below is the only interactive element.
      */}
      <div className="sd-cat-head">
        <div className="sd-cat-top">
          <span className="sd-cat-label">{category.label}</span>
          <span className="tag">{pct(category.weight, 0)} of composite</span>
          <span className="sd-cat-score">
            <ValueOrMissing
              value={category.score}
              format={fmtScore}
              label={`${category.label} score`}
              status="missing"
              detail="This category did not score, so its weight was dropped and the other categories were renormalised."
            />
          </span>
        </div>

        <Meter
          value={category.score}
          label={`${category.label} score for ${ticker}`}
          valueText={category.score === null ? undefined : `${fmtScore(category.score)} of 100`}
          size="lg"
        />

        <div className="sd-cat-facts">
          <span>
            Contributes {category.contribution === null ? DASH : num(category.contribution, 1)} pts
          </span>
          <span>Coverage {pct(category.coverage, 0)}</span>
          <span>
            {category.n_metrics} metrics · {category.n_missing} without data
          </span>
        </div>

        {category.assessment && <AssessmentPill assessment={category.assessment} />}

        <button
          type="button"
          className="sd-cat-toggle"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={onToggle}
        >
          <Chevron size={12} />
          <span>
            {open ? 'Hide' : 'Show'} the {category.label.toLowerCase()} metrics behind this score
          </span>
        </button>
      </div>

      <div className="sd-cat-body" id={panelId} hidden={!open}>
        <div className="sd-metric-group">
          <h4 className="sd-section-title">Scored metrics ({scored.length})</h4>
          <MetricTable metrics={scored} ariaLabel={`Scored ${category.label} metrics for ${ticker}`} />
          {scored.length > 0 ? (
            <div className="sd-sum" role="note">
              <span>Sum of the contributions above</span>
              <span className="sd-sum-eq">{num(contributionSum, 1)}</span>
              <span>= the {category.label} score the API reported</span>
              <span className="sd-sum-eq">{category.score === null ? DASH : fmtScore(category.score)}</span>
              <span className="muted">
                (the weights of the scored metrics are renormalised to 100% of the category)
              </span>
            </div>
          ) : (
            <p className="sd-group-note">
              No metric in this category scored, so there is no arithmetic to show and the category was dropped
              from the composite.
            </p>
          )}
        </div>

        {absent.length > 0 && (
          <div className="sd-metric-group">
            <h4 className="sd-section-title">No data ({absent.length})</h4>
            <p className="sd-group-note">
              These metrics were expected for this company but no value arrived. They were left out of the score and
              the remaining weights were renormalised — they were not scored as zero.
            </p>
            <MetricTable metrics={absent} ariaLabel={`${category.label} metrics with no data for ${ticker}`} />
          </div>
        )}

        {notApplicable.length > 0 && (
          <div className="sd-metric-group is-na">
            <h4 className="sd-section-title">Not applicable ({notApplicable.length})</h4>
            <p className="sd-group-note">
              These measures do not apply to this kind of business, so they were never expected. Unlike the group
              above, nothing is missing here.
            </p>
            <MetricTable metrics={notApplicable} ariaLabel={`${category.label} metrics that do not apply to ${ticker}`} />
          </div>
        )}
      </div>
    </section>
  );
}
