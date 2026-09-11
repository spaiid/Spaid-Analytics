/**
 * ScoreBreakdown — the complete explanation of how the composite was reached.
 *
 * Four category cards, each expandable down to the individual metrics, and a
 * footer that shows the arithmetic: the contributions of the categories that
 * scored, divided by the weights that survived, equals the composite the API
 * reported.
 */

import { useState } from 'react';
import type { CategoryDetail } from '../../api/types';
import { Card, EmptyState } from '../ui';
import { DASH, num, pct, score as fmtScore } from '../../lib/format';
import { CategoryCard } from './CategoryCard';

export interface ScoreBreakdownProps {
  ticker: string;
  categories: CategoryDetail[];
  /** The composite the API reported, for the arithmetic footer. */
  composite: number | null;
  coverage: number | null;
}

export function ScoreBreakdown({ ticker, categories, composite, coverage }: ScoreBreakdownProps) {
  const [openKey, setOpenKey] = useState<string | null>(null);

  if (categories.length === 0) {
    return (
      <Card title="How the score was calculated">
        <EmptyState
          title="No category scores for this company"
          message="The scoring stage has not produced category scores for this ticker on the current as-of date."
        />
      </Card>
    );
  }

  const scoredCategories = categories.filter((category) => category.score !== null);
  const droppedCategories = categories.filter((category) => category.score === null);
  const contributionSum = scoredCategories.reduce(
    (total, category) => (category.contribution === null ? total : total + category.contribution),
    0,
  );
  const weightSum = scoredCategories.reduce((total, category) => total + category.weight, 0);
  // Shown so the arithmetic can be checked, never to replace the API's composite.
  const quotient = weightSum > 0 ? contributionSum / weightSum : null;

  return (
    <Card
      title="How the score was calculated"
      subtitle="Every number below comes from the API. Open a category to see each metric, its peer percentile, its weight and the points it contributed."
    >
      <div className="sd-cats">
        {categories.map((category) => (
          <CategoryCard
            key={category.key}
            ticker={ticker}
            category={category}
            open={openKey === category.key}
            onToggle={() => setOpenKey((current) => (current === category.key ? null : category.key))}
          />
        ))}
      </div>

      <div className="sd-sum" role="note" style={{ marginTop: 14 }}>
        <span>
          Sum of the {scoredCategories.length} scored {scoredCategories.length === 1 ? 'category' : 'categories'}
        </span>
        <span className="sd-sum-eq">{num(contributionSum, 1)} pts</span>
        <span>÷ their weights</span>
        <span className="sd-sum-eq">{pct(weightSum, 0)}</span>
        <span>=</span>
        <span className="sd-sum-eq">{quotient === null ? DASH : fmtScore(quotient)}</span>
        <span className="muted">· composite reported by the API</span>
        <span className="sd-sum-eq">{composite === null ? DASH : fmtScore(composite)}</span>
      </div>

      <p className="sd-note">
        {droppedCategories.length === 0
          ? 'Every category scored, so the published weights apply unchanged.'
          : `${droppedCategories
              .map((category) => `${category.label} (${pct(category.weight, 0)})`)
              .join(', ')} did not have enough data to score. ${
              droppedCategories.length === 1 ? 'Its weight was' : 'Their weights were'
            } dropped and the remaining categories were renormalised, which keeps the composite on the same 0-100 scale instead of dragging it toward zero.`}{' '}
        Metric coverage across the whole company is {pct(coverage, 0)}.
      </p>
    </Card>
  );
}
