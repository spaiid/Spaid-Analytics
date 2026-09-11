/**
 * RiskPanel — the risk measures behind the risk score.
 *
 * The score's reading comes from the API's own assessment; this panel only
 * lays the numbers out and says which way the scale runs.
 */

import type { Level, RiskDetail } from '../../api/types';
import { AssessmentPill, Card, EmptyState, Meter, ValueOrMissing, type MeterTone } from '../ui';
import { compactCurrency, int, num, pct, score as fmtScore } from '../../lib/format';

/** The API's level, narrowed to the tones a meter can paint. */
function meterTone(level: Level | null | undefined): MeterTone {
  switch (level) {
    case 'good':
      return 'good';
    case 'warning':
      return 'warning';
    case 'critical':
      return 'critical';
    default:
      return 'neutral';
  }
}

interface Row {
  term: string;
  value: number | null;
  format: (value: number) => string;
  hint?: string;
}

export interface RiskPanelProps {
  risk: RiskDetail | null;
  ticker: string;
}

export function RiskPanel({ risk, ticker }: RiskPanelProps) {
  if (!risk) {
    return (
      <Card title="Risk" headingLevel={3}>
        <EmptyState
          title="No risk measures"
          message="The risk stage has not produced measures for this ticker. Missing measures are not a low-risk reading."
        />
      </Card>
    );
  }

  const rows: Row[] = [
    { term: 'Volatility, annualised', value: risk.volatility, format: (value) => pct(value, 1) },
    { term: 'Beta', value: risk.beta, format: (value) => num(value, 2) },
    { term: 'Downside beta', value: risk.downside_beta, format: (value) => num(value, 2) },
    { term: 'Downside deviation', value: risk.downside_deviation, format: (value) => pct(value, 1) },
    { term: 'Max drawdown, 1 year', value: risk.max_drawdown_1y, format: (value) => pct(value, 1) },
    { term: 'Max drawdown, 5 years', value: risk.max_drawdown_5y, format: (value) => pct(value, 1) },
    {
      term: 'Median dollar volume',
      value: risk.dollar_volume_median,
      format: (value) => compactCurrency(value),
    },
    { term: 'Estimated spread', value: risk.spread_bps, format: (value) => `${num(value, 1)} bps` },
    { term: 'Days to earnings', value: risk.days_to_earnings, format: (value) => int(value) },
    { term: 'Altman Z-score', value: risk.distress_score, format: (value) => num(value, 2) },
    { term: 'Piotroski F-score', value: risk.piotroski, format: (value) => `${int(value)} of 9` },
  ];

  return (
    <Card title="Risk" headingLevel={3} subtitle="Higher risk score means more risk, not a worse company.">
      <div className="row" style={{ marginBottom: 12 }}>
        <span className="stat-value is-small tnum">
          <ValueOrMissing value={risk.risk_score} format={fmtScore} label="Risk score" />
        </span>
        <span className="muted t-small">{risk.risk_label ?? 'Not labelled'}</span>
      </div>
      <Meter
        value={risk.risk_score}
        label={`Risk score for ${ticker}, higher is riskier`}
        valueText={risk.risk_score === null ? undefined : `${fmtScore(risk.risk_score)} of 100, ${risk.risk_label ?? 'not labelled'}`}
        tone={meterTone(risk.assessment ? risk.assessment.level : null)}
        size="lg"
      />
      {risk.assessment && (
        <p style={{ marginTop: 10 }}>
          <AssessmentPill assessment={risk.assessment} />
        </p>
      )}

      <dl className="sd-kv" style={{ marginTop: 14 }}>
        {rows.map((row) => (
          <div key={row.term} style={{ display: 'contents' }}>
            <dt>{row.term}</dt>
            <dd>
              <ValueOrMissing value={row.value} format={row.format} label={row.term} />
            </dd>
          </div>
        ))}
        <div style={{ display: 'contents' }}>
          <dt>Distress reading</dt>
          <dd>{risk.distress_label ?? 'Not assessed'}</dd>
        </div>
      </dl>
    </Card>
  );
}
