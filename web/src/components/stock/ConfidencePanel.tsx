/**
 * ConfidencePanel — what the confidence number is made of.
 *
 * Confidence is never a decoration on the score: its components and its
 * caveats are shown in full, with the weights the API used.
 */

import type { Confidence, ConfidenceComponent } from '../../api/types';
import { Card, DataTable, EmptyState, Meter, Pill, QuestionIcon, type Column } from '../ui';
import { pct } from '../../lib/format';
import { confidenceFraction, confidenceText } from './metricFormat';

function columns(): Array<Column<ConfidenceComponent>> {
  return [
    {
      key: 'label',
      header: 'Component',
      render: (component) => (
        <span>
          <span className="strong">{component.label}</span>
          {component.detail && <div className="sd-sub">{component.detail}</div>}
        </span>
      ),
      sortValue: (component) => component.label,
      sortLabel: 'component name',
    },
    {
      key: 'score',
      header: 'Score',
      numeric: true,
      width: '150px',
      render: (component) => (
        <div className="score-cell">
          <span className="score-num">{confidenceText(component.score)}</span>
          <Meter
            value={confidenceFraction(component.score)}
            min={0}
            max={1}
            label={`${component.label} confidence component`}
            valueText={confidenceText(component.score)}
          />
        </div>
      ),
      sortValue: (component) => component.score,
      sortLabel: 'component score',
    },
    {
      key: 'weight',
      header: 'Weight',
      numeric: true,
      render: (component) => <span className="tnum">{pct(component.weight, 0)}</span>,
      sortValue: (component) => component.weight,
      sortLabel: 'component weight',
    },
  ];
}

export interface ConfidencePanelProps {
  confidence: Confidence | null;
  ticker: string;
}

export function ConfidencePanel({ confidence, ticker }: ConfidencePanelProps) {
  if (!confidence) {
    return (
      <Card title="Confidence" headingLevel={3}>
        <EmptyState
          title="No confidence assessment"
          message="Without a confidence reading, treat the score above as unqualified — not as certain."
        />
      </Card>
    );
  }

  return (
    <Card
      title="Confidence"
      headingLevel={3}
      subtitle="How much weight the score deserves. Read it next to the score, never after it."
    >
      <div className="row" style={{ marginBottom: 10 }}>
        <span className="stat-value is-small tnum">{confidenceText(confidence.score)}</span>
        <Pill tone="neutral" icon={false}>
          {confidence.label}
        </Pill>
      </div>
      <Meter
        value={confidenceFraction(confidence.score)}
        min={0}
        max={1}
        label={`Confidence in the score for ${ticker}`}
        valueText={`${confidenceText(confidence.score)}, ${confidence.label}`}
        size="lg"
      />

      <div style={{ marginTop: 14 }}>
        <DataTable<ConfidenceComponent>
          columns={columns()}
          rows={confidence.components}
          rowKey={(component) => component.key}
          ariaLabel={`Confidence components for ${ticker}`}
          empty={<span className="muted">No components were recorded.</span>}
        />
      </div>

      <h4 className="sd-section-title" style={{ marginTop: 16 }}>
        What could still be wrong
      </h4>
      {confidence.caveats.length === 0 ? (
        <p className="sd-empty-line">No caveats were recorded. That is not a guarantee of accuracy.</p>
      ) : (
        <ul className="sd-list sd-list-info">
          {confidence.caveats.map((caveat) => (
            <li key={caveat}>
              <QuestionIcon size={13} className="sd-list-icon" />
              <span>{caveat}</span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
