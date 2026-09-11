/**
 * EvidenceLists — strengths, weaknesses, risks and catalysts.
 *
 * Four labelled lists, each with its own icon as well as its own colour, and an
 * explicit line when a list is empty so "nothing listed" cannot read as
 * "nothing to worry about".
 */

import type { ReactNode } from 'react';
import type { StockDetail } from '../../api/types';
import { Card, GoodIcon, InfoIcon, SeriousIcon, WarningIcon } from '../ui';

type Tone = 'good' | 'warning' | 'serious' | 'info';

const ICON: Record<Tone, typeof GoodIcon> = {
  good: GoodIcon,
  warning: WarningIcon,
  serious: SeriousIcon,
  info: InfoIcon,
};

interface EvidenceListProps {
  title: string;
  subtitle: ReactNode;
  items: string[];
  tone: Tone;
  emptyText: string;
}

function EvidenceList({ title, subtitle, items, tone, emptyText }: EvidenceListProps) {
  const Icon = ICON[tone];
  return (
    <Card title={title} subtitle={subtitle} headingLevel={3}>
      {items.length === 0 ? (
        <p className="sd-empty-line">{emptyText}</p>
      ) : (
        <ul className={`sd-list sd-list-${tone}`}>
          {items.map((item) => (
            <li key={item}>
              <Icon size={13} className="sd-list-icon" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

export interface EvidenceListsProps {
  detail: StockDetail;
}

export function EvidenceLists({ detail }: EvidenceListsProps) {
  return (
    <div className="sd-evidence">
      <EvidenceList
        title="Strengths"
        subtitle="Where this company ranks well against its peers."
        items={detail.strengths}
        tone="good"
        emptyText="No standout strengths were recorded for this company."
      />
      <EvidenceList
        title="Weaknesses"
        subtitle="Where it ranks poorly against its peers."
        items={detail.weaknesses}
        tone="warning"
        emptyText="No standout weaknesses were recorded. Absence of a listed weakness is not evidence of strength."
      />
      <EvidenceList
        title="Risks"
        subtitle="What the risk and distress measures flagged."
        items={detail.risks}
        tone="serious"
        emptyText="No risks were written up. The risk panel above still carries the measures."
      />
      <EvidenceList
        title="Catalysts"
        subtitle="Dated or scheduled events the pipeline knows about."
        items={detail.catalysts}
        tone="info"
        emptyText="No catalysts were recorded for this company."
      />
    </div>
  );
}
