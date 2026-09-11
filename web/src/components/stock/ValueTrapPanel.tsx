/**
 * ValueTrapPanel — every trap signal, fired or not.
 *
 * Signals that did not fire are listed too: a short list of fired signals means
 * something only when the signals that stayed quiet are visible beside them.
 */

import type { TrapSignal, ValueTrap } from '../../api/types';
import { Card, DataTable, EmptyState, Pill, ValueOrMissing, type Column } from '../ui';
import { num, score as fmtScore } from '../../lib/format';

function columns(): Array<Column<TrapSignal>> {
  return [
    {
      key: 'label',
      header: 'Signal',
      render: (signal) => (
        <span>
          <span className={signal.fired ? 'strong' : undefined}>{signal.label}</span>
          {signal.detail && <div className="sd-sub">{signal.detail}</div>}
        </span>
      ),
      sortValue: (signal) => signal.label,
      sortLabel: 'signal name',
    },
    {
      key: 'fired',
      header: 'Status',
      render: (signal) =>
        signal.fired ? <Pill tone="warning">Fired</Pill> : <Pill tone="neutral">Did not fire</Pill>,
      sortValue: (signal) => (signal.fired ? 1 : 0),
      sortLabel: 'whether the signal fired',
    },
    {
      key: 'value',
      header: 'Value',
      numeric: true,
      render: (signal) => (
        <ValueOrMissing
          value={signal.value}
          format={(value) => num(value, 2)}
          label={signal.label}
          status="missing"
          detail="No value was recorded for this signal."
        />
      ),
      sortValue: (signal) => signal.value,
      sortLabel: 'signal value',
    },
  ];
}

export interface ValueTrapPanelProps {
  valueTrap: ValueTrap | null;
  ticker: string;
}

export function ValueTrapPanel({ valueTrap, ticker }: ValueTrapPanelProps) {
  if (!valueTrap) {
    return (
      <Card title="Value trap" headingLevel={3}>
        <EmptyState
          title="No value-trap assessment"
          message="The risk stage has not assessed this ticker for value-trap signals. No assessment is not a clean bill of health."
        />
      </Card>
    );
  }

  return (
    <Card
      title="Value trap"
      headingLevel={3}
      subtitle="Cheap for a reason, or cheap and fine? These are the signals the engine checked."
    >
      <div className="row" style={{ marginBottom: 12 }}>
        <Pill tone="neutral" size="lg" icon={false}>
          {valueTrap.classification_label}
        </Pill>
        <span className="muted t-small">
          {valueTrap.n_fired} of {valueTrap.signals.length} signals fired
        </span>
      </div>
      <dl className="sd-kv" style={{ marginBottom: 14 }}>
        <div style={{ display: 'contents' }}>
          <dt>Trap score</dt>
          <dd>
            <ValueOrMissing
              value={valueTrap.trap_score}
              format={fmtScore}
              label="Trap score"
              status="missing"
              detail="No trap score was recorded for this company."
            />
          </dd>
        </div>
      </dl>
      <DataTable<TrapSignal>
        columns={columns()}
        rows={valueTrap.signals}
        rowKey={(signal) => signal.key}
        ariaLabel={`Value-trap signals for ${ticker}`}
        empty={<span className="muted">No signals were recorded.</span>}
      />
    </Card>
  );
}
