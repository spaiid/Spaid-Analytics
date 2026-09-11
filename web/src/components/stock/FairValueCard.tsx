/**
 * FairValueCard — the range is the headline, the midpoint is a footnote.
 *
 * A fair value is never presented as one precise number: the primary visual is
 * the bear/base/bull range with the traded price inside it, and every number is
 * shown next to the confidence the API assigned to it.
 */

import type { FairValue, MethodValue, ScenarioValue, Sensitivity } from '../../api/types';
import { RangeBar } from '../charts';
import {
  Card,
  DataTable,
  EmptyState,
  MissingValue,
  Pill,
  StatTile,
  ValueOrMissing,
  WarningIcon,
  type Column,
} from '../ui';
import { DASH, currency, humanize, pct, signedPct } from '../../lib/format';
import { confidenceText } from './metricFormat';

function SwingBar({ value, max }: { value: number | null; max: number }) {
  const known = typeof value === 'number' && Number.isFinite(value);
  const fraction = known && max > 0 ? Math.min(1, Math.abs(value) / max) : 0;
  return (
    <div className="sd-swing">
      <span className="sd-swing-val">{known ? pct(Math.abs(value), 1) : DASH}</span>
      <span className="sd-swing-track" aria-hidden="true">
        {known && <span className="sd-swing-fill" style={{ width: `${Math.max(2, fraction * 100)}%` }} />}
      </span>
    </div>
  );
}

function methodColumns(usedWeight: number): Array<Column<MethodValue>> {
  return [
    {
      key: 'label',
      header: 'Method',
      render: (row) => (
        <span className={row.used ? 'strong' : undefined}>
          {row.label}
          {!row.used && <span className="sr-only"> (skipped)</span>}
        </span>
      ),
    },
    {
      key: 'used',
      header: 'Used',
      render: (row) =>
        row.used ? (
          <Pill tone="good">Used</Pill>
        ) : (
          <Pill tone="neutral">Skipped</Pill>
        ),
    },
    {
      key: 'weight',
      header: 'Weight',
      numeric: true,
      title: 'The weight the valuation spec declares for this method.',
      render: (row) => (row.used ? pct(row.weight, 0) : <span className="muted">{DASH}</span>),
    },
    {
      key: 'effective',
      header: 'Effective',
      numeric: true,
      title:
        'Declared weight divided by the weights of the methods that actually ran — the share this method really had in the blend.',
      render: (row) =>
        row.used && usedWeight > 0 ? (
          <span className="tnum">{pct(row.weight / usedWeight, 0)}</span>
        ) : (
          <span className="muted">{DASH}</span>
        ),
    },
    {
      key: 'value',
      header: 'Value per share',
      numeric: true,
      render: (row) => (
        <ValueOrMissing
          value={row.value_per_share}
          format={(value) => currency(value)}
          label={row.label}
          status={row.used ? 'missing' : 'not_applicable'}
          detail={row.reason ?? 'This method produced no value for this company.'}
        />
      ),
    },
    {
      key: 'reason',
      header: 'Reason',
      render: (row) => <span className="muted">{row.reason ?? (row.used ? DASH : 'No reason given.')}</span>,
    },
  ];
}

function scenarioColumns(): Array<Column<ScenarioValue>> {
  return [
    {
      key: 'label',
      header: 'Scenario',
      render: (row) => (
        <span>
          <span className="strong">{humanize(row.label)}</span>
          {row.summary && <div className="sd-sub">{row.summary}</div>}
        </span>
      ),
    },
    {
      key: 'value',
      header: 'Value per share',
      numeric: true,
      render: (row) => currency(row.value_per_share),
    },
    { key: 'probability', header: 'Probability', numeric: true, render: (row) => pct(row.probability, 0) },
    {
      key: 'discount',
      header: 'Discount rate',
      numeric: true,
      render: (row) => (
        <ValueOrMissing
          value={row.discount_rate}
          format={(value) => pct(value, 1)}
          label="Discount rate"
          status="not_applicable"
          detail="This scenario was not built from a discounted cash-flow model."
        />
      ),
    },
    {
      key: 'terminal',
      header: 'Terminal growth',
      numeric: true,
      render: (row) => (
        <ValueOrMissing
          value={row.terminal_growth}
          format={(value) => pct(value, 1)}
          label="Terminal growth"
          status="not_applicable"
          detail="This scenario was not built from a discounted cash-flow model."
        />
      ),
    },
  ];
}

function sensitivityColumns(maxSwing: number): Array<Column<Sensitivity>> {
  return [
    {
      key: 'assumption',
      header: 'Assumption',
      render: (row) => (
        <span>
          <span className="strong">{row.assumption}</span>
          {row.description && <div className="sd-sub">{row.description}</div>}
        </span>
      ),
    },
    { key: 'low', header: 'Low', numeric: true, render: (row) => currency(row.low_value) },
    { key: 'base', header: 'Base', numeric: true, render: (row) => currency(row.base_value) },
    { key: 'high', header: 'High', numeric: true, render: (row) => currency(row.high_value) },
    {
      key: 'swing',
      header: 'Swing',
      numeric: true,
      width: '170px',
      title: 'How far the fair value moves when this assumption moves, as a share of the base value.',
      render: (row) => <SwingBar value={row.swing_pct} max={maxSwing} />,
    },
  ];
}

export interface FairValueCardProps {
  ticker: string;
  fairValue: FairValue | null;
}

export function FairValueCard({ ticker, fairValue }: FairValueCardProps) {
  if (!fairValue) {
    return (
      <Card title="Fair value" subtitle="Range, methods, scenarios and what would make it wrong.">
        <EmptyState
          title="No fair value estimate for this company"
          message="The valuation stage has not produced an estimate for this ticker. That is an absence, not a verdict: it says nothing about whether the shares are cheap."
        />
      </Card>
    );
  }

  const skipped = fairValue.methods.filter((method) => !method.used);
  const used = fairValue.methods.filter((method) => method.used);
  const usedWeight = used.reduce((total, method) => total + method.weight, 0);
  const swings = fairValue.sensitivities
    .map((row) => row.swing_pct)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
    .map(Math.abs);
  const maxSwing = swings.length > 0 ? Math.max(...swings) : 0;

  const rangeLabel = [
    `Fair value range for ${ticker}.`,
    `Bear ${fairValue.bear === null ? DASH : currency(fairValue.bear)},`,
    `base ${fairValue.base === null ? DASH : currency(fairValue.base)},`,
    `bull ${fairValue.bull === null ? DASH : currency(fairValue.bull)}.`,
    `Overall range ${fairValue.range_low === null ? DASH : currency(fairValue.range_low)} to ${
      fairValue.range_high === null ? DASH : currency(fairValue.range_high)
    },`,
    `fair value ${fairValue.midpoint === null ? DASH : currency(fairValue.midpoint)}.`,
    `The traded price is ${currency(fairValue.price)}.`,
  ].join(' ');

  return (
    <Card
      title="Fair value"
      subtitle="A range, not a number. The estimate is the blend of the methods that ran; the width of the range and the confidence beside it matter more."
    >
      <RangeBar
        ariaLabel={rangeLabel}
        price={fairValue.price}
        bear={fairValue.bear}
        base={fairValue.base}
        bull={fairValue.bull}
        rangeLow={fairValue.range_low}
        rangeHigh={fairValue.range_high}
        midpoint={fairValue.midpoint}
        subtitle={`Business model: ${fairValue.business_model}. Spec ${fairValue.spec_version}.`}
      />

      <div className="sd-fv-facts">
        <StatTile
          label="Upside to fair value"
          hint="The distance from today's price to the weighted blend of the methods that ran. Not the centre of the range, which sits higher because value compounds. The range matters more than this single number."
          value={
            <ValueOrMissing
              value={fairValue.upside}
              format={(value) => signedPct(value, 1)}
              label="Upside to fair value"
              status="missing"
              detail="No fair value was produced, so there is no distance to report."
            />
          }
          sub={`Fair value ${fairValue.midpoint === null ? DASH : currency(fairValue.midpoint)} · price ${currency(
            fairValue.price,
          )}`}
        />
        <StatTile
          label="Classification"
          value={
            <Pill tone="neutral" size="lg" icon={false}>
              {fairValue.classification_label}
            </Pill>
          }
          sub="Decided by the valuation engine, not by this interface."
        />
        <StatTile
          label="Valuation confidence"
          value={
            <ValueOrMissing
              value={fairValue.confidence}
              format={(value) => confidenceText(value)}
              label="Valuation confidence"
              status="missing"
              detail="The valuation engine reported no confidence for this estimate."
            />
          }
          sub={fairValue.confidence_label ?? 'No confidence label'}
          footer={
            fairValue.method_dispersion === null ? (
              used.length === 1 ? (
                'Only one method ran, so nothing cross-checks it'
              ) : (
                'Method dispersion not reported'
              )
            ) : (
              <>Method dispersion {pct(fairValue.method_dispersion, 0)} — how far the methods disagree</>
            )
          }
        />
      </div>

      <hr className="hr" />

      <h3 className="sd-section-title">Methods</h3>
      <DataTable<MethodValue>
        columns={methodColumns(usedWeight)}
        rows={fairValue.methods}
        rowKey={(row) => row.method}
        ariaLabel={`Valuation methods for ${ticker}`}
        rowClassName={(row) => (row.used ? undefined : 'is-muted-row')}
        empty={<span className="muted">No methods were recorded for this estimate.</span>}
      />
      <p className="sd-note">
        {used.length} of {fairValue.methods.length} methods ran. The weight column is the weight the valuation spec
        declares; the blend divides by the {pct(usedWeight, 0)} that actually ran, so each surviving method&rsquo;s
        real share is the effective column.{' '}
        {skipped.length > 0
          ? `The ${skipped.length} skipped ${
              skipped.length === 1 ? 'method was dropped from the blend' : 'methods were dropped from the blend'
            } — never averaged in as a zero, which would have dragged the estimate down.`
          : 'Nothing was skipped, so the declared weights and the effective weights differ only by rounding.'}
      </p>

      <hr className="hr" />

      <h3 className="sd-section-title">Scenarios</h3>
      <DataTable<ScenarioValue>
        columns={scenarioColumns()}
        rows={fairValue.scenarios}
        rowKey={(row) => row.label}
        ariaLabel={`Valuation scenarios for ${ticker}`}
        empty={<span className="muted">No scenarios were recorded for this estimate.</span>}
      />

      <hr className="hr" />

      <h3 className="sd-section-title">Sensitivity, most to least impactful</h3>
      <DataTable<Sensitivity>
        columns={sensitivityColumns(maxSwing)}
        rows={fairValue.sensitivities}
        rowKey={(row) => row.assumption}
        ariaLabel={`Fair value sensitivity to each assumption for ${ticker}, ordered by impact`}
        empty={<span className="muted">No sensitivity analysis was recorded for this estimate.</span>}
      />
      <p className="sd-note">
        Bars are scaled against the largest swing in this table, so they compare assumptions with each other — not
        with any other company.
      </p>

      <hr className="hr" />

      <h3 className="sd-section-title">Why this estimate may be wrong</h3>
      {fairValue.caveats.length === 0 ? (
        <p className="sd-empty-line">
          The valuation engine listed no caveats. That is not the same as certainty: the range above, and the
          confidence beside it, still carry the doubt.
        </p>
      ) : (
        <ul className="sd-caveats">
          {fairValue.caveats.map((caveat) => (
            <li className="sd-caveat" key={caveat}>
              <WarningIcon size={13} className="sd-caveat-icon" />
              <span>{caveat}</span>
            </li>
          ))}
        </ul>
      )}
      {fairValue.methods.length === 0 && fairValue.scenarios.length === 0 && (
        <p className="sd-note">
          <MissingValue status="missing" label="Method detail" detail="No method or scenario detail was stored with this estimate." />{' '}
          The range above is all the engine recorded.
        </p>
      )}
    </Card>
  );
}
