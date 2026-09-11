/**
 * The top of the validation page: what this result may claim, and why.
 *
 * Everything here is a sentence the API wrote. The client picks an icon and a
 * tone from the status word it was given; it never looks at a return, a Sharpe
 * ratio or a gate value and decides for itself that something has been
 * validated. That separation is the point of the whole page — the one failure
 * mode worth engineering against is a screen that says "validated" because the
 * number looked good.
 */

import type { ReactNode } from 'react';
import type {
  ConclusionKey,
  Gate,
  PeriodInfo,
  StrategySummary,
  UniverseStatus,
  ValidationStatusKey,
  Verdict,
} from '../../api/validation-types';
import { Card, DataTable, Pill, StatTile, type Column, type PillTone } from '../ui';
import { date, dateTime, humanize, int, num, pct } from '../../lib/format';

/* --------------------------------------------------------------- statuses */

interface StatusCopy {
  tone: PillTone;
  meaning: string;
}

const STATUS: Record<ValidationStatusKey, StatusCopy> = {
  not_testable: {
    tone: 'critical',
    meaning: 'A precondition on the data is unmet, so no backtest was run at all.',
  },
  exploratory_only: {
    tone: 'warning',
    meaning:
      'The run completed, but the data behind it cannot support a claim about an edge. Read it as a diagnostic, never as performance.',
  },
  in_sample: {
    tone: 'info',
    meaning:
      'Measured over the development period, which was visible while the system was built. In-sample by construction.',
  },
  validation_passed: { tone: 'good', meaning: 'Cleared every predefined threshold out of sample.' },
  validation_failed: { tone: 'serious', meaning: 'Did not clear the thresholds set before the test.' },
  holdout_passed: { tone: 'good', meaning: 'Cleared the thresholds on the sealed period.' },
  holdout_failed: { tone: 'critical', meaning: 'Did not clear the thresholds on the sealed period.' },
  paper_tracking: { tone: 'info', meaning: 'Recommendations are being recorded forward, not simulated.' },
  live_tracking: { tone: 'info', meaning: 'Tracking real positions.' },
};

const CONCLUSION_TONE: Record<ConclusionKey, PillTone> = {
  evidence_of_predictive_ability: 'good',
  no_meaningful_evidence: 'serious',
  evidence_is_mixed: 'warning',
  insufficient_data: 'neutral',
  backtest_invalid: 'critical',
};

function statusCopy(status: ValidationStatusKey): StatusCopy {
  return STATUS[status] ?? { tone: 'neutral', meaning: 'Status reported by the API.' };
}

/* ------------------------------------------------------------------ parts */

export function VerdictPanel({ verdict, periodLabel }: { verdict: Verdict; periodLabel: string }) {
  const status = statusCopy(verdict.status);
  return (
    <Card
      title="Validation conclusion"
      subtitle={`Frozen Strategy Version 1 over the ${humanize(periodLabel).toLowerCase()} period.`}
      actions={
        <span className="row-tight">
          <Pill tone={status.tone}>{verdict.status_label}</Pill>
          <Pill tone={CONCLUSION_TONE[verdict.conclusion] ?? 'neutral'}>{verdict.conclusion_label}</Pill>
        </span>
      }
    >
      <div className="stack">
        <p className="t-body strong" style={{ maxWidth: '78ch' }}>
          {verdict.headline}
        </p>
        <p className="note" style={{ maxWidth: '78ch' }}>
          {status.meaning}
        </p>
        {verdict.reasoning.length > 0 && (
          <ul className="stack" style={{ gap: 7 }}>
            {verdict.reasoning.map((line) => (
              <li key={line} className="t-body secondary" style={{ display: 'flex', gap: 9 }}>
                <span aria-hidden="true" className="muted">
                  •
                </span>
                <span>{line}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}

function GateTable({ gates, title, subtitle }: { gates: Gate[]; title: string; subtitle: ReactNode }) {
  const columns: Array<Column<Gate>> = [
    {
      key: 'label',
      header: 'Requirement',
      width: '30%',
      sortValue: (row) => row.label,
      sortLabel: 'requirement',
      render: (row) => (
        <span style={{ display: 'inline-flex', flexDirection: 'column', gap: 2 }}>
          <span className="strong">{row.label}</span>
          <span className="t-micro muted mono">{row.key}</span>
        </span>
      ),
    },
    {
      key: 'passed',
      header: 'Met',
      sortValue: (row) => (row.passed ? 1 : 0),
      sortLabel: 'whether the requirement is met',
      render: (row) => <Pill tone={row.passed ? 'good' : 'critical'}>{row.passed ? 'Met' : 'Not met'}</Pill>,
    },
    {
      key: 'value',
      header: 'Measured',
      numeric: true,
      sortValue: (row) => row.value,
      sortLabel: 'measured value',
      render: (row) =>
        row.value === null ? (
          <span className="muted">—</span>
        ) : (
          <span className="tnum">
            {num(row.value, 4)}
            {row.threshold !== null && <span className="t-micro muted"> / {num(row.threshold, 4)}</span>}
          </span>
        ),
    },
    {
      key: 'detail',
      header: 'What was found',
      width: '46%',
      render: (row) => (
        <span className="t-small secondary" style={{ display: 'block', maxWidth: '62ch' }}>
          {row.detail}
          {row.requirement !== '' && (
            <>
              <br />
              <span className="t-micro muted">{row.requirement}</span>
            </>
          )}
        </span>
      ),
    },
  ];
  return (
    <Card title={title} subtitle={subtitle} flush>
      <DataTable
        columns={columns}
        rows={gates}
        rowKey={(row) => row.key}
        ariaLabel={title}
        defaultSort={{ key: 'passed', direction: 'asc' }}
      />
    </Card>
  );
}

export function DataGates({ gates }: { gates: Gate[] }) {
  return (
    <GateTable
      gates={gates}
      title="Data gates"
      subtitle={
        <>
          Preconditions on the evidence, not on the result. While any of these is unmet, no run may be labelled
          anything stronger than exploratory — however good the numbers look.
        </>
      }
    />
  );
}

export function ResultGates({ gates }: { gates: Gate[] }) {
  return (
    <GateTable
      gates={gates}
      title="Result gates"
      subtitle={
        <>
          Thresholds fixed in <code className="mono">spaid/config/validation.py</code> before the first test ran. They
          are not adjustable after the fact, which is what makes passing one mean something.
        </>
      }
    />
  );
}

export function LimitationsPanel({ verdict }: { verdict: Verdict }) {
  if (verdict.limitations.length === 0) {
    return (
      <Card title="Limitations">
        <div className="row">
          <Pill tone="good">None outstanding</Pill>
          <span className="secondary t-body">Every declared data gate is satisfied for this run.</span>
        </div>
      </Card>
    );
  }
  return (
    <Card
      title="Limitations and failed gates"
      subtitle="Stated here rather than discovered later. Each one caps what this result is allowed to claim."
      actions={<Pill tone="warning">{`${verdict.limitations.length} outstanding`}</Pill>}
    >
      <ul className="stack" style={{ gap: 10 }}>
        {verdict.limitations.map((line) => (
          <li key={line} className="row" style={{ alignItems: 'flex-start', flexWrap: 'nowrap' }}>
            <Pill tone="warning">Limit</Pill>
            <span className="t-body secondary" style={{ minWidth: 0, maxWidth: '74ch' }}>
              {line}
            </span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

/* ------------------------------------------------------------- strategy */

export function StrategyPanel({
  strategy,
  dataVersion,
  universeVersion,
}: {
  strategy: StrategySummary;
  dataVersion: string | null;
  universeVersion: string | null;
}) {
  const unavailable = Object.entries(strategy.unavailable_point_in_time);
  return (
    <Card
      title="Strategy version"
      subtitle="Frozen before testing. The checksum covers the weights, the eligibility rules, the execution assumptions and the cost model, so a change to any of them makes these results stop applying."
      actions={<Pill tone={strategy.status === 'frozen' ? 'good' : 'warning'}>{humanize(strategy.status)}</Pill>}
    >
      <div className="stack">
        <div className="grid grid-4">
          <StatTile label="Version" value={strategy.version} sub={strategy.scoring_version} small />
          <StatTile
            label="Config checksum"
            value={strategy.config_checksum.slice(0, 12)}
            sub="of the frozen configuration"
            small
          />
          <StatTile label="Code commit" value={strategy.code_commit ?? '—'} sub="that produced the run" small />
          <StatTile
            label="Frozen"
            value={strategy.frozen_at === null ? '—' : date(strategy.frozen_at)}
            sub={strategy.frozen_at === null ? 'not recorded' : dateTime(strategy.frozen_at)}
            small
          />
        </div>

        <div className="grid grid-3">
          <StatTile label="Data version" value={dataVersion ?? '—'} sub="checksum of the tables read" small />
          <StatTile label="Universe version" value={universeVersion ?? '—'} sub="membership reconstruction" small />
          <StatTile
            label="Metrics"
            value={`${int(strategy.n_metrics_point_in_time)} of ${int(strategy.n_metrics)}`}
            sub="computable at a historical date"
            small
          />
        </div>

        {unavailable.length > 0 && (
          <div>
            <h3 className="expand-title">Metrics unavailable point-in-time</h3>
            <p className="note" style={{ maxWidth: '78ch' }}>
              These depend on analyst estimates, which the source publishes only as a current snapshot. At a historical
              date they are absent, and the scoring spec drops them and renormalises the surviving weights — the same
              rule that handles a bank with no gross margin. Back-filling today's values instead would hand the
              backtest years of foresight.
            </p>
            <ul className="stack" style={{ gap: 6, marginTop: 8 }}>
              {unavailable.map(([key, reason]) => (
                <li key={key} className="t-small secondary" style={{ display: 'flex', gap: 9 }}>
                  <span className="mono strong" style={{ minWidth: 170 }}>
                    {key}
                  </span>
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div>
          <h3 className="expand-title">Category weight observed point-in-time</h3>
          <div className="grid grid-4" style={{ marginTop: 8 }}>
            {Object.entries(strategy.category_coverage_point_in_time).map(([category, coverage]) => (
              <StatTile
                key={category}
                label={humanize(category)}
                value={pct(coverage, 0)}
                sub={`weight ${pct(strategy.category_weights[category] ?? 0, 0)} in the composite`}
                small
              />
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}

/* -------------------------------------------------------------- universe */

export function UniversePanel({ universe }: { universe: UniverseStatus }) {
  const tone: PillTone =
    universe.survivorship_status === 'removed'
      ? 'good'
      : universe.survivorship_status === 'partially removed'
        ? 'warning'
        : 'critical';

  return (
    <Card
      title="Universe and survivorship"
      subtitle="Who the ranking could choose from on each historical date, and how much of that population we can actually price."
      actions={<Pill tone={tone}>{`Survivorship bias: ${universe.survivorship_status}`}</Pill>}
    >
      <div className="stack">
        <p className="t-body secondary" style={{ maxWidth: '80ch' }}>
          {universe.survivorship_detail}
        </p>

        {universe.built ? (
          <>
            <div className="grid grid-4">
              <StatTile label="Securities seen" value={int(universe.securities)} sub="ever in the index" small />
              <StatTile label="Membership spells" value={int(universe.spells)} sub="entry-to-exit periods" small />
              <StatTile label="Observed exits" value={int(universe.exits_observed)} sub="companies dropped" small />
              <StatTile
                label="Monthly snapshots"
                value={int(universe.snapshots)}
                sub={`${universe.coverage_start ?? '—'} to ${universe.coverage_end ?? '—'}`}
                small
              />
            </div>
            <div className="grid grid-3">
              <StatTile
                label="Removed and priced"
                value={int(universe.removed_with_prices)}
                sub={`of ${int(universe.removed_securities)} removed`}
                small
              />
              <StatTile
                label="Removed, no prices"
                value={int(universe.removed_without_prices)}
                sub="acquired, merged or failed"
                small
              />
              <StatTile
                label="Delisting returns known"
                value={int(universe.delisting_returns_known)}
                sub="never assumed to be zero"
                small
              />
            </div>
            <p className="note" style={{ maxWidth: '80ch' }}>
              Membership comes from month-end revisions of the published constituents list, so an entry or exit is
              located within a month rather than to the day. Both edges of that bracket are stored, and the backtest
              takes the conservative one: a company is bought only once it was certainly a member, and held through
              the window in which it was dropped.
            </p>
          </>
        ) : (
          <p className="note">{universe.reason}</p>
        )}
      </div>
    </Card>
  );
}

/* --------------------------------------------------------------- periods */

export function PeriodsPanel({ periods, current }: { periods: PeriodInfo[]; current: string }) {
  const columns: Array<Column<PeriodInfo>> = [
    {
      key: 'label',
      header: 'Period',
      sortValue: (row) => row.label,
      sortLabel: 'period',
      render: (row) => (
        <span className="row-tight">
          <span className="strong">{humanize(row.label)}</span>
          {row.label === current && <Pill tone="info">Showing</Pill>}
          {row.sealed && <Pill tone="neutral">Sealed</Pill>}
        </span>
      ),
    },
    {
      key: 'span',
      header: 'Span',
      sortValue: (row) => row.start,
      sortLabel: 'start date',
      render: (row) => (
        <span className="t-small secondary nowrap">
          {row.start === row.end ? 'not started' : `${date(row.start)} – ${date(row.end)}`}
        </span>
      ),
    },
    {
      key: 'years',
      header: 'Years',
      numeric: true,
      sortValue: (row) => row.years,
      sortLabel: 'length in years',
      render: (row) => <span className="tnum">{row.years === 0 ? '—' : num(row.years, 1)}</span>,
    },
    {
      key: 'evaluated',
      header: 'Evaluated',
      sortValue: (row) => (row.evaluated ? 1 : 0),
      sortLabel: 'whether it has been evaluated',
      render: (row) =>
        row.label === 'holdout' && !row.evaluated ? (
          <Pill tone="neutral">Never opened</Pill>
        ) : row.evaluated ? (
          <Pill tone="info">{row.accesses > 0 ? `${int(row.accesses)} time(s)` : 'Yes'}</Pill>
        ) : (
          <Pill tone="neutral">Not yet</Pill>
        ),
    },
    {
      key: 'purpose',
      header: 'Purpose',
      width: '48%',
      render: (row) => (
        <span className="t-small secondary" style={{ display: 'block', maxWidth: '64ch' }}>
          {row.purpose}
        </span>
      ),
    },
  ];

  return (
    <Card
      title="Validation structure"
      subtitle="Three consecutive calendar periods, never a shuffled split. The holdout requires an explicit command and writes a permanent audit record."
      flush
    >
      <DataTable columns={columns} rows={periods} rowKey={(row) => row.label} ariaLabel="Validation periods" />
    </Card>
  );
}
