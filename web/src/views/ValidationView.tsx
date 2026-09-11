/**
 * ValidationView — whether the score predicts anything, and what may be claimed.
 *
 * The page is ordered by what a reader should believe first. The conclusion and
 * its gates come before any performance number, because a good-looking equity
 * curve read before the caveats is how a person talks themselves into an edge
 * that is not there. The ranking diagnostics come before the portfolio results
 * for the same reason: twenty holdings is too small a sample to learn from,
 * while an information coefficient is measured across the whole cross-section.
 *
 * Three renders are kept distinct throughout. "No run has been stored" is a
 * state, not an error and not an empty table; it tells you the command to run.
 */

import { useEffect, useRef, useState } from 'react';
import { apiPaths, useApi } from '../api/client';
import type { TrialRegistry, ValidationReport } from '../api/validation-types';
import {
  AblationTable,
  BandTable,
  BenchmarkTable,
  CategoryTable,
  ContributionPanel,
  DataGates,
  DecileChart,
  DrawdownChart,
  GrowthChart,
  HeadlineTiles,
  IcOverTime,
  IcTable,
  LimitationsPanel,
  OverfittingPanel,
  PeriodsPanel,
  PersistenceTiles,
  RegimeCut,
  ResultGates,
  RobustnessTable,
  RollingExcessChart,
  SectorCut,
  SizeCut,
  StrategyPanel,
  TrialRegistryTable,
  UniversePanel,
  VariantTable,
  VerdictPanel,
  YearCut,
  YearlyReturns,
} from '../components/validation';
import { Banner, Card, EmptyState, ErrorState, Pill, SkeletonTable, StatTile } from '../components/ui';
import { dateTime, humanize, int, num } from '../lib/format';

const PERIODS = ['development', 'validation', 'holdout'] as const;

function PeriodPicker({
  value,
  onChange,
  available,
}: {
  value: string;
  onChange: (period: string) => void;
  available: Set<string>;
}) {
  return (
    <div className="row-tight" role="group" aria-label="Validation period">
      {PERIODS.map((period) => {
        const isCurrent = period === value;
        const hasRun = available.has(period);
        return (
          <button
            key={period}
            type="button"
            className={isCurrent ? 'btn btn-sm is-active' : 'btn btn-sm'}
            aria-pressed={isCurrent}
            onClick={() => onChange(period)}
            title={hasRun ? `Show the ${period} period` : `No run stored for the ${period} period`}
          >
            {humanize(period)}
            {!hasRun && <span className="t-micro muted"> · none</span>}
          </button>
        );
      })}
    </div>
  );
}

export interface ValidationViewProps {
  refreshToken?: number | null;
}

export function ValidationView({ refreshToken = null }: ValidationViewProps) {
  const [period, setPeriod] = useState<string>('development');
  const report = useApi<ValidationReport>(apiPaths.validation(period));
  const trials = useApi<TrialRegistry>(apiPaths.validationTrials(200));

  const reloadRef = useRef({ report: report.reload, trials: trials.reload });
  reloadRef.current = { report: report.reload, trials: trials.reload };
  const seenTokenRef = useRef(refreshToken);
  useEffect(() => {
    if (refreshToken === null || refreshToken === seenTokenRef.current) return;
    seenTokenRef.current = refreshToken;
    reloadRef.current.report();
    reloadRef.current.trials();
  }, [refreshToken]);

  if (report.state === 'loading') {
    return (
      <Card title="Validation" subtitle="Whether the ranking predicts future returns, and what may be claimed about it.">
        <SkeletonTable rows={8} columns={5} label="Loading the validation report" />
      </Card>
    );
  }

  if (report.state === 'error') {
    return (
      <Card title="Validation">
        <ErrorState
          title="Could not read the validation report"
          message={report.message}
          detail={apiPaths.validation(period)}
          onRetry={report.reload}
        />
      </Card>
    );
  }

  const data = report.data;
  const evaluated = new Set(data.periods.filter((p) => p.evaluated).map((p) => p.label));

  if (!data.available) {
    return (
      <div className="stack">
        <Card
          title="Validation"
          subtitle="No backtest has been stored for this period yet."
          actions={<PeriodPicker value={period} onChange={setPeriod} available={evaluated} />}
        >
          <EmptyState
            title={`Nothing stored for the ${humanize(period).toLowerCase()} period`}
            message={
              data.message ??
              'Run the validation protocol to produce a report. Nothing has failed — this period simply has not been evaluated.'
            }
          >
            <code>spaid validate --period {period}</code>
          </EmptyState>
        </Card>
        {data.universe !== null && <UniversePanel universe={data.universe} />}
        <PeriodsPanel periods={data.periods} current={period} />
      </div>
    );
  }

  const primary =
    data.variants.find((variant) => variant.is_primary) ?? (data.variants.length > 0 ? data.variants[0] : null);
  const benchmarkNames = primary === null ? [] : primary.benchmarks.map((b) => b.benchmark);

  return (
    <div className="stack">
      <Card
        title="Validation"
        subtitle={
          <>
            Frozen Strategy Version 1, tested as it stands. Nothing on this page has been tuned to the result:{' '}
            the periods, the portfolio variants, the costs, the execution delay and every threshold were declared in{' '}
            <code className="mono">spaid/config/validation.py</code> before the first run.
          </>
        }
        actions={<PeriodPicker value={period} onChange={setPeriod} available={evaluated} />}
        busy={report.refreshing}
      >
        <div className="grid grid-4">
          <StatTile
            label="Decision dates"
            value={int(data.decision_dates)}
            sub={`${data.period?.start ?? ''} to ${data.period?.end ?? ''}`}
            small
          />
          <StatTile
            label="Universe per date"
            value={num(data.universe_per_date.median_eligible ?? null, 0)}
            sub={`eligible, median (${int(data.universe_per_date.min_eligible ?? null)}–${int(
              data.universe_per_date.max_eligible ?? null,
            )})`}
            small
          />
          <StatTile
            label="Execution"
            value={`${int(data.execution.delay_days ?? null)}-day delay`}
            sub={`${humanize(data.execution.rebalance ?? 'monthly')} rebalance`}
            small
          />
          <StatTile
            label="Assumed cost"
            value={`${num(data.costs.one_way_bps ?? null, 1)} bps`}
            sub="per side, spread + slippage + commission"
            small
          />
        </div>
      </Card>

      {data.verdict !== null && <VerdictPanel verdict={data.verdict} periodLabel={data.period_label} />}
      {data.verdict !== null && <LimitationsPanel verdict={data.verdict} />}
      {data.universe !== null && <UniversePanel universe={data.universe} />}
      {data.verdict !== null && <DataGates gates={data.verdict.data_gates} />}
      {data.verdict !== null && <ResultGates gates={data.verdict.result_gates} />}

      {data.signal !== null && (
        <>
          <Banner tone="info" title="Does the ranking itself carry information?">
            These measurements use the whole cross-section on every decision date, not the twenty names the portfolio
            happened to hold. A portfolio result can be produced by two lucky holdings; an information coefficient
            cannot.
          </Banner>
          <IcTable rows={data.signal.ic} />
          <IcOverTime signal={data.signal} />
          <DecileChart signal={data.signal} />
          <BandTable rows={data.signal.bands} />
          <CategoryTable rows={data.signal.categories} />
          <AblationTable rows={data.signal.ablations} />
          <PersistenceTiles signal={data.signal} />
          <div className="grid grid-2">
            <SectorCut rows={data.signal.by_sector} />
            <SizeCut rows={data.signal.by_size} />
          </div>
          <div className="grid grid-2">
            <YearCut rows={data.signal.by_year} />
            <RegimeCut rows={data.signal.by_regime} />
          </div>
        </>
      )}

      {primary !== null && (
        <>
          <HeadlineTiles variant={primary} />
          <GrowthChart points={data.equity_curve} benchmarks={benchmarkNames} />
          <DrawdownChart points={data.equity_curve} />
          <RollingExcessChart points={data.equity_curve} benchmark="SPY" />
          <VariantTable variants={data.variants} />
          <BenchmarkTable variant={primary} />
          <YearlyReturns variant={primary} />
          <ContributionPanel byStock={data.contribution_by_stock} bySector={data.contribution_by_sector} />
        </>
      )}

      <RobustnessTable rows={data.robustness} summary={data.robustness_summary} />
      <OverfittingPanel overfitting={data.overfitting} />

      {data.strategy !== null && (
        <StrategyPanel
          strategy={data.strategy}
          dataVersion={data.data_version}
          universeVersion={data.universe_version}
        />
      )}
      <PeriodsPanel periods={data.periods} current={period} />

      {trials.state === 'data' ? (
        <TrialRegistryTable trials={trials.data.trials} note={trials.data.note} />
      ) : trials.state === 'error' ? (
        <Card title="Trial registry">
          <ErrorState title="Could not read the trial registry" message={trials.message} onRetry={trials.reload} />
        </Card>
      ) : (
        <Card title="Trial registry">
          <SkeletonTable rows={5} columns={6} label="Loading the trial registry" />
        </Card>
      )}

      <p className="t-micro muted">
        <Pill tone="neutral">{data.period_label}</Pill> run generated{' '}
        {data.generated_at === null ? 'at an unrecorded time' : dateTime(data.generated_at)} · data {data.data_version} ·
        universe {data.universe_version} · code {data.code_commit}
      </p>
    </div>
  );
}
