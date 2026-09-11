/**
 * StockHeader — identity, price, composite score and confidence.
 *
 * Confidence sits in the same tile shape and the same type size as the score:
 * the interface must never show a score without its confidence beside it.
 */

import type { ScoreChange, StockDetail } from '../../api/types';
import { ArrowDownIcon, ArrowUpIcon, FlatIcon, Meter, Pill, StatTile, ValueOrMissing } from '../ui';
import { currency, date, humanize, int, pct, points, score as fmtScore, signedPct } from '../../lib/format';
import { confidenceFraction, confidenceText } from './metricFormat';

function direction(value: number | null): 'up' | 'down' | 'flat' {
  if (typeof value !== 'number' || !Number.isFinite(value) || value === 0) return 'flat';
  return value > 0 ? 'up' : 'down';
}

function ChangeChip({ change }: { change: ScoreChange }) {
  const dir = direction(change.delta);
  const Icon = dir === 'up' ? ArrowUpIcon : dir === 'down' ? ArrowDownIcon : FlatIcon;
  const known = typeof change.delta === 'number' && Number.isFinite(change.delta);
  return (
    <span className="tag" title={change.detail ?? undefined}>
      <span className="muted">{change.window}</span>
      <span className={`stat-delta ${dir}`}>
        {known && <Icon size={10} />}
        <span>
          <span className="sr-only">Score change over {change.window}: </span>
          {points(change.delta)}
        </span>
        {known && <span className="sr-only"> points</span>}
      </span>
    </span>
  );
}

export interface StockHeaderProps {
  detail: StockDetail;
}

export function StockHeader({ detail }: StockHeaderProps) {
  const confidence = detail.confidence;
  const meta: string[] = [
    detail.sector ?? 'Sector unknown',
    detail.industry ?? 'Industry unknown',
    detail.business_model ? `${humanize(detail.business_model)} model` : 'Business model unknown',
  ];

  return (
    <div className="sd-head">
      <div className="sd-ident">
        <div className="sd-ticker">{detail.ticker}</div>
        <div className="sd-name">{detail.name}</div>
        <div className="sd-meta">
          {meta.map((item) => (
            <span className="tag" key={item}>
              {item}
            </span>
          ))}
        </div>
        {detail.score_changes.length > 0 && (
          <div className="sd-meta">
            {detail.score_changes.map((change) => (
              <ChangeChip change={change} key={change.window} />
            ))}
          </div>
        )}
        <div className="sd-sub">Scores as of {date(detail.as_of)}</div>
      </div>

      <div className="sd-tiles">
        <StatTile
          label="Price"
          value={<ValueOrMissing value={detail.price} format={(value) => currency(value)} label="Price" />}
          delta={
            typeof detail.price_change_1d === 'number' && Number.isFinite(detail.price_change_1d)
              ? {
                  text: signedPct(detail.price_change_1d),
                  direction: direction(detail.price_change_1d),
                  srLabel: 'over 1 day',
                }
              : null
          }
          sub="1-day change"
        />

        <StatTile
          label="Composite score"
          hint="0-100, the weighted blend of the four category scores. Sent by the API, never computed here."
          value={<ValueOrMissing value={detail.score} format={fmtScore} label="Composite score" />}
          sub={
            detail.band ? (
              <Pill tone="neutral" icon={false}>
                Band: {detail.band}
              </Pill>
            ) : (
              'Band not assigned'
            )
          }
        >
          <div className="sd-tile-meter">
            <Meter
              value={detail.score}
              label={`Composite score for ${detail.ticker}`}
              valueText={`${fmtScore(detail.score)} of 100`}
              size="lg"
            />
          </div>
        </StatTile>

        <StatTile
          label="Confidence"
          hint="How much weight to put on the score above: coverage, data age, valuation agreement and score stability."
          value={
            <ValueOrMissing
              value={confidence ? confidence.score : null}
              format={(value) => confidenceText(value)}
              label="Confidence"
              status="missing"
              detail="No confidence was recorded for this company."
            />
          }
          sub={
            confidence ? (
              <Pill tone="neutral" icon={false}>
                {confidence.label}
              </Pill>
            ) : (
              'Not assessed'
            )
          }
        >
          <div className="sd-tile-meter">
            <Meter
              value={confidence ? confidenceFraction(confidence.score) : null}
              min={0}
              max={1}
              label={`Confidence in the score for ${detail.ticker}`}
              valueText={confidence ? `${confidenceText(confidence.score)}, ${confidence.label}` : undefined}
              size="lg"
            />
          </div>
        </StatTile>

        <StatTile
          label="Rank"
          value={
            <ValueOrMissing
              value={detail.rank}
              format={(value) => `#${int(value)}`}
              label="Rank"
              status="missing"
              detail="This company has no rank because it was not scored on this date."
            />
          }
          sub={
            typeof detail.universe_size === 'number' && Number.isFinite(detail.universe_size)
              ? `of ${int(detail.universe_size)} scored`
              : 'Universe size unknown'
          }
          footer={
            typeof detail.percentile === 'number' && Number.isFinite(detail.percentile)
              ? `${pct(detail.percentile, 0)} percentile · coverage ${pct(detail.coverage, 0)}`
              : `Coverage ${pct(detail.coverage, 0)}`
          }
        />
      </div>
    </div>
  );
}
