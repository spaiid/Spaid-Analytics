import type { OpportunitiesResponse } from '../../api/types';
import * as fmt from '../../lib/format';
import { Card, Meter, MissingValue, Pill, StatTile } from '../ui';
import { ALL, type BandCount } from './derive';
import { bandTone } from './labels';

export interface OpportunityHeaderProps {
  data: OpportunitiesResponse;
  /** Rows currently passing the filters. */
  shown: number;
  bandCounts: BandCount[];
  selectedBand: string;
  onSelectBand: (band: string) => void;
}

/**
 * What this page is made of: how much of the universe is scored, when it was
 * scored, which spec decided the labels, and how the bands are distributed.
 */
export function OpportunityHeader({
  data,
  shown,
  bandCounts,
  selectedBand,
  onSelectBand,
}: OpportunityHeaderProps) {
  const coverage =
    data.universe_size > 0 ? (data.scored_count / data.universe_size) * 100 : null;

  return (
    <Card
      title="Opportunities"
      subtitle="The ranked list. Every score, band and label on this page is computed by the API; the page formats and lays them out."
    >
      <div className="grid grid-4">
        <StatTile
          label="Scored"
          value={fmt.int(data.scored_count)}
          sub={`of ${fmt.int(data.universe_size)} in the universe`}
        >
          <div style={{ marginTop: 8 }}>
            <Meter
              value={coverage}
              label="Share of the universe with a composite score"
              valueText={coverage === null ? undefined : `${fmt.pct(coverage / 100)} of the universe`}
            />
          </div>
        </StatTile>

        <StatTile
          label="Shown"
          value={fmt.int(shown)}
          sub={shown === data.rows.length ? 'no filters applied' : `of ${fmt.int(data.rows.length)} ranked rows`}
        />

        <StatTile
          label="As of"
          value={data.as_of ? fmt.date(data.as_of) : <MissingValue status="missing" label="As-of date" />}
          sub={
            data.freshness.age_hours === null
              ? (data.freshness.detail ?? 'Age unknown')
              : `${fmt.ageHours(data.freshness.age_hours)} old`
          }
          small
        >
          {data.freshness.is_stale ? (
            <div style={{ marginTop: 8 }}>
              <Pill tone="warning" title={data.freshness.detail ?? undefined}>
                Stale
              </Pill>
            </div>
          ) : null}
        </StatTile>

        <StatTile
          label="Scoring spec"
          value={<span className="mono">{fmt.dash(data.spec_version)}</span>}
          sub="Bands, labels and assessments come from this spec, server-side."
          small
        />
      </div>

      <div className="hr" />

      <div className="stack" style={{ gap: 10 }}>
        <div className="t-label">Band distribution</div>
        <div className="band-chips" role="group" aria-label="Filter by band">
          {bandCounts.map((entry) => {
            const pressed = selectedBand === entry.value;
            return (
              <button
                key={entry.value}
                type="button"
                className="band-chip"
                aria-pressed={pressed}
                aria-label={`${entry.label}, ${fmt.int(entry.count)} ${entry.count === 1 ? 'company' : 'companies'}`}
                onClick={() => onSelectBand(pressed ? ALL : entry.value)}
              >
                <Pill tone={bandTone(entry.label)}>{entry.label}</Pill>
                <span className="band-chip-count">{fmt.int(entry.count)}</span>
              </button>
            );
          })}
        </div>
      </div>
    </Card>
  );
}
