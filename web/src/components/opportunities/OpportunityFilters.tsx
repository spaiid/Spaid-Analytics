import { useId } from 'react';
import * as fmt from '../../lib/format';
import { ALL, UNBANDED, UNBANDED_LABEL, hasActiveFilters, type OpportunityFilterState } from './derive';

export interface OpportunityFiltersProps {
  sectors: readonly string[];
  bands: readonly string[];
  /** Offer the "not scored" option only when such rows exist. */
  showUnbanded: boolean;
  value: OpportunityFilterState;
  onChange: (next: OpportunityFilterState) => void;
  onReset: () => void;
  /** Rows after filtering. */
  shown: number;
  /** Rows before filtering. */
  total: number;
  /** Rows dropped purely because they have no score to compare. */
  unscoredHidden: number;
}

/**
 * Sector, band, minimum score and free text, all client-side over the ranked
 * set the API already sent. Every control has a visible label; the result count
 * is a polite live region so a filter change is announced, not just seen.
 */
export function OpportunityFilters({
  sectors,
  bands,
  showUnbanded,
  value,
  onChange,
  onReset,
  shown,
  total,
  unscoredHidden,
}: OpportunityFiltersProps) {
  const sectorId = useId();
  const bandId = useId();
  const minScoreId = useId();
  const searchId = useId();
  const active = hasActiveFilters(value);

  return (
    <div className="filterbar opp-filters" role="group" aria-label="Filter the ranked list">
      <div className="filter-field">
        <label className="t-label" htmlFor={sectorId}>
          Sector
        </label>
        <span className="control">
          <select
            id={sectorId}
            value={value.sector}
            onChange={(event) => onChange({ ...value, sector: event.target.value })}
          >
            <option value={ALL}>All sectors</option>
            {sectors.map((sector) => (
              <option key={sector} value={sector}>
                {sector}
              </option>
            ))}
          </select>
        </span>
      </div>

      <div className="filter-field">
        <label className="t-label" htmlFor={bandId}>
          Band
        </label>
        <span className="control">
          <select
            id={bandId}
            value={value.band}
            onChange={(event) => onChange({ ...value, band: event.target.value })}
          >
            <option value={ALL}>All bands</option>
            {bands.map((band) => (
              <option key={band} value={band}>
                {band}
              </option>
            ))}
            {showUnbanded && <option value={UNBANDED}>{UNBANDED_LABEL}</option>}
          </select>
        </span>
      </div>

      <div className="filter-field">
        <label className="t-label" htmlFor={minScoreId}>
          Minimum score
        </label>
        <span className="control range-control">
          <input
            id={minScoreId}
            type="range"
            min={0}
            max={100}
            step={1}
            value={value.minScore}
            aria-valuetext={value.minScore === 0 ? 'No minimum' : `${value.minScore} out of 100`}
            onChange={(event) => onChange({ ...value, minScore: Number(event.target.value) })}
          />
          <span className="range-value">{value.minScore === 0 ? fmt.DASH : fmt.int(value.minScore)}</span>
        </span>
      </div>

      <div className="filter-field">
        <label className="t-label" htmlFor={searchId}>
          Search
        </label>
        <input
          id={searchId}
          className="search-input"
          type="search"
          placeholder="Ticker or company name"
          value={value.search}
          onChange={(event) => onChange({ ...value, search: event.target.value })}
        />
      </div>

      <button type="button" className="btn" onClick={onReset} disabled={!active}>
        Clear filters
      </button>

      <p className="result-count" role="status" aria-live="polite">
        Showing <strong>{fmt.int(shown)}</strong> of {fmt.int(total)}{' '}
        {total === 1 ? 'company' : 'companies'}
        {active ? ' after filters' : ''}.
        {unscoredHidden > 0
          ? ` ${fmt.int(unscoredHidden)} with no score ${unscoredHidden === 1 ? 'is' : 'are'} hidden by the minimum-score filter.`
          : ''}
      </p>
    </div>
  );
}
