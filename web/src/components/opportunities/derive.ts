/**
 * derive.ts — presentation arithmetic over rows the API already ranked.
 *
 * Allowed here: filtering, counting, ordering by a server-supplied order, and
 * summarising the numbers the API sent (histogram bins, a mean per sector) so
 * they can be drawn. NOT allowed here, and absent: any threshold, band, verdict
 * or metric. Every qualitative reading on this page is the server's own string.
 */

import type { OpportunityRow } from '../../api/types';
import { isNum } from '../charts/chartUtils';

/** Sentinel for the "no filter" option in the sector and band selects. */
export const ALL = 'all';
/** Sentinel for rows the API left without a band (they have no score yet). */
export const UNBANDED = '__unbanded__';
/** Label for those rows — matches the server's own wording for a null score. */
export const UNBANDED_LABEL = 'Not scored';

export interface OpportunityFilterState {
  /** ALL, or an exact sector string from the response. */
  sector: string;
  /** ALL, UNBANDED, or an exact band string from the response. */
  band: string;
  /** 0 means "no minimum". */
  minScore: number;
  search: string;
}

export const EMPTY_FILTERS: OpportunityFilterState = {
  sector: ALL,
  band: ALL,
  minScore: 0,
  search: '',
};

export function hasActiveFilters(filters: OpportunityFilterState): boolean {
  return (
    filters.sector !== ALL ||
    filters.band !== ALL ||
    filters.minScore > 0 ||
    filters.search.trim() !== ''
  );
}

export interface FilterOutcome {
  rows: OpportunityRow[];
  /**
   * Rows that matched every other filter but were dropped because they have no
   * score to compare against the minimum. Surfaced in the UI so a minimum score
   * never silently deletes the companies whose data is merely missing.
   */
  unscoredHidden: number;
}

export function applyFilters(
  rows: readonly OpportunityRow[],
  filters: OpportunityFilterState,
): FilterOutcome {
  const needle = filters.search.trim().toLowerCase();
  const out: OpportunityRow[] = [];
  let unscoredHidden = 0;

  for (const row of rows) {
    if (filters.sector !== ALL && row.sector !== filters.sector) continue;

    if (filters.band === UNBANDED) {
      if (row.band) continue;
    } else if (filters.band !== ALL && row.band !== filters.band) {
      continue;
    }

    if (needle !== '') {
      const ticker = row.ticker.toLowerCase();
      const name = row.name.toLowerCase();
      if (!ticker.includes(needle) && !name.includes(needle)) continue;
    }

    if (filters.minScore > 0) {
      if (!isNum(row.score)) {
        unscoredHidden += 1;
        continue;
      }
      if (row.score < filters.minScore) continue;
    }

    out.push(row);
  }

  return { rows: out, unscoredHidden };
}

/**
 * Rank each band by its position in the response's `bands` array. The API sends
 * them best-first, so this is the server's ordering, not one invented here.
 */
export function bandOrderMap(bands: readonly string[]): Map<string, number> {
  const map = new Map<string, number>();
  bands.forEach((band, index) => map.set(band, index));
  return map;
}

export interface BandCount {
  /** The value to put in the filter state. */
  value: string;
  /** The text to show — the server's band label, or "Not scored". */
  label: string;
  count: number;
}

/** One entry per band the API knows about, in its order, plus unbanded rows. */
export function countByBand(
  rows: readonly OpportunityRow[],
  bands: readonly string[],
): BandCount[] {
  const counts = new Map<string, number>();
  let unbanded = 0;
  for (const row of rows) {
    if (!row.band) {
      unbanded += 1;
      continue;
    }
    counts.set(row.band, (counts.get(row.band) ?? 0) + 1);
  }

  const known = new Set(bands);
  const out: BandCount[] = bands.map((band) => ({
    value: band,
    label: band,
    count: counts.get(band) ?? 0,
  }));

  // A band the response did not list but a row carries anyway: show it rather
  // than drop the row's category on the floor.
  for (const [band, count] of counts) {
    if (!known.has(band)) out.push({ value: band, label: band, count });
  }

  if (unbanded > 0) out.push({ value: UNBANDED, label: UNBANDED_LABEL, count: unbanded });
  return out;
}

export interface ScoreBin {
  /** e.g. "45–50". */
  label: string;
  from: number;
  to: number;
  count: number;
}

/**
 * Histogram bins over the composite scores that came back. Bin edges are round
 * multiples of `width` so the axis reads cleanly; the counts are counts.
 */
export function binScores(scores: readonly number[], width = 5): ScoreBin[] {
  const finite = scores.filter((value) => Number.isFinite(value));
  if (finite.length === 0 || width <= 0) return [];

  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const low = Math.floor(min / width) * width;
  const high = Math.max(low + width, Math.ceil(max / width) * width);
  const count = Math.max(1, Math.round((high - low) / width));

  const bins: ScoreBin[] = Array.from({ length: count }, (_, index) => {
    const from = low + index * width;
    return { label: `${from}–${from + width}`, from, to: from + width, count: 0 };
  });

  for (const value of finite) {
    const index = Math.min(count - 1, Math.max(0, Math.floor((value - low) / width)));
    bins[index].count += 1;
  }
  return bins;
}

export interface SectorMean {
  sector: string;
  /** Mean of the composite scores the API returned for this sector. */
  mean: number;
  count: number;
}

export interface SectorMeans {
  /** Highest mean first. */
  sectors: SectorMean[];
  /** Mean across every scored row in the set, or null when there are none. */
  overallMean: number | null;
  scored: number;
  /** Scored rows the API gave no sector — excluded rather than lumped in. */
  withoutSector: number;
}

export function meanScoreBySector(rows: readonly OpportunityRow[]): SectorMeans {
  const sums = new Map<string, { sum: number; count: number }>();
  let total = 0;
  let scored = 0;
  let withoutSector = 0;

  for (const row of rows) {
    if (!isNum(row.score)) continue;
    scored += 1;
    total += row.score;
    if (!row.sector) {
      withoutSector += 1;
      continue;
    }
    const entry = sums.get(row.sector) ?? { sum: 0, count: 0 };
    entry.sum += row.score;
    entry.count += 1;
    sums.set(row.sector, entry);
  }

  const sectors: SectorMean[] = [...sums.entries()]
    .map(([sector, entry]) => ({ sector, mean: entry.sum / entry.count, count: entry.count }))
    .sort((a, b) => b.mean - a.mean);

  return {
    sectors,
    overallMean: scored > 0 ? total / scored : null,
    scored,
    withoutSector,
  };
}
