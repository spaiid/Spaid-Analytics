/** Pieces of the Opportunities view. Import from '../components/opportunities'. */

export { OpportunityHeader } from './OpportunityHeader';
export type { OpportunityHeaderProps } from './OpportunityHeader';

export { OpportunityFilters } from './OpportunityFilters';
export type { OpportunityFiltersProps } from './OpportunityFilters';

export { OpportunitiesTable, DEFAULT_OPPORTUNITY_SORT } from './OpportunitiesTable';
export type { OpportunitiesTableProps } from './OpportunitiesTable';

export { OpportunityCharts, ScoreDistributionChart, SectorScoreChart } from './OpportunityCharts';
export type {
  OpportunityChartsProps,
  ScoreDistributionChartProps,
  SectorScoreChartProps,
} from './OpportunityCharts';

export { ScoreDelta } from './ScoreDelta';
export type { ScoreDeltaProps } from './ScoreDelta';

export { bandTone, valuationTone } from './labels';

export {
  ALL,
  EMPTY_FILTERS,
  UNBANDED,
  UNBANDED_LABEL,
  applyFilters,
  bandOrderMap,
  binScores,
  countByBand,
  hasActiveFilters,
  meanScoreBySector,
} from './derive';
export type {
  BandCount,
  FilterOutcome,
  OpportunityFilterState,
  ScoreBin,
  SectorMean,
  SectorMeans,
} from './derive';

export { useDebouncedValue } from './useDebouncedValue';
