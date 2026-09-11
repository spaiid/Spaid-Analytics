import { ArrowDownIcon, ArrowUpIcon, FlatIcon, MissingValue } from '../ui';
import { isNum } from '../charts/chartUtils';
import * as fmt from '../../lib/format';

export interface ScoreDeltaProps {
  /** Signed points from the API, e.g. `score_change_1w`. */
  value: number | null;
  /** Spoken window, e.g. "1 week". */
  windowLabel: string;
}

/**
 * A change in the composite score. Direction is carried three ways — arrow
 * glyph, sign character and colour — so it survives greyscale and CVD, and the
 * window is spoken for screen-reader users who never hear the column header
 * alongside the cell.
 */
export function ScoreDelta({ value, windowLabel }: ScoreDeltaProps) {
  if (!isNum(value)) {
    return <MissingValue status="missing" label={`Score change over ${windowLabel}`} />;
  }

  const direction = value > 0 ? 'up' : value < 0 ? 'down' : 'flat';
  const spoken = direction === 'up' ? 'up' : direction === 'down' ? 'down' : 'unchanged';
  const Icon = direction === 'up' ? ArrowUpIcon : direction === 'down' ? ArrowDownIcon : FlatIcon;

  return (
    <span className={`stat-delta ${direction}`}>
      <Icon size={11} />
      <span>
        <span className="sr-only">{`${spoken} over ${windowLabel}, `}</span>
        {value === 0 ? fmt.num(0, 1) : fmt.points(value)}
        <span className="sr-only"> points</span>
      </span>
    </span>
  );
}
