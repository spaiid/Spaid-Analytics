import { useCallback, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';

export type SortDirection = 'asc' | 'desc';

export interface SortState {
  key: string;
  direction: SortDirection;
}

export type SortValue = number | string | null | undefined;

export interface Column<T> {
  /** Stable id; also the sort key. */
  key: string;
  header: ReactNode;
  render: (row: T, index: number) => ReactNode;
  /** Right-align + tabular numerals. */
  numeric?: boolean;
  align?: 'left' | 'right' | 'center';
  /** Provide this to make the column sortable. Return null for "no value". */
  sortValue?: (row: T) => SortValue;
  /** First click direction. Defaults to 'desc' for numeric, 'asc' otherwise. */
  defaultDirection?: SortDirection;
  /** Accessible name for the sort button, e.g. "Composite score". */
  sortLabel?: string;
  /** Column header tooltip. */
  title?: string;
  width?: string;
  className?: string;
}

export interface DataTableProps<T> {
  columns: Array<Column<T>>;
  rows: T[];
  rowKey: (row: T, index: number) => string;
  /** Visible caption. Prefer this over ariaLabel. */
  caption?: ReactNode;
  /** Accessible name when there is no visible caption. */
  ariaLabel?: string;
  /** Makes rows clickable AND keyboard operable (Enter/Space, arrow keys). */
  onRowActivate?: (row: T, index: number) => void;
  /** Accessible name for an activatable row, e.g. "Open AAPL". Required for a11y when onRowActivate is set. */
  rowLabel?: (row: T) => string;
  isRowActive?: (row: T) => boolean;
  rowClassName?: (row: T, index: number) => string | undefined;
  /** Controlled sort. Omit for internal sort state. */
  sort?: SortState | null;
  onSortChange?: (sort: SortState) => void;
  defaultSort?: SortState | null;
  /** When true the parent has already ordered `rows`; the table only reports clicks. */
  manualSort?: boolean;
  /** Rendered in place of the body when there are no rows. */
  empty?: ReactNode;
  sticky?: boolean;
  footer?: ReactNode;
  className?: string;
}

function compareValues(a: SortValue, b: SortValue, direction: SortDirection): number {
  const aMissing = a === null || a === undefined || (typeof a === 'number' && !Number.isFinite(a));
  const bMissing = b === null || b === undefined || (typeof b === 'number' && !Number.isFinite(b));
  // Missing values always sort last, in both directions — absence is not "low".
  if (aMissing && bMissing) return 0;
  if (aMissing) return 1;
  if (bMissing) return -1;

  let result: number;
  if (typeof a === 'number' && typeof b === 'number') {
    result = a - b;
  } else {
    result = String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' });
  }
  return direction === 'asc' ? result : -result;
}

function ariaSortFor(active: boolean, direction: SortDirection): 'ascending' | 'descending' | 'none' {
  if (!active) return 'none';
  return direction === 'asc' ? 'ascending' : 'descending';
}

/**
 * A generic table.
 *
 * Accessibility contract:
 *  - sortable headers are real <button>s inside <th scope="col" aria-sort>
 *  - activatable rows are focusable and respond to Enter/Space, with Arrow
 *    Up/Down moving between rows and Home/End jumping to the ends
 *  - a caption or aria-label always names the table
 */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  caption,
  ariaLabel,
  onRowActivate,
  rowLabel,
  isRowActive,
  rowClassName,
  sort,
  onSortChange,
  defaultSort = null,
  manualSort = false,
  empty,
  sticky = false,
  footer,
  className,
}: DataTableProps<T>) {
  const [internalSort, setInternalSort] = useState<SortState | null>(defaultSort);
  const bodyRef = useRef<HTMLTableSectionElement>(null);

  const activeSort = sort !== undefined ? sort : internalSort;

  const handleSort = useCallback(
    (column: Column<T>) => {
      const fallback: SortDirection = column.defaultDirection ?? (column.numeric ? 'desc' : 'asc');
      const next: SortState =
        activeSort && activeSort.key === column.key
          ? { key: column.key, direction: activeSort.direction === 'asc' ? 'desc' : 'asc' }
          : { key: column.key, direction: fallback };
      if (sort === undefined) setInternalSort(next);
      onSortChange?.(next);
    },
    [activeSort, onSortChange, sort],
  );

  const sortedRows = useMemo(() => {
    if (manualSort || !activeSort) return rows;
    const column = columns.find((candidate) => candidate.key === activeSort.key);
    if (!column || !column.sortValue) return rows;
    const accessor = column.sortValue;
    return [...rows].sort((a, b) => compareValues(accessor(a), accessor(b), activeSort.direction));
  }, [rows, columns, activeSort, manualSort]);

  const focusRow = useCallback((delta: number, from: HTMLTableRowElement) => {
    const body = bodyRef.current;
    if (!body) return;
    const focusable = Array.from(body.querySelectorAll<HTMLTableRowElement>('tr[data-row="true"]'));
    const index = focusable.indexOf(from);
    if (index === -1) return;
    const target = focusable[Math.min(focusable.length - 1, Math.max(0, index + delta))];
    target?.focus();
  }, []);

  const handleRowKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTableRowElement>, row: T, index: number) => {
      if (!onRowActivate) return;
      const isSelf = event.target === event.currentTarget;
      switch (event.key) {
        case 'Enter':
        case ' ':
        case 'Spacebar':
          if (!isSelf) return;
          event.preventDefault();
          onRowActivate(row, index);
          break;
        case 'ArrowDown':
          event.preventDefault();
          focusRow(1, event.currentTarget);
          break;
        case 'ArrowUp':
          event.preventDefault();
          focusRow(-1, event.currentTarget);
          break;
        case 'Home':
          event.preventDefault();
          focusRow(-rows.length, event.currentTarget);
          break;
        case 'End':
          event.preventDefault();
          focusRow(rows.length, event.currentTarget);
          break;
        default:
          break;
      }
    },
    [focusRow, onRowActivate, rows.length],
  );

  const cellClass = (column: Column<T>): string | undefined => {
    const align = column.align ?? (column.numeric ? 'right' : 'left');
    const classes = [
      column.numeric ? 'num' : null,
      align === 'center' ? 'center' : null,
      !column.numeric && align === 'right' ? 'num' : null,
      column.className,
    ].filter(Boolean);
    return classes.length > 0 ? classes.join(' ') : undefined;
  };

  const tableClasses = ['data', sticky ? 'is-sticky' : null].filter(Boolean).join(' ');

  return (
    <div className={['table-wrap', className].filter(Boolean).join(' ')}>
      {/* A <caption> names the table for AT; aria-label covers the captionless case. */}
      <table className={tableClasses} aria-label={caption === undefined ? ariaLabel : undefined}>
        {caption !== undefined && <caption>{caption}</caption>}
        <thead>
          <tr>
            {columns.map((column) => {
              const sortable = Boolean(column.sortValue);
              const active = Boolean(activeSort && activeSort.key === column.key);
              const direction = activeSort?.direction ?? 'desc';
              return (
                <th
                  key={column.key}
                  scope="col"
                  className={cellClass(column)}
                  style={column.width ? { width: column.width } : undefined}
                  title={column.title}
                  aria-sort={sortable ? ariaSortFor(active, direction) : undefined}
                >
                  {sortable ? (
                    <button
                      type="button"
                      className="th-sort"
                      onClick={() => handleSort(column)}
                      aria-label={column.sortLabel ? `Sort by ${column.sortLabel}` : undefined}
                    >
                      <span>{column.header}</span>
                      <span className="sort-caret" aria-hidden="true">
                        {active ? (direction === 'asc' ? '▲' : '▼') : '↕'}
                      </span>
                    </button>
                  ) : (
                    column.header
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody ref={bodyRef}>
          {sortedRows.length === 0 ? (
            <tr>
              <td colSpan={columns.length}>{empty ?? <span className="muted">No rows.</span>}</td>
            </tr>
          ) : (
            sortedRows.map((row, index) => {
              const activatable = Boolean(onRowActivate);
              const classes = [
                activatable ? 'clickable' : null,
                isRowActive?.(row) ? 'is-active' : null,
                rowClassName?.(row, index),
              ]
                .filter(Boolean)
                .join(' ');
              return (
                <tr
                  key={rowKey(row, index)}
                  className={classes === '' ? undefined : classes}
                  data-row={activatable ? 'true' : undefined}
                  tabIndex={activatable ? 0 : undefined}
                  aria-label={activatable ? rowLabel?.(row) : undefined}
                  onClick={activatable ? () => onRowActivate?.(row, index) : undefined}
                  onKeyDown={activatable ? (event) => handleRowKeyDown(event, row, index) : undefined}
                >
                  {columns.map((column) => (
                    <td key={column.key} className={cellClass(column)}>
                      {column.render(row, index)}
                    </td>
                  ))}
                </tr>
              );
            })
          )}
        </tbody>
        {footer !== undefined && <tfoot>{footer}</tfoot>}
      </table>
    </div>
  );
}
