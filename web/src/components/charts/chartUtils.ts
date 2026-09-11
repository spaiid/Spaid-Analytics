/**
 * chartUtils.ts — the geometry ported from spaid/api/web/charts.js.
 *
 * Mark specs these helpers exist to serve:
 *   line 2px round cap/join · bar <=24px with a 4px rounded data-end, square at
 *   the baseline · markers r>=4 with a 2px surface ring · area fill ~10% ·
 *   hairline solid grid · hit targets >=24px · legend for >=2 series ·
 *   every chart ships a table-view twin.
 */

import { useLayoutEffect, useRef, useState, type RefObject } from 'react';

export const SERIES_VARS = [
  '--series-1',
  '--series-2',
  '--series-3',
  '--series-4',
  '--series-5',
  '--series-6',
  '--series-7',
  '--series-8',
] as const;

/**
 * Categorical hue for slot `index`. Colour follows the entity, never its rank:
 * callers pass a stable slot index, never a filtered row number.
 */
export function seriesColor(index: number): string {
  const name = SERIES_VARS[((index % SERIES_VARS.length) + SERIES_VARS.length) % SERIES_VARS.length];
  return `var(${name})`;
}

/** Sequential blue ramp, light -> dark. */
export const SEQ_VARS = ['--seq-100', '--seq-200', '--seq-300', '--seq-400', '--seq-500', '--seq-600', '--seq-700'] as const;

export function sequentialColor(fraction: number): string {
  if (!Number.isFinite(fraction)) return 'var(--surface-3)';
  const index = Math.min(SEQ_VARS.length - 1, Math.max(0, Math.floor(fraction * SEQ_VARS.length)));
  return `var(${SEQ_VARS[index]})`;
}

export function isNum(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/** Axis ticks at 1/2/5 x 10^n steps, covering [min, max]. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0];
  let lo = min;
  let hi = max;
  if (lo === hi) {
    const pad = Math.abs(lo) || 1;
    lo -= pad * 0.5;
    hi += pad * 0.5;
  }
  const span = hi - lo;
  const raw = span / Math.max(1, count);
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const normalised = raw / magnitude;
  const step = (normalised >= 7.5 ? 10 : normalised >= 3.5 ? 5 : normalised >= 1.5 ? 2 : 1) * magnitude;
  const start = Math.ceil(lo / step) * step;
  const ticks: number[] = [];
  for (let value = start; value <= hi + step * 1e-9; value += step) {
    ticks.push(Math.abs(value) < step * 1e-9 ? 0 : value);
  }
  return ticks.length > 0 ? ticks : [lo, hi];
}

export type BarDirection = 'up' | 'down' | 'left' | 'right';

/** Bar path: `r`px rounded data-end, square at the baseline. */
export function barPath(x: number, y: number, w: number, h: number, r: number, direction: BarDirection): string {
  const rr = Math.max(0, Math.min(r, Math.min(w, h) / 2));
  if (h <= 0.5 || w <= 0.5) return '';
  switch (direction) {
    case 'up': // column grows up from the baseline at y+h
      return `M${x},${y + h} L${x},${y + rr} Q${x},${y} ${x + rr},${y} L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr} L${x + w},${y + h} Z`;
    case 'down':
      return `M${x},${y} L${x},${y + h - rr} Q${x},${y + h} ${x + rr},${y + h} L${x + w - rr},${y + h} Q${x + w},${y + h} ${x + w},${y + h - rr} L${x + w},${y} Z`;
    case 'right': // bar grows right from the baseline at x
      return `M${x},${y} L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr} L${x + w},${y + h - rr} Q${x + w},${y + h} ${x + w - rr},${y + h} L${x},${y + h} Z`;
    case 'left':
      return `M${x + w},${y} L${x + rr},${y} Q${x},${y} ${x},${y + rr} L${x},${y + h - rr} Q${x},${y + h} ${x + rr},${y + h} L${x + w},${y + h} Z`;
    default:
      return '';
  }
}

/** Bar geometry constants from the ported chart code. */
export const BAR = { GAP: 2, MAX_WIDTH: 24, RADIUS: 4 } as const;

/** Polyline path with gaps at nulls. */
export function linePath(
  values: Array<number | null | undefined>,
  sx: (index: number) => number,
  sy: (value: number) => number,
): string {
  let d = '';
  let pen = false;
  values.forEach((value, index) => {
    if (!isNum(value)) {
      pen = false;
      return;
    }
    d += `${pen ? 'L' : 'M'}${sx(index)},${sy(value)}`;
    pen = true;
  });
  return d;
}

/** Measure a container so the SVG viewBox matches CSS pixels 1:1. */
export function useElementWidth<T extends HTMLElement>(fallback = 640): [RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(fallback);

  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;
    const measure = () => {
      const next = node.clientWidth;
      if (next > 0) setWidth((previous) => (Math.abs(next - previous) > 0.5 ? next : previous));
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return [ref, width];
}

/** Index of the nearest x slot under the pointer. */
export function indexFromPointer(
  clientX: number,
  svg: SVGSVGElement,
  viewWidth: number,
  padLeft: number,
  plotWidth: number,
  count: number,
): number {
  const rect = svg.getBoundingClientRect();
  if (rect.width === 0 || count <= 1) return 0;
  const px = ((clientX - rect.left) / rect.width) * viewWidth;
  const index = Math.round(((px - padLeft) / plotWidth) * (count - 1));
  return clamp(index, 0, count - 1);
}
