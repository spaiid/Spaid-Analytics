/* ==========================================================================
   charts.js — hand-rolled SVG chart primitives.
   No build step, no runtime CDN. Follows the dataviz mark specs:
     line 2px round cap/join · bar <=24px with 4px rounded data-end square at
     the baseline · markers r>=4 with a 2px surface ring · area fill ~10% ·
     hairline solid grid · 2px surface gap between adjacent fills ·
     hit targets >=24px · legend for >=2 series · selective direct labels ·
     every chart has a table-view twin.
   All external strings enter the DOM via textContent, never innerHTML.
   ========================================================================== */

const SVG_NS = 'http://www.w3.org/2000/svg';

export const SERIES_VARS = [
  '--series-1', '--series-2', '--series-3', '--series-4',
  '--series-5', '--series-6', '--series-7', '--series-8',
];

/** Categorical hue for slot i. Color follows the entity, never its rank —
 *  callers pass a stable slot index, never a filtered row number. */
export function seriesColor(i) {
  return `var(${SERIES_VARS[i % SERIES_VARS.length]})`;
}

/* ---------------------------------------------------------------- helpers */

function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    node.setAttribute(k, String(v));
  }
  if (parent) parent.appendChild(node);
  return node;
}

function text(tag, attrs, content, parent) {
  const node = el(tag, attrs, parent);
  node.textContent = content;   // untrusted-safe
  return node;
}

function niceTicks(min, max, count = 5) {
  if (!isFinite(min) || !isFinite(max)) return [0];
  if (min === max) { const p = Math.abs(min) || 1; min -= p * 0.5; max += p * 0.5; }
  const span = max - min;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const start = Math.ceil(min / step) * step;
  const out = [];
  for (let v = start; v <= max + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out.length ? out : [min, max];
}

export const fmt = {
  num(v, d = 2) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  },
  compact(v) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    const a = Math.abs(v);
    if (a >= 1e12) return (v / 1e12).toFixed(1) + 'T';
    if (a >= 1e9) return (v / 1e9).toFixed(1) + 'B';
    if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (a >= 1e3) return (v / 1e3).toFixed(1) + 'K';
    return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  },
  pct(v, d = 1) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return (v * 100).toFixed(d) + '%';
  },
  signedPct(v, d = 1) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%';
  },
  signed(v, d = 2) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return (v >= 0 ? '+' : '') + v.toFixed(d);
  },
  date(s) {
    if (!s) return '—';
    const d = typeof s === 'string' ? new Date(s + 'T00:00:00') : s;
    if (isNaN(d)) return String(s);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  },
  dateShort(s) {
    if (!s) return '—';
    const d = typeof s === 'string' ? new Date(s + 'T00:00:00') : s;
    if (isNaN(d)) return String(s);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  },
};

/** Tooltip attached to a chart container. Values lead, names follow. */
function makeTooltip(container) {
  const tip = document.createElement('div');
  tip.className = 'tooltip';
  container.appendChild(tip);

  return {
    node: tip,
    show(title, rows, x, y) {
      tip.textContent = '';
      if (title) {
        const t = document.createElement('div');
        t.className = 'tooltip-title';
        t.textContent = title;
        tip.appendChild(t);
      }
      for (const r of rows) {
        const row = document.createElement('div');
        row.className = 'tooltip-row';
        if (r.color) {
          const key = document.createElement('span');
          key.className = 'tooltip-key';
          key.style.background = r.color;
          row.appendChild(key);
        }
        const name = document.createElement('span');
        name.className = 'tooltip-name';
        name.textContent = r.name;             // untrusted-safe
        row.appendChild(name);
        const val = document.createElement('span');
        val.className = 'tooltip-val';
        val.textContent = r.value;
        row.appendChild(val);
        tip.appendChild(row);
      }
      tip.classList.add('is-visible');
      const cw = container.clientWidth;
      const tw = tip.offsetWidth;
      let left = x + 14;
      if (left + tw > cw - 4) left = x - tw - 14;
      if (left < 4) left = 4;
      tip.style.left = left + 'px';
      tip.style.top = Math.max(4, y - tip.offsetHeight / 2) + 'px';
    },
    hide() { tip.classList.remove('is-visible'); },
  };
}

/** Legend — always present for >=2 series, absent for exactly one. */
function renderLegend(host, series, kind = 'line') {
  const old = host.querySelector('.legend');
  if (old) old.remove();
  if (series.length < 2) return;
  const wrap = document.createElement('div');
  wrap.className = 'legend';
  for (const s of series) {
    const item = document.createElement('span');
    item.className = 'legend-item';
    const key = document.createElement('span');
    key.className = 'legend-key' + (kind === 'rect' ? ' rect' : '');
    key.style.background = s.color;
    item.appendChild(key);
    const label = document.createElement('span');
    label.textContent = s.name;                // untrusted-safe
    item.appendChild(label);
    wrap.appendChild(item);
  }
  host.appendChild(wrap);
}

/** The table-view twin every chart ships (also the light-mode relief rule). */
function renderTableView(host, headers, rows) {
  let box = host.querySelector('.chart-table');
  if (!box) {
    box = document.createElement('div');
    box.className = 'chart-table table-wrap';
    host.appendChild(box);
  }
  box.textContent = '';
  const table = document.createElement('table');
  table.className = 'data';
  const thead = document.createElement('thead');
  const htr = document.createElement('tr');
  headers.forEach((h, i) => {
    const th = document.createElement('th');
    th.textContent = h;
    if (i > 0) th.className = 'num';
    htr.appendChild(th);
  });
  thead.appendChild(htr);
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (const r of rows) {
    const tr = document.createElement('tr');
    r.forEach((c, i) => {
      const td = document.createElement('td');
      td.textContent = c;
      if (i > 0) td.className = 'num';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  box.appendChild(table);
}

function chartFrame(host) {
  host.textContent = '';
  host.classList.add('chart');
  const body = document.createElement('div');
  body.className = 'chart-body';
  host.appendChild(body);
  return body;
}

/** Bar path: 4px rounded data-end, square at the baseline. */
function barPath(x, y, w, h, r, dir) {
  const rr = Math.max(0, Math.min(r, Math.min(w, h) / 2));
  if (h <= 0.5 || w <= 0.5) return '';
  switch (dir) {
    case 'up':    // column grows up from baseline at y+h
      return `M${x},${y + h} L${x},${y + rr} Q${x},${y} ${x + rr},${y} L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr} L${x + w},${y + h} Z`;
    case 'down':
      return `M${x},${y} L${x},${y + h - rr} Q${x},${y + h} ${x + rr},${y + h} L${x + w - rr},${y + h} Q${x + w},${y + h} ${x + w},${y + h - rr} L${x + w},${y} Z`;
    case 'right': // bar grows right from baseline at x
      return `M${x},${y} L${x + w - rr},${y} Q${x + w},${y} ${x + w},${y + rr} L${x + w},${y + h - rr} Q${x + w},${y + h} ${x + w - rr},${y + h} L${x},${y + h} Z`;
    case 'left':
      return `M${x + w},${y} L${x + rr},${y} Q${x},${y} ${x},${y + rr} L${x},${y + h - rr} Q${x},${y + h} ${x + rr},${y + h} L${x + w},${y + h} Z`;
  }
  return '';
}

/* ============================================================ line chart */
/**
 * spec: {
 *   x: string[]            shared x categories (dates)
 *   series: [{name, values:number[], color?, dashed?, emphasis?}]
 *   yFormat, yLabel, height, directLabelEnds, baselineAt
 * }
 */
export function lineChart(host, spec) {
  const body = chartFrame(host);
  const {
    x = [], series = [], height = 260,
    yFormat = (v) => fmt.num(v, 2),
    directLabelEnds = true,
    baselineAt = null,
    tooltipTitle = (i) => fmt.date(x[i]),
  } = spec;

  const W = Math.max(320, host.clientWidth || 640);
  const padL = 54, padR = directLabelEnds && series.length <= 4 ? 66 : 16, padT = 12, padB = 28;
  const plotW = W - padL - padR, plotH = height - padT - padB;

  const svg = el('svg', { viewBox: `0 0 ${W} ${height}`, height, role: 'img' }, body);

  const flat = series.flatMap(s => s.values).filter(v => v !== null && isFinite(v));
  if (!x.length || !flat.length) {
    text('text', { x: W / 2, y: height / 2, 'text-anchor': 'middle', class: 'tick-text' }, 'No data', svg);
    return;
  }
  let yMin = Math.min(...flat), yMax = Math.max(...flat);
  if (baselineAt !== null) { yMin = Math.min(yMin, baselineAt); yMax = Math.max(yMax, baselineAt); }
  const pad = (yMax - yMin) * 0.08 || Math.abs(yMax || 1) * 0.1;
  yMin -= pad; yMax += pad;

  const sx = (i) => padL + (x.length === 1 ? plotW / 2 : (i / (x.length - 1)) * plotW);
  const sy = (v) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  // recessive hairline grid
  for (const t of niceTicks(yMin, yMax, 5)) {
    const y = sy(t);
    el('line', { x1: padL, x2: padL + plotW, y1: y, y2: y, class: 'grid-line' }, svg);
    text('text', { x: padL - 9, y: y + 3.5, 'text-anchor': 'end', class: 'tick-text' }, yFormat(t), svg);
  }
  if (baselineAt !== null) {
    el('line', { x1: padL, x2: padL + plotW, y1: sy(baselineAt), y2: sy(baselineAt), class: 'zero-line' }, svg);
  }

  const nTicks = Math.min(6, x.length);
  const seen = new Set();
  for (let k = 0; k < nTicks; k++) {
    const i = Math.round((k / Math.max(1, nTicks - 1)) * (x.length - 1));
    if (seen.has(i)) continue;
    seen.add(i);
    text('text', {
      x: sx(i), y: height - 8,
      'text-anchor': k === 0 ? 'start' : k === nTicks - 1 ? 'end' : 'middle',
      class: 'tick-text',
    }, fmt.dateShort(x[i]), svg);
  }
  el('line', { x1: padL, x2: padL + plotW, y1: padT + plotH, y2: padT + plotH, class: 'axis-line' }, svg);

  const resolved = series.map((s, i) => ({ ...s, color: s.color || seriesColor(i) }));

  for (const s of resolved) {
    let d = '', pen = false;
    s.values.forEach((v, i) => {
      if (v === null || !isFinite(v)) { pen = false; return; }
      d += (pen ? 'L' : 'M') + sx(i) + ',' + sy(v);
      pen = true;
    });
    if (!d) continue;
    el('path', {
      d, class: 'series-line',
      stroke: s.emphasis === false ? 'var(--de-emphasis)' : s.color,
      'stroke-dasharray': s.dashed ? '5 4' : null,
      'stroke-width': s.emphasis === false ? 1.5 : 2,
    }, svg);
  }

  // selective direct labels: the end point only, and only when they can't collide
  if (directLabelEnds && resolved.length <= 4) {
    const ends = resolved.map(s => {
      for (let i = s.values.length - 1; i >= 0; i--) {
        if (s.values[i] !== null && isFinite(s.values[i])) return { s, i, v: s.values[i] };
      }
      return null;
    }).filter(Boolean).sort((a, b) => sy(a.v) - sy(b.v));

    // nudge only enough to avoid overlap; leader lines carry the connection
    const MIN_GAP = 13;
    const placed = [];
    ends.forEach(e => {
      let y = sy(e.v);
      if (placed.length && y - placed[placed.length - 1] < MIN_GAP) y = placed[placed.length - 1] + MIN_GAP;
      placed.push(y);
      const px = sx(e.i);
      el('circle', { cx: px, cy: sy(e.v), r: 4, fill: e.s.color, class: 'hover-dot' }, svg);
      if (Math.abs(y - sy(e.v)) > 1.5) {
        el('line', { x1: px + 5, y1: sy(e.v), x2: px + 11, y2: y, stroke: 'var(--axis)', 'stroke-width': 1 }, svg);
      }
      text('text', { x: px + 13, y: y + 3.5, class: 'end-label' }, yFormat(e.v), svg);
    });
  }

  // crosshair layer — reader aims at a date, never at a 2px line
  const cross = el('line', { class: 'crosshair', y1: padT, y2: padT + plotH, opacity: 0 }, svg);
  const dots = resolved.map(s => el('circle', { r: 4.5, fill: s.color, class: 'hover-dot', opacity: 0 }, svg));
  const hit = el('rect', { x: padL, y: padT, width: plotW, height: plotH, class: 'hit' }, svg);
  const tip = makeTooltip(host);

  function moveTo(clientX) {
    const rect = svg.getBoundingClientRect();
    const px = ((clientX - rect.left) / rect.width) * W;
    let i = Math.round(((px - padL) / plotW) * (x.length - 1));
    i = Math.max(0, Math.min(x.length - 1, i));
    const gx = sx(i);
    cross.setAttribute('x1', gx); cross.setAttribute('x2', gx); cross.setAttribute('opacity', 1);
    const rows = [];
    resolved.forEach((s, k) => {
      const v = s.values[i];
      if (v === null || v === undefined || !isFinite(v)) { dots[k].setAttribute('opacity', 0); return; }
      dots[k].setAttribute('cx', gx); dots[k].setAttribute('cy', sy(v)); dots[k].setAttribute('opacity', 1);
      rows.push({ color: s.color, name: s.name, value: yFormat(v) });
    });
    if (rows.length) tip.show(tooltipTitle(i), rows, gx, padT + plotH / 2);
    else tip.hide();
  }
  hit.addEventListener('pointermove', (e) => moveTo(e.clientX));
  hit.addEventListener('pointerleave', () => {
    cross.setAttribute('opacity', 0);
    dots.forEach(d => d.setAttribute('opacity', 0));
    tip.hide();
  });

  renderLegend(body, resolved, 'line');
  renderTableView(host, ['Date', ...resolved.map(s => s.name)],
    x.map((d, i) => [fmt.date(d), ...resolved.map(s => {
      const v = s.values[i];
      return (v === null || v === undefined || !isFinite(v)) ? '—' : yFormat(v);
    })]));
}

/* ============================================================= area chart */
export function areaChart(host, spec) {
  const body = chartFrame(host);
  const { x = [], values = [], height = 200, color = 'var(--series-8)',
          yFormat = (v) => fmt.pct(v, 1), name = 'Value' } = spec;

  const W = Math.max(320, host.clientWidth || 640);
  const padL = 54, padR = 16, padT = 12, padB = 28;
  const plotW = W - padL - padR, plotH = height - padT - padB;
  const svg = el('svg', { viewBox: `0 0 ${W} ${height}`, height }, body);

  const finite = values.filter(v => v !== null && isFinite(v));
  if (!finite.length) {
    text('text', { x: W / 2, y: height / 2, 'text-anchor': 'middle', class: 'tick-text' }, 'No data', svg);
    return;
  }
  const yMax = Math.max(0, ...finite), yMin = Math.min(0, ...finite);
  const span = (yMax - yMin) || 1;
  const sx = (i) => padL + (values.length === 1 ? plotW / 2 : (i / (values.length - 1)) * plotW);
  const sy = (v) => padT + plotH - ((v - yMin) / span) * plotH;

  for (const t of niceTicks(yMin, yMax, 4)) {
    const y = sy(t);
    el('line', { x1: padL, x2: padL + plotW, y1: y, y2: y, class: 'grid-line' }, svg);
    text('text', { x: padL - 9, y: y + 3.5, 'text-anchor': 'end', class: 'tick-text' }, yFormat(t), svg);
  }

  let d = '';
  values.forEach((v, i) => { if (v !== null && isFinite(v)) d += (d ? 'L' : 'M') + sx(i) + ',' + sy(v); });
  if (d) {
    const zero = sy(Math.max(yMin, Math.min(0, yMax)));
    el('path', { d: `${d} L${sx(values.length - 1)},${zero} L${sx(0)},${zero} Z`, fill: color, opacity: 0.10 }, svg);
    el('path', { d, class: 'series-line', stroke: color }, svg);
  }

  const nTicks = Math.min(6, x.length);
  for (let k = 0; k < nTicks; k++) {
    const i = Math.round((k / Math.max(1, nTicks - 1)) * (x.length - 1));
    text('text', {
      x: sx(i), y: height - 8,
      'text-anchor': k === 0 ? 'start' : k === nTicks - 1 ? 'end' : 'middle', class: 'tick-text',
    }, fmt.dateShort(x[i]), svg);
  }
  el('line', { x1: padL, x2: padL + plotW, y1: sy(0), y2: sy(0), class: 'axis-line' }, svg);

  const cross = el('line', { class: 'crosshair', y1: padT, y2: padT + plotH, opacity: 0 }, svg);
  const dot = el('circle', { r: 4.5, fill: color, class: 'hover-dot', opacity: 0 }, svg);
  const hit = el('rect', { x: padL, y: padT, width: plotW, height: plotH, class: 'hit' }, svg);
  const tip = makeTooltip(host);
  hit.addEventListener('pointermove', (e) => {
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let i = Math.round(((px - padL) / plotW) * (values.length - 1));
    i = Math.max(0, Math.min(values.length - 1, i));
    const v = values[i];
    if (v === null || !isFinite(v)) return;
    const gx = sx(i);
    cross.setAttribute('x1', gx); cross.setAttribute('x2', gx); cross.setAttribute('opacity', 1);
    dot.setAttribute('cx', gx); dot.setAttribute('cy', sy(v)); dot.setAttribute('opacity', 1);
    tip.show(fmt.date(x[i]), [{ color, name, value: yFormat(v) }], gx, padT + plotH / 2);
  });
  hit.addEventListener('pointerleave', () => {
    cross.setAttribute('opacity', 0); dot.setAttribute('opacity', 0); tip.hide();
  });

  renderTableView(host, ['Date', name], x.map((d2, i) => [fmt.date(d2), yFormat(values[i])]));
}

/* ============================================================== bar chart */
/**
 * spec: { labels, values, horizontal, height, color | colorFor(i),
 *         valueFormat, diverging, directLabels }
 * One series -> one color for every bar (never a value-ramp on nominal cats).
 */
export function barChart(host, spec) {
  const body = chartFrame(host);
  const {
    labels = [], values = [], horizontal = false, height = 260,
    color = 'var(--series-1)', colorFor = null,
    valueFormat = (v) => fmt.num(v, 2),
    diverging = false, directLabels = true, barName = 'Value',
  } = spec;

  const W = Math.max(320, host.clientWidth || 640);
  const svg0 = { GAP: 2, MAXW: 24, R: 4 };
  const finite = values.filter(v => v !== null && isFinite(v));
  if (!labels.length || !finite.length) {
    const svg = el('svg', { viewBox: `0 0 ${W} ${height}`, height }, body);
    text('text', { x: W / 2, y: height / 2, 'text-anchor': 'middle', class: 'tick-text' }, 'No data', svg);
    return;
  }

  const tip = makeTooltip(host);

  if (horizontal) {
    const rowH = 26;
    const H = Math.max(height, labels.length * rowH + 34);
    const padL = 118, padR = 58, padT = 8, padB = 26;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H }, body);

    const vMax = Math.max(0, ...finite), vMin = Math.min(0, ...finite);
    const lo = diverging ? Math.min(vMin, -Math.abs(vMax)) : Math.min(0, vMin);
    const hi = diverging ? Math.max(vMax, Math.abs(vMin)) : Math.max(0, vMax);
    const span = (hi - lo) || 1;
    const sx = (v) => padL + ((v - lo) / span) * plotW;
    const zero = sx(0);

    for (const t of niceTicks(lo, hi, 4)) {
      const gx = sx(t);
      el('line', { x1: gx, x2: gx, y1: padT, y2: padT + plotH, class: 'grid-line' }, svg);
      text('text', { x: gx, y: H - 8, 'text-anchor': 'middle', class: 'tick-text' }, valueFormat(t), svg);
    }
    el('line', { x1: zero, x2: zero, y1: padT, y2: padT + plotH, class: 'axis-line' }, svg);

    const band = plotH / labels.length;
    const bh = Math.min(svg0.MAXW, band - svg0.GAP * 2);

    labels.forEach((lab, i) => {
      const v = values[i];
      if (v === null || !isFinite(v)) return;
      const y = padT + i * band + (band - bh) / 2;
      const c = colorFor ? colorFor(i, v) : (diverging ? (v >= 0 ? 'var(--div-pos)' : 'var(--div-neg)') : color);
      const x0 = Math.min(zero, sx(v)), w = Math.abs(sx(v) - zero);
      el('path', { d: barPath(x0, y, w, bh, svg0.R, v >= 0 ? 'right' : 'left'), fill: c, class: 'mark' }, svg);

      text('text', { x: padL - 10, y: y + bh / 2 + 4, 'text-anchor': 'end', class: 'tick-text' }, lab, svg);

      // label outside the bar end — never clipped inside a short bar
      if (directLabels) {
        const tx = v >= 0 ? sx(v) + 7 : sx(v) - 7;
        text('text', { x: tx, y: y + bh / 2 + 4, 'text-anchor': v >= 0 ? 'start' : 'end', class: 'value-label' },
          valueFormat(v), svg);
      }

      // hit target spans the whole band (>=24px), not just the painted bar
      const hitR = el('rect', { x: padL, y: padT + i * band, width: plotW, height: band, class: 'bar-hit' }, svg);
      hitR.addEventListener('pointermove', () =>
        tip.show(lab, [{ color: c, name: barName, value: valueFormat(v) }], sx(v), y + bh / 2));
      hitR.addEventListener('pointerleave', () => tip.hide());
    });
  } else {
    const padL = 52, padR = 14, padT = 12, padB = 38;
    const plotW = W - padL - padR, plotH = height - padT - padB;
    const svg = el('svg', { viewBox: `0 0 ${W} ${height}`, height }, body);

    const vMax = Math.max(0, ...finite), vMin = Math.min(0, ...finite);
    const lo = diverging ? Math.min(vMin, -Math.abs(vMax)) : Math.min(0, vMin);
    const hi = diverging ? Math.max(vMax, Math.abs(vMin)) : Math.max(0, vMax) * 1.06;
    const span = (hi - lo) || 1;
    const sy = (v) => padT + plotH - ((v - lo) / span) * plotH;
    const zero = sy(0);

    for (const t of niceTicks(lo, hi, 5)) {
      const y = sy(t);
      el('line', { x1: padL, x2: padL + plotW, y1: y, y2: y, class: 'grid-line' }, svg);
      text('text', { x: padL - 9, y: y + 3.5, 'text-anchor': 'end', class: 'tick-text' }, valueFormat(t), svg);
    }
    el('line', { x1: padL, x2: padL + plotW, y1: zero, y2: zero, class: 'axis-line' }, svg);

    const band = plotW / labels.length;
    const bw = Math.min(svg0.MAXW, band - svg0.GAP * 2);

    labels.forEach((lab, i) => {
      const v = values[i];
      if (v === null || !isFinite(v)) return;
      const x = padL + i * band + (band - bw) / 2;
      const c = colorFor ? colorFor(i, v) : (diverging ? (v >= 0 ? 'var(--div-pos)' : 'var(--div-neg)') : color);
      const y0 = Math.min(zero, sy(v)), h = Math.abs(sy(v) - zero);
      el('path', { d: barPath(x, y0, bw, h, svg0.R, v >= 0 ? 'up' : 'down'), fill: c, class: 'mark' }, svg);

      if (labels.length <= 16) {
        text('text', { x: x + bw / 2, y: height - 20, 'text-anchor': 'middle', class: 'tick-text' }, lab, svg);
      }
      if (directLabels && labels.length <= 14) {
        text('text', {
          x: x + bw / 2, y: v >= 0 ? y0 - 6 : y0 + h + 12,
          'text-anchor': 'middle', class: 'value-label',
        }, valueFormat(v), svg);
      }
      const hitR = el('rect', { x: padL + i * band, y: padT, width: band, height: plotH, class: 'bar-hit' }, svg);
      hitR.addEventListener('pointermove', () =>
        tip.show(lab, [{ color: c, name: barName, value: valueFormat(v) }], x + bw / 2, y0));
      hitR.addEventListener('pointerleave', () => tip.hide());
    });
  }

  renderTableView(host, ['Category', barName], labels.map((l, i) => [l, valueFormat(values[i])]));
}

/* ================================================================ heatmap */
/** Sequential single-hue ramp (blue, light->dark). Per-cell hover. */
export function heatmap(host, spec) {
  const body = chartFrame(host);
  const { rows = [], cols = [], matrix = [], valueFormat = (v) => fmt.num(v, 2), diverging = true } = spec;

  const W = Math.max(320, host.clientWidth || 640);
  const padL = 108, padT = 62, padR = 10, padB = 10;
  const cell = Math.max(16, Math.min(38, (W - padL - padR) / Math.max(1, cols.length)));
  const H = padT + rows.length * cell + padB;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, height: H }, body);
  const tip = makeTooltip(host);

  const flat = matrix.flat().filter(v => v !== null && isFinite(v));
  if (!flat.length) {
    text('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle', class: 'tick-text' }, 'No data', svg);
    return;
  }
  const absMax = Math.max(...flat.map(Math.abs)) || 1;
  const vMin = Math.min(...flat), vMax = Math.max(...flat);

  const SEQ = ['--seq-100', '--seq-200', '--seq-300', '--seq-400', '--seq-500', '--seq-600', '--seq-700'];
  function colorOf(v) {
    if (v === null || !isFinite(v)) return 'var(--surface-3)';
    if (diverging) {
      // two hues + neutral gray midpoint
      const t = Math.abs(v) / absMax;
      if (t < 0.06) return 'var(--div-mid)';
      const pole = v >= 0 ? 'var(--div-pos)' : 'var(--div-neg)';
      return `color-mix(in srgb, ${pole} ${Math.round(18 + t * 82)}%, var(--div-mid))`;
    }
    const t = (v - vMin) / ((vMax - vMin) || 1);
    return `var(${SEQ[Math.min(SEQ.length - 1, Math.floor(t * SEQ.length))]})`;
  }

  cols.forEach((c, j) => {
    const x = padL + j * cell + cell / 2;
    const t = text('text', { x, y: padT - 9, 'text-anchor': 'start', class: 'tick-text' }, c, svg);
    t.setAttribute('transform', `rotate(-52 ${x} ${padT - 9})`);
  });

  rows.forEach((r, i) => {
    text('text', { x: padL - 9, y: padT + i * cell + cell / 2 + 3.5, 'text-anchor': 'end', class: 'tick-text' }, r, svg);
    cols.forEach((c, j) => {
      const v = matrix[i] ? matrix[i][j] : null;
      // 2px surface gap does the separating — never a stroke around the mark
      const rect = el('rect', {
        x: padL + j * cell + 1, y: padT + i * cell + 1,
        width: cell - 2, height: cell - 2, rx: 2,
        fill: colorOf(v), class: 'mark',
      }, svg);
      rect.addEventListener('pointermove', () =>
        tip.show(`${r} · ${c}`, [{ name: 'Correlation', value: valueFormat(v) }],
          padL + j * cell + cell / 2, padT + i * cell + cell / 2));
      rect.addEventListener('pointerleave', () => tip.hide());
    });
  });

  renderTableView(host, ['', ...cols], rows.map((r, i) => [r, ...cols.map((c, j) => valueFormat(matrix[i]?.[j]))]));
}

/* ============================================================== sparkline */
export function sparkline(host, values, opts = {}) {
  host.textContent = '';
  const { height = 30, color = 'var(--series-1)', emphasisLast = true } = opts;
  const W = Math.max(60, host.clientWidth || 120);
  const svg = el('svg', { viewBox: `0 0 ${W} ${height}`, height, preserveAspectRatio: 'none' }, host);
  const finite = values.filter(v => v !== null && isFinite(v));
  if (finite.length < 2) return;
  const lo = Math.min(...finite), hi = Math.max(...finite), span = (hi - lo) || 1;
  const sx = (i) => (i / (values.length - 1)) * (W - 4) + 2;
  const sy = (v) => height - 3 - ((v - lo) / span) * (height - 6);
  let d = '';
  values.forEach((v, i) => { if (v !== null && isFinite(v)) d += (d ? 'L' : 'M') + sx(i) + ',' + sy(v); });
  // de-emphasis hue for the run, accent for the current period
  el('path', { d, fill: 'none', stroke: 'var(--de-emphasis)', 'stroke-width': 1.5, 'stroke-linejoin': 'round' }, svg);
  if (emphasisLast && values.length > 1) {
    const n = Math.max(2, Math.floor(values.length * 0.25));
    let d2 = '';
    values.slice(-n).forEach((v, k) => {
      const i = values.length - n + k;
      if (v !== null && isFinite(v)) d2 += (d2 ? 'L' : 'M') + sx(i) + ',' + sy(v);
    });
    el('path', { d: d2, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
    const last = values[values.length - 1];
    if (last !== null && isFinite(last)) {
      el('circle', { cx: sx(values.length - 1), cy: sy(last), r: 2.5, fill: color }, svg);
    }
  }
}

/* ============ chart/table toggle wiring (the table-view twin, per card) === */
export function attachChartToggle(cardHead, chartHost) {
  const wrap = document.createElement('div');
  wrap.className = 'chart-toggle';
  const bChart = document.createElement('button');
  bChart.textContent = 'Chart'; bChart.className = 'is-active';
  const bTable = document.createElement('button');
  bTable.textContent = 'Table';
  wrap.append(bChart, bTable);
  cardHead.appendChild(wrap);

  bChart.addEventListener('click', () => {
    bChart.classList.add('is-active'); bTable.classList.remove('is-active');
    chartHost.querySelector('.chart-body')?.classList.remove('is-hidden');
    chartHost.querySelector('.chart-table')?.classList.remove('is-active');
    chartHost.querySelector('.legend')?.style.removeProperty('display');
  });
  bTable.addEventListener('click', () => {
    bTable.classList.add('is-active'); bChart.classList.remove('is-active');
    chartHost.querySelector('.chart-body')?.classList.add('is-hidden');
    chartHost.querySelector('.chart-table')?.classList.add('is-active');
  });
}

/** Re-render charts on resize without a layout jump. */
export function onResize(fn) {
  let t = null;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(fn, 140); });
}
