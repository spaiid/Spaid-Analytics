/* ==========================================================================
   app.js — Spaid Analytics dashboard.
   Renders whatever factor groups the API returns; nothing here hard-codes
   the model's internals, so the research layer can evolve independently.
   ========================================================================== */

import {
  lineChart, areaChart, barChart, heatmap, sparkline,
  attachChartToggle, seriesColor, onResize, fmt,
} from './charts.js';

/* ------------------------------------------------------------------ state */
const state = {
  view: 'picks',
  status: null,
  picks: null,
  screener: null,
  signals: null,
  backtest: null,
  health: null,
  ticker: null,
  tickerSym: null,
  filters: { limit: 25, sector: 'all', rating: 'all' },
  backtestOpts: { topn: 20, cost_bps: 5, long_short: false },
  sortBy: { key: 'rank', dir: 1 },
};

const VIEW_TITLES = {
  picks: 'Today’s picks',
  screener: 'Screener',
  ticker: 'Ticker detail',
  signals: 'Signal quality',
  backtest: 'Backtest',
  data: 'Data health',
};

/* -------------------------------------------------------------------- api */
async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* keep */ }
    throw new Error(detail);
  }
  return res.json();
}

/* ----------------------------------------------------------------- helpers */
function h(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;          // untrusted-safe
    else if (k === 'html') n.innerHTML = v;            // only for literal icon markup
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
    else n.setAttribute(k, String(v));
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    n.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
  }
  return n;
}

function card(title, sub, ...body) {
  const head = h('div', { class: 'card-head' }, h('h3', { class: 'card-title', text: title }));
  const c = h('div', { class: 'card' }, head);
  if (sub) c.appendChild(h('p', { class: 'card-sub', text: sub }));
  body.flat().forEach(b => b && c.appendChild(b));
  c._head = head;
  return c;
}

function statTile(label, value, opts = {}) {
  const kids = [
    h('div', { class: 'stat-label' }, opts.dot ? h('span', { class: `dot dot-${opts.dot}` }) : null, label),
    h('div', { class: 'stat-value' + (opts.small ? ' is-small' : ''), text: value }),
  ];
  if (opts.delta !== undefined && opts.delta !== null) {
    const dir = opts.delta > 0 ? 'up' : opts.delta < 0 ? 'down' : 'flat';
    kids.push(h('div', { class: 'stat-foot' },
      h('span', { class: `stat-delta ${dir}`, text: opts.deltaText || fmt.signedPct(opts.delta) }),
      opts.deltaLabel ? ' ' + opts.deltaLabel : ''));
  } else if (opts.foot) {
    kids.push(h('div', { class: 'stat-foot', text: opts.foot }));
  }
  const tile = h('div', { class: 'stat' }, ...kids);
  if (opts.spark && opts.spark.length > 1) {
    const sp = h('div', { class: 'stat-spark' });
    tile.appendChild(sp);
    requestAnimationFrame(() => sparkline(sp, opts.spark, { color: opts.sparkColor || 'var(--series-1)' }));
  }
  return tile;
}

const RATING_STYLE = {
  buy:    { pill: 'pill-good',     icon: '▲', label: 'Buy' },
  accumulate: { pill: 'pill-good', icon: '▲', label: 'Accumulate' },
  hold:   { pill: 'pill-neutral',  icon: '■', label: 'Hold' },
  reduce: { pill: 'pill-serious',  icon: '▼', label: 'Reduce' },
  avoid:  { pill: 'pill-critical', icon: '▼', label: 'Avoid' },
};

/** Status colors always ship with an icon + label, never color alone. */
function ratingPill(rating) {
  const key = String(rating || 'hold').toLowerCase();
  const s = RATING_STYLE[key] || RATING_STYLE.hold;
  return h('span', { class: `pill ${s.pill}` },
    h('span', { text: s.icon, style: { fontSize: '9px' } }), s.label);
}

function meter(pct, color) {
  return h('div', { class: 'meter' },
    h('div', { class: 'meter-fill', style: { width: Math.max(2, Math.min(100, pct)) + '%', background: color || 'var(--seq-400)' } }));
}

function empty(title, msg, actionLabel, action) {
  return h('div', { class: 'empty' },
    h('div', { class: 'empty-title', text: title }),
    h('div', { text: msg }),
    actionLabel ? h('button', { class: 'btn btn-primary', onclick: action, text: actionLabel }) : null);
}

function groupColorMap(groups) {
  const map = new Map();
  groups.forEach((g, i) => map.set(g, seriesColor(i)));   // colour follows the entity
  return map;
}

/* --------------------------------------------------- attribution renderer */
/** Diverging bars: what pushed this name up or down, with magnitudes. */
function attribution(contributions, opts = {}) {
  const rows = (contributions || []).slice(0, opts.limit || 8);
  if (!rows.length) return h('div', { class: 'muted', text: 'No attribution available.' });
  const maxAbs = Math.max(...rows.map(r => Math.abs(r.contribution)), 1e-9);
  const wrap = h('div', { class: 'attrib' });
  for (const r of rows) {
    const frac = Math.abs(r.contribution) / maxAbs;
    const pos = r.contribution >= 0;
    const bar = h('div', {
      class: `attrib-bar ${pos ? 'pos' : 'neg'}`,
      style: pos
        ? { left: '50%', width: (frac * 50) + '%' }
        : { right: '50%', width: (frac * 50) + '%' },
    });
    wrap.appendChild(h('div', { class: 'attrib-row' },
      h('div', { class: 'attrib-name', text: r.label || r.group }),
      h('div', { class: 'attrib-track' }, h('div', { class: 'attrib-axis' }), bar),
      h('div', { class: 'attrib-val', text: fmt.signed(r.contribution, 2) })));
  }
  return wrap;
}

/* ================================================================= views */

/* ------------------------------------------------------------ picks view */
function renderPicks(root) {
  const d = state.picks;
  if (!d || !d.picks) {
    root.replaceChildren(card('Today’s picks', null,
      empty('No model output yet',
        'Fetch data and train the model to produce a ranked buy list.',
        'Run pipeline', runPipeline)));
    return;
  }

  const m = d.model || {};
  const regime = d.regime || {};
  const picks = d.picks;
  const buys = picks.filter(p => ['buy', 'accumulate'].includes(String(p.rating).toLowerCase()));

  // Hero: the one number the dashboard leads with.
  const hero = h('div', { class: 'card' },
    h('div', { class: 'hero' },
      h('div', { class: 'hero-label', text: `Buy-rated names · ${fmt.date(d.asof)}` }),
      h('div', { class: 'hero-figure', text: String(buys.length) }),
      h('div', { class: 'stat-foot', text: `of ${d.universe_size} in universe · ${d.horizon_days}-day horizon` })));

  const icDot = m.rank_ic_oos > 0.03 ? 'good' : m.rank_ic_oos > 0.01 ? 'warning' : 'critical';
  const kpis = h('div', { class: 'grid grid-4' },
    statTile('Out-of-sample rank IC', fmt.num(m.rank_ic_oos, 3), {
      dot: icDot,
      foot: m.rank_ic_t !== undefined ? `t = ${fmt.num(m.rank_ic_t, 1)} over ${m.n_folds ?? '—'} folds` : null,
      spark: m.ic_history || null,
    }),
    statTile('Top-decile spread', fmt.signedPct(m.decile_spread), {
      foot: 'Top minus bottom decile, annualised',
    }),
    statTile('Market regime', regime.label || '—', {
      small: true, foot: regime.detail || null,
    }),
    statTile('Model trained through', fmt.date(m.trained_through), {
      small: true, foot: m.n_train ? `${fmt.compact(m.n_train)} training rows` : null,
    }));

  // Honest framing — the app should not overstate its edge.
  const caveat = h('div', { class: 'note' },
    h('strong', { text: 'Read this as a ranking, not a forecast. ' }),
    `Scores order names by expected relative strength over the next ${d.horizon_days} trading days. `,
    `Out-of-sample rank IC of ${fmt.num(m.rank_ic_oos, 3)} means the ordering is `,
    m.rank_ic_oos > 0.02 ? 'weakly informative' : 'barely distinguishable from noise',
    ` — edge shows up across many names over time, never in any single pick.`);

  const table = picksTable(picks);

  const dist = h('div', { class: 'chart' });
  const distCard = card('Score distribution', 'Where the universe sits on the composite score.', dist);
  attachChartToggle(distCard._head, dist);

  const sectorHost = h('div', { class: 'chart' });
  const sectorCard = card('Average score by sector', 'Which corners of the market the model favours right now.', sectorHost);
  attachChartToggle(sectorCard._head, sectorHost);

  root.replaceChildren(
    h('div', { class: 'grid grid-2', style: { marginBottom: '14px' } }, hero, h('div', { class: 'stack' }, caveat)),
    kpis,
    h('div', { style: { height: '14px' } }),
    filterBar(),
    card(`Ranked candidates`, `Click a row to see which data points drove the call.`, table),
    h('div', { style: { height: '14px' } }),
    h('div', { class: 'grid grid-2' }, distCard, sectorCard));

  requestAnimationFrame(() => {
    if (d.distribution) {
      barChart(dist, {
        labels: d.distribution.labels, values: d.distribution.counts,
        height: 220, color: 'var(--series-1)', barName: 'Names',
        valueFormat: (v) => fmt.num(v, 0),
      });
    }
    if (d.sector_scores) {
      barChart(sectorHost, {
        labels: d.sector_scores.map(s => s.sector),
        values: d.sector_scores.map(s => s.mean_score),
        horizontal: true, diverging: true, height: 260, barName: 'Mean score',
        valueFormat: (v) => fmt.num(v, 2),
      });
    }
  });
}

function filterBar() {
  const d = state.picks || {};
  const sectors = ['all', ...new Set((d.picks || []).map(p => p.sector).filter(Boolean))].sort();
  const sectorSel = h('select', {
    onchange: (e) => { state.filters.sector = e.target.value; rerender(); },
  }, ...sectors.map(s => h('option', { value: s, text: s === 'all' ? 'All sectors' : s, selected: state.filters.sector === s })));

  const ratingSel = h('select', {
    onchange: (e) => { state.filters.rating = e.target.value; rerender(); },
  }, ...['all', 'buy', 'accumulate', 'hold', 'reduce', 'avoid'].map(r =>
    h('option', { value: r, text: r === 'all' ? 'All ratings' : RATING_STYLE[r].label, selected: state.filters.rating === r })));

  const seg = h('div', { class: 'segmented' }, ...[10, 25, 50, 100].map(n =>
    h('button', {
      class: state.filters.limit === n ? 'is-active' : '',
      text: String(n),
      onclick: () => { state.filters.limit = n; rerender(); },
    })));

  return h('div', { class: 'filterbar' },
    h('label', { class: 'control' }, h('span', { class: 'control-label', text: 'Sector' }), sectorSel),
    h('label', { class: 'control' }, h('span', { class: 'control-label', text: 'Rating' }), ratingSel),
    h('div', { class: 'control-label', text: 'Show' }), seg);
}

function applyFilters(picks) {
  let out = picks;
  if (state.filters.sector !== 'all') out = out.filter(p => p.sector === state.filters.sector);
  if (state.filters.rating !== 'all') out = out.filter(p => String(p.rating).toLowerCase() === state.filters.rating);
  return out.slice(0, state.filters.limit);
}

function picksTable(allPicks) {
  const picks = applyFilters(allPicks);
  const wrap = h('div', { class: 'table-wrap' });
  const tbody = h('tbody');

  const cols = [
    { key: 'rank', label: '#', num: false },
    { key: 'ticker', label: 'Name', num: false },
    { key: 'sector', label: 'Sector', num: false },
    { key: 'rating', label: 'Rating', num: false },
    { key: 'score', label: 'Score', num: true },
    { key: 'percentile', label: 'Pctile', num: true },
    { key: 'price', label: 'Price', num: true },
    { key: 'chg_20d', label: '20d', num: true },
    { key: 'top_driver', label: 'Lead driver', num: false },
  ];

  const thead = h('thead', h('tr', ...cols.map(c =>
    h('th', { class: (c.num ? 'num ' : '') + 'sortable', onclick: () => sortPicks(c.key) },
      c.label,
      state.sortBy.key === c.key ? h('span', { class: 'sort-caret', text: state.sortBy.dir > 0 ? '▲' : '▼' }) : null))));

  for (const p of picks) {
    const row = h('tr', { class: 'clickable' },
      h('td', h('span', { class: 'rank-badge', text: String(p.rank) })),
      h('td', h('div', { class: 'ticker-cell' },
        h('span', { class: 'ticker-sym', text: p.ticker }),
        h('span', { class: 'ticker-name', text: p.name || '' }))),
      h('td', h('span', { class: 'tag', text: p.sector || '—' })),
      h('td', ratingPill(p.rating)),
      h('td', { class: 'num' }, h('div', { class: 'score-cell' },
        h('span', { class: 'score-num', text: fmt.num(p.score, 2) }),
        meter((p.percentile ?? 0) * 100, p.score >= 0 ? 'var(--seq-400)' : 'var(--de-emphasis)'))),
      h('td', { class: 'num', text: fmt.num((p.percentile ?? 0) * 100, 0) }),
      h('td', { class: 'num', text: p.price ? fmt.num(p.price, 2) : '—' }),
      h('td', { class: 'num' }, h('span', {
        class: p.chg_20d >= 0 ? 'stat-delta up' : 'stat-delta down',
        text: fmt.signedPct(p.chg_20d),
      })),
      h('td', { class: 'muted', text: p.top_driver || (p.contributions?.[0]?.label ?? '—') }));

    const detail = h('tr', { class: 'expand-panel', style: { display: 'none' } });
    const detailCell = h('td', { colspan: String(cols.length) });
    detail.appendChild(detailCell);

    let built = false;
    row.addEventListener('click', () => {
      const open = detail.style.display !== 'none';
      detail.style.display = open ? 'none' : '';
      row.classList.toggle('is-open', !open);
      if (!built && !open) {
        built = true;
        detailCell.replaceChildren(buildPickDetail(p));
      }
    });

    tbody.append(row, detail);
  }

  if (!picks.length) {
    tbody.appendChild(h('tr', h('td', { colspan: String(cols.length) },
      empty('Nothing matches', 'Loosen the filters above to see more names.'))));
  }

  wrap.appendChild(h('table', { class: 'data' }, thead, tbody));
  return wrap;
}

function buildPickDetail(p) {
  const left = h('div', {},
    h('h4', { class: 'expand-title', text: `Why ${p.ticker} ranks here` }),
    attribution(p.contributions, { limit: 9 }),
    h('div', { class: 'stat-foot', style: { marginTop: '10px' } },
      'Contribution to the composite score, in score points. Positive pushes the name up the ranking.'));

  const facts = h('div', {},
    h('h4', { class: 'expand-title', text: 'Snapshot' }),
    h('table', { class: 'data' }, h('tbody',
      ...[
        ['Composite score', fmt.num(p.score, 3)],
        ['Universe percentile', fmt.num((p.percentile ?? 0) * 100, 0) + 'th'],
        ['Expected excess (model)', p.expected_excess !== undefined && p.expected_excess !== null ? fmt.signedPct(p.expected_excess) : '—'],
        ['Agreement across groups', p.agreement !== undefined && p.agreement !== null ? fmt.pct(p.agreement, 0) : '—'],
        ['Sector', p.sector || '—'],
        ['Last close', p.price ? fmt.num(p.price, 2) : '—'],
        ['1-day', fmt.signedPct(p.chg_1d)],
        ['20-day', fmt.signedPct(p.chg_20d)],
      ].map(([k, v]) => h('tr', h('td', { class: 'muted', text: k }), h('td', { class: 'num strong', text: v }))))),
    h('button', {
      class: 'btn', style: { marginTop: '12px' },
      text: 'Open full detail →',
      onclick: (e) => { e.stopPropagation(); openTicker(p.ticker); },
    }));

  return h('div', { class: 'expand-grid' }, left, facts);
}

function sortPicks(key) {
  const s = state.sortBy;
  s.dir = s.key === key ? -s.dir : (key === 'rank' ? 1 : -1);
  s.key = key;
  const arr = state.picks.picks;
  arr.sort((a, b) => {
    const av = a[key], bv = b[key];
    if (av === bv) return 0;
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    return (av > bv ? 1 : -1) * s.dir;
  });
  rerender();
}

/* --------------------------------------------------------- screener view */
function renderScreener(root) {
  const d = state.picks;
  if (!d || !d.picks) {
    root.replaceChildren(card('Screener', null, empty('No model output yet', 'Run the pipeline first.', 'Run pipeline', runPipeline)));
    return;
  }
  const search = h('input', {
    class: 'search-input', placeholder: 'Filter by ticker or name…',
    oninput: (e) => {
      const q = e.target.value.trim().toLowerCase();
      const rows = root.querySelectorAll('tbody tr');
      rows.forEach(r => {
        const t = r.getAttribute('data-search') || '';
        r.style.display = !q || t.includes(q) ? '' : 'none';
      });
    },
  });

  const tbody = h('tbody');
  const full = [...d.picks].sort((a, b) => a.rank - b.rank);
  for (const p of full) {
    tbody.appendChild(h('tr', {
      class: 'clickable',
      'data-search': `${p.ticker} ${p.name || ''}`.toLowerCase(),
      onclick: () => openTicker(p.ticker),
    },
      h('td', { class: 'num muted', text: String(p.rank) }),
      h('td', h('div', { class: 'ticker-cell' },
        h('span', { class: 'ticker-sym', text: p.ticker }),
        h('span', { class: 'ticker-name', text: p.name || '' }))),
      h('td', h('span', { class: 'tag', text: p.sector || '—' })),
      h('td', ratingPill(p.rating)),
      h('td', { class: 'num strong', text: fmt.num(p.score, 2) }),
      ...(d.groups || []).map(g => {
        const c = (p.contributions || []).find(x => x.group === g);
        return h('td', { class: 'num', text: c ? fmt.signed(c.contribution, 2) : '—' });
      }),
      h('td', { class: 'num', text: p.price ? fmt.num(p.price, 2) : '—' })));
  }

  const table = h('table', { class: 'data' },
    h('thead', h('tr',
      h('th', { class: 'num', text: '#' }),
      h('th', { text: 'Name' }),
      h('th', { text: 'Sector' }),
      h('th', { text: 'Rating' }),
      h('th', { class: 'num', text: 'Score' }),
      ...(d.groups || []).map(g => h('th', { class: 'num', text: g })),
      h('th', { class: 'num', text: 'Price' }))),
    tbody);

  root.replaceChildren(
    h('div', { class: 'filterbar' }, search,
      h('span', { class: 'control-label', text: `${full.length} names · click any row for detail` })),
    card('Full universe', 'Every name the model scored, with each factor group’s contribution.',
      h('div', { class: 'table-wrap', style: { maxHeight: '70vh' } }, table)));
}

/* ----------------------------------------------------------- ticker view */
async function openTicker(sym) {
  state.tickerSym = sym;
  switchView('ticker');
  const root = document.getElementById('view-ticker');
  root.replaceChildren(card('Loading…', null, h('div', { class: 'empty' }, h('span', { class: 'spinner' }))));
  try {
    state.ticker = await api(`/api/ticker/${encodeURIComponent(sym)}`);
    renderTicker(root);
  } catch (e) {
    root.replaceChildren(card('Ticker detail', null, empty('Could not load', e.message)));
  }
}

function renderTicker(root) {
  const t = state.ticker;
  if (!t) {
    root.replaceChildren(card('Ticker detail', null,
      empty('No ticker selected', 'Pick a name from Today’s picks or the Screener.')));
    return;
  }

  const head = h('div', { class: 'card' },
    h('div', { style: { display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' } },
      h('div', {},
        h('div', { style: { display: 'flex', alignItems: 'center', gap: '10px' } },
          h('span', { style: { fontSize: '22px', fontWeight: '600', letterSpacing: '-0.02em' }, text: t.ticker }),
          ratingPill(t.rating)),
        h('div', { class: 'muted', text: `${t.name || ''}${t.sector ? ' · ' + t.sector : ''}` })),
      h('div', { style: { flex: '1' } }),
      h('div', { class: 'grid grid-3', style: { minWidth: '380px', gap: '10px' } },
        statTile('Composite score', fmt.num(t.score, 2), { small: true, foot: `${fmt.num((t.percentile ?? 0) * 100, 0)}th percentile` }),
        statTile('Rank', t.rank ? `#${t.rank}` : '—', { small: true, foot: t.universe_size ? `of ${t.universe_size}` : null }),
        statTile('Last close', t.price ? fmt.num(t.price, 2) : '—', { small: true, delta: t.chg_1d, deltaLabel: 'today' }))));

  const priceHost = h('div', { class: 'chart' });
  const priceCard = card('Price', 'Adjusted close.', priceHost);
  attachChartToggle(priceCard._head, priceHost);

  const scoreHost = h('div', { class: 'chart' });
  const scoreCard = card('Score history', 'How the model’s view of this name has evolved.', scoreHost);
  attachChartToggle(scoreCard._head, scoreHost);

  const attribCard = card('What drives the score today', 'Contribution to the composite, in score points.',
    attribution(t.contributions, { limit: 12 }));

  const featRows = (t.features || []).map(f => h('tr',
    h('td', { class: 'muted', text: f.group || '' }),
    h('td', { text: f.label || f.name }),
    h('td', { class: 'num', text: f.value === null || f.value === undefined ? '—' : fmt.num(f.value, 3) }),
    h('td', { class: 'num strong', text: f.z === null || f.z === undefined ? '—' : fmt.signed(f.z, 2) }),
    h('td', { class: 'num' }, meter((f.percentile ?? 0) * 100))));

  const featCard = card('Feature detail', 'Raw value, cross-sectional z-score, and percentile within the universe.',
    h('div', { class: 'table-wrap', style: { maxHeight: '420px' } },
      h('table', { class: 'data' },
        h('thead', h('tr',
          h('th', { text: 'Group' }), h('th', { text: 'Feature' }),
          h('th', { class: 'num', text: 'Value' }), h('th', { class: 'num', text: 'Z' }),
          h('th', { class: 'num', text: 'Percentile' }))),
        h('tbody', ...featRows))));

  const funds = (t.fundamentals || []).length
    ? card('Fundamentals', 'As reported, keyed to the SEC filing date the market actually saw.',
        h('div', { class: 'table-wrap' }, h('table', { class: 'data' },
          h('thead', h('tr', h('th', { text: 'Metric' }), h('th', { class: 'num', text: 'Value' }), h('th', { class: 'num', text: 'Filed' }))),
          h('tbody', ...t.fundamentals.map(f => h('tr',
            h('td', { class: 'muted', text: f.label }),
            h('td', { class: 'num strong', text: f.display || fmt.compact(f.value) }),
            h('td', { class: 'num muted', text: fmt.date(f.filed) })))))))
    : null;

  root.replaceChildren(
    head, h('div', { style: { height: '14px' } }),
    h('div', { class: 'grid grid-2' }, priceCard, scoreCard),
    h('div', { style: { height: '14px' } }),
    h('div', { class: 'grid grid-2' }, attribCard, featCard),
    funds ? h('div', { style: { height: '14px' } }) : null, funds);

  requestAnimationFrame(() => {
    if (t.price_history) {
      lineChart(priceHost, {
        x: t.price_history.dates,
        series: [{ name: t.ticker, values: t.price_history.close }],
        height: 240, yFormat: (v) => fmt.num(v, 0),
      });
    }
    if (t.score_history) {
      lineChart(scoreHost, {
        x: t.score_history.dates,
        series: [{ name: 'Composite score', values: t.score_history.score }],
        height: 240, baselineAt: 0, yFormat: (v) => fmt.num(v, 1),
      });
    }
  });
}

/* ---------------------------------------------------------- signals view */
function renderSignals(root) {
  const d = state.signals;
  if (!d || !d.features) {
    root.replaceChildren(card('Signal quality', null,
      empty('No diagnostics yet', 'Run the pipeline to evaluate each signal out of sample.', 'Run pipeline', runPipeline)));
    return;
  }

  const feats = [...d.features].sort((a, b) => Math.abs(b.ic_mean) - Math.abs(a.ic_mean));
  const icHost = h('div', { class: 'chart' });
  const icCard = card('Information coefficient by feature',
    'Mean cross-sectional rank correlation with forward excess return, measured out of sample.', icHost);
  attachChartToggle(icCard._head, icHost);

  const decayHost = h('div', { class: 'chart' });
  const decayCard = card('Signal decay', 'How predictive the composite stays as the holding period extends.', decayHost);
  attachChartToggle(decayCard._head, decayHost);

  const corrHost = h('div', { class: 'chart' });
  const corrCard = card('Factor group correlation', 'Groups that move together are not independent evidence.', corrHost);
  attachChartToggle(corrCard._head, corrHost);

  const tbody = h('tbody', ...feats.map(f => h('tr',
    h('td', h('span', { class: 'tag', text: f.group || '—' })),
    h('td', { class: 'strong', text: f.label || f.name }),
    h('td', { class: 'num', text: fmt.num(f.ic_mean, 4) }),
    h('td', { class: 'num', text: fmt.num(f.ic_std, 4) }),
    h('td', { class: 'num' }, h('span', {
      class: Math.abs(f.ic_t ?? 0) >= 2 ? 'stat-delta up' : 'muted',
      text: fmt.num(f.ic_t, 2),
    })),
    h('td', { class: 'num', text: fmt.pct(f.hit_rate, 1) }),
    h('td', { class: 'num', text: fmt.num(f.coverage, 0) }))));

  const table = card('Per-feature diagnostics',
    'A |t| below about 2 means the feature has not proven itself on this sample.',
    h('div', { class: 'table-wrap', style: { maxHeight: '480px' } },
      h('table', { class: 'data' },
        h('thead', h('tr',
          h('th', { text: 'Group' }), h('th', { text: 'Feature' }),
          h('th', { class: 'num', text: 'IC mean' }), h('th', { class: 'num', text: 'IC sd' }),
          h('th', { class: 'num', text: 't-stat' }), h('th', { class: 'num', text: 'Hit rate' }),
          h('th', { class: 'num', text: 'Coverage' }))),
        tbody)));

  root.replaceChildren(
    h('div', { class: 'grid grid-2' }, icCard, decayCard),
    h('div', { style: { height: '14px' } }),
    h('div', { class: 'grid grid-2' }, corrCard, table));

  requestAnimationFrame(() => {
    barChart(icHost, {
      labels: feats.slice(0, 18).map(f => f.label || f.name),
      values: feats.slice(0, 18).map(f => f.ic_mean),
      horizontal: true, diverging: true, height: 300, barName: 'Rank IC',
      valueFormat: (v) => fmt.num(v, 3),
    });
    if (d.decay) {
      lineChart(decayHost, {
        x: d.decay.horizons.map(String),
        series: [{ name: 'Composite rank IC', values: d.decay.ic }],
        height: 260, baselineAt: 0, yFormat: (v) => fmt.num(v, 3),
        tooltipTitle: (i) => `${d.decay.horizons[i]}-day horizon`,
      });
      const b = decayHost.querySelectorAll('.tick-text');
      b.forEach(() => {});
    }
    if (d.correlation) {
      heatmap(corrHost, {
        rows: d.correlation.names, cols: d.correlation.names,
        matrix: d.correlation.matrix, diverging: true,
        valueFormat: (v) => fmt.num(v, 2),
      });
    }
  });
}

/* --------------------------------------------------------- backtest view */
function renderBacktest(root) {
  const d = state.backtest;
  if (!d || !d.equity) {
    root.replaceChildren(card('Backtest', null,
      empty('No backtest yet', 'Run the pipeline to generate a walk-forward backtest.', 'Run pipeline', runPipeline)));
    return;
  }
  const m = d.metrics || {};

  const kpis = h('div', { class: 'grid grid-5' },
    statTile('CAGR', fmt.signedPct(m.cagr), { foot: 'Net of costs' }),
    statTile('Sharpe', fmt.num(m.sharpe, 2), { dot: m.sharpe > 1 ? 'good' : m.sharpe > 0.5 ? 'warning' : 'critical' }),
    statTile('Max drawdown', fmt.pct(m.max_drawdown, 1), { foot: 'Peak to trough' }),
    statTile('vs benchmark', fmt.signedPct(m.excess_cagr), { foot: m.benchmark_name || 'Equal-weight universe' }),
    statTile('Annual turnover', fmt.pct(m.turnover, 0), { foot: `${fmt.num(m.cost_bps, 0)} bps per side` }));

  const eqHost = h('div', { class: 'chart' });
  const eqCard = card('Equity curve', 'Walk-forward, out of sample, net of transaction costs. Growth of 1.', eqHost);
  attachChartToggle(eqCard._head, eqHost);

  const ddHost = h('div', { class: 'chart' });
  const ddCard = card('Drawdown', 'Distance below the running peak.', ddHost);
  attachChartToggle(ddCard._head, ddHost);

  const decHost = h('div', { class: 'chart' });
  const decCard = card('Decile spread', 'Mean forward excess return by predicted decile. A monotone staircase is the goal.', decHost);
  attachChartToggle(decCard._head, decHost);

  const controls = h('div', { class: 'filterbar' },
    h('label', { class: 'control' }, h('span', { class: 'control-label', text: 'Portfolio size' }),
      h('select', { onchange: (e) => { state.backtestOpts.topn = +e.target.value; loadBacktest(); } },
        ...[10, 20, 30, 50].map(n => h('option', { value: n, text: String(n), selected: state.backtestOpts.topn === n })))),
    h('label', { class: 'control' }, h('span', { class: 'control-label', text: 'Cost (bps/side)' }),
      h('select', { onchange: (e) => { state.backtestOpts.cost_bps = +e.target.value; loadBacktest(); } },
        ...[0, 5, 10, 20].map(n => h('option', { value: n, text: String(n), selected: state.backtestOpts.cost_bps === n })))),
    h('label', { class: 'control' },
      h('input', {
        type: 'checkbox', checked: state.backtestOpts.long_short,
        onchange: (e) => { state.backtestOpts.long_short = e.target.checked; loadBacktest(); },
      }), h('span', { class: 'control-label', text: 'Long / short' })));

  const caveat = h('div', { class: 'note' },
    h('strong', { text: 'What this backtest does and does not prove. ' }),
    'Every point is generated by a model trained only on data available before that date, with an embargo between train and test. ',
    'It still cannot capture market impact, borrow cost, or the fact that this strategy was designed with hindsight about which factors are known to work.');

  root.replaceChildren(
    controls, kpis,
    h('div', { style: { height: '14px' } }), eqCard,
    h('div', { style: { height: '14px' } }),
    h('div', { class: 'grid grid-2' }, ddCard, decCard),
    h('div', { style: { height: '14px' } }), caveat);

  requestAnimationFrame(() => {
    lineChart(eqHost, {
      x: d.equity.dates,
      series: [
        { name: 'Model', values: d.equity.strategy },
        { name: d.equity.benchmark_name || 'Benchmark', values: d.equity.benchmark },
      ],
      height: 300, yFormat: (v) => fmt.num(v, 2), baselineAt: 1,
    });
    if (d.drawdown) {
      areaChart(ddHost, {
        x: d.drawdown.dates, values: d.drawdown.values,
        height: 220, color: 'var(--series-8)', name: 'Drawdown',
        yFormat: (v) => fmt.pct(v, 0),
      });
    }
    if (d.deciles) {
      barChart(decHost, {
        labels: d.deciles.labels, values: d.deciles.values,
        height: 220, diverging: true, barName: 'Mean excess',
        valueFormat: (v) => fmt.pct(v, 1),
      });
    }
  });
}

/* ------------------------------------------------------------- data view */
function renderData(root) {
  const d = state.health;
  if (!d) {
    root.replaceChildren(card('Data health', null, empty('No data yet', 'Run the pipeline to populate the local store.', 'Run pipeline', runPipeline)));
    return;
  }

  const tiles = h('div', { class: 'grid grid-4' },
    ...(d.sources || []).map(s => statTile(s.name, s.rows !== undefined ? fmt.compact(s.rows) : '—', {
      dot: s.status === 'ok' ? 'good' : s.status === 'stale' ? 'warning' : 'critical',
      small: true,
      foot: `${s.coverage !== undefined && s.coverage !== null ? fmt.pct(s.coverage, 0) + ' coverage · ' : ''}${s.last_updated ? 'updated ' + fmt.date(s.last_updated) : 'never fetched'}`,
    })));

  const rows = (d.sources || []).map(s => h('tr',
    h('td', h('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
      h('span', { class: `dot dot-${s.status === 'ok' ? 'good' : s.status === 'stale' ? 'warning' : 'critical'}` }),
      h('span', { class: 'strong', text: s.name }))),
    h('td', { class: 'muted', text: s.detail || '' }),
    h('td', { class: 'num', text: s.rows !== undefined ? fmt.compact(s.rows) : '—' }),
    h('td', { class: 'num', text: s.entities !== undefined ? fmt.compact(s.entities) : '—' }),
    h('td', { class: 'num', text: s.span || '—' }),
    h('td', { class: 'num muted', text: s.last_updated ? fmt.date(s.last_updated) : '—' })));

  const warnings = (d.warnings || []).length
    ? card('Warnings', 'Issues that would quietly degrade the model if left alone.',
        h('div', { class: 'stack' }, ...d.warnings.map(w =>
          h('div', { class: 'note' },
            h('strong', { text: (w.severity || 'note') + ': ' }), w.message))))
    : null;

  root.replaceChildren(
    tiles, h('div', { style: { height: '14px' } }),
    card('Sources', 'Everything the model reads, and how fresh it is.',
      h('div', { class: 'table-wrap' }, h('table', { class: 'data' },
        h('thead', h('tr',
          h('th', { text: 'Source' }), h('th', { text: 'Detail' }),
          h('th', { class: 'num', text: 'Rows' }), h('th', { class: 'num', text: 'Entities' }),
          h('th', { class: 'num', text: 'Span' }), h('th', { class: 'num', text: 'Updated' }))),
        h('tbody', ...rows)))),
    warnings ? h('div', { style: { height: '14px' } }) : null, warnings);
}

/* ------------------------------------------------------------- pipeline */
let pollTimer = null;

async function runPipeline() {
  const btn = document.getElementById('btn-refresh');
  btn.disabled = true;
  try {
    await api('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    pollRun();
  } catch (e) {
    btn.disabled = false;
    showStatus('critical', e.message);
  }
}

function showStatus(kind, msg, spinning) {
  const box = document.getElementById('run-status');
  box.style.display = '';
  box.replaceChildren(
    spinning ? h('span', { class: 'spinner' }) : h('span', { class: `dot dot-${kind}` }),
    h('span', { text: msg }));
}

async function pollRun() {
  clearTimeout(pollTimer);
  try {
    const s = await api('/api/run/status');
    if (s.running) {
      showStatus('warning', `${s.stage || 'Working'}… ${s.progress !== undefined ? Math.round(s.progress * 100) + '%' : ''}`, true);
      pollTimer = setTimeout(pollRun, 1200);
    } else {
      document.getElementById('btn-refresh').disabled = false;
      if (s.error) showStatus('critical', s.error);
      else {
        showStatus('good', 'Up to date');
        setTimeout(() => { document.getElementById('run-status').style.display = 'none'; }, 3500);
      }
      await loadAll();
    }
  } catch {
    document.getElementById('btn-refresh').disabled = false;
  }
}

/* ---------------------------------------------------------------- router */
function switchView(name) {
  state.view = name;
  document.querySelectorAll('.nav-item').forEach(b => b.classList.toggle('is-active', b.dataset.view === name));
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('is-active', v.id === 'view-' + name));
  document.getElementById('view-title').textContent = VIEW_TITLES[name] || name;
  rerender();
}

function rerender() {
  const root = document.getElementById('view-' + state.view);
  if (!root) return;
  switch (state.view) {
    case 'picks': renderPicks(root); break;
    case 'screener': renderScreener(root); break;
    case 'ticker': renderTicker(root); break;
    case 'signals': renderSignals(root); break;
    case 'backtest': renderBacktest(root); break;
    case 'data': renderData(root); break;
  }
}

/* ------------------------------------------------------------------ load */
async function loadBacktest() {
  const q = new URLSearchParams({
    topn: state.backtestOpts.topn,
    cost_bps: state.backtestOpts.cost_bps,
    long_short: state.backtestOpts.long_short ? '1' : '0',
  });
  const root = document.getElementById('view-backtest');
  root.classList.add('is-loading');       // hold the frame, no skeleton flash
  try {
    state.backtest = await api('/api/backtest?' + q);
  } catch { state.backtest = null; }
  root.classList.remove('is-loading');
  if (state.view === 'backtest') rerender();
}

async function loadAll() {
  const settle = (p) => p.then(v => v).catch(() => null);
  const [status, picks, signals, health] = await Promise.all([
    settle(api('/api/status')),
    settle(api('/api/picks?limit=600')),
    settle(api('/api/signals')),
    settle(api('/api/health')),
  ]);
  state.status = status; state.picks = picks; state.signals = signals; state.health = health;

  document.getElementById('foot-asof').textContent = picks?.asof ? 'As of ' + fmt.date(picks.asof) : 'No data yet';
  document.getElementById('foot-model').textContent = status?.model_label || (picks?.model?.name ?? '');

  loadBacktest();
  rerender();
}

/* ------------------------------------------------------------------ init */
function init() {
  document.getElementById('nav').addEventListener('click', (e) => {
    const btn = e.target.closest('.nav-item');
    if (btn) switchView(btn.dataset.view);
  });
  document.getElementById('btn-refresh').addEventListener('click', runPipeline);
  document.getElementById('btn-theme').addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('spaid-theme', next);
    rerender();
  });
  const saved = localStorage.getItem('spaid-theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);

  onResize(rerender);
  loadAll();
  pollRun();
}

init();
