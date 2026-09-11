/**
 * Sidebar — primary navigation.
 *
 * Every destination is a real link, so it can be focused, opened in a new tab
 * and announced with `aria-current="page"`. The views that are not built yet
 * are still reachable: they land on a panel that says what will live there,
 * and they are marked "Soon" in the nav (text, not colour).
 *
 * Below 900px the labels collapse to icons (base.css), so each link also
 * carries an `aria-label` — the accessible name survives the collapse.
 */

import { NavLink } from 'react-router-dom';
import type { ReactNode } from 'react';
import { date, ofTotal } from '../../lib/format';

/* ------------------------------------------------------------------ icons */
/* 16px line glyphs. Decorative: the link text (or aria-label) carries meaning. */

interface GlyphProps {
  children: ReactNode;
}

function Glyph({ children }: GlyphProps) {
  return (
    <svg className="ico" viewBox="0 0 16 16" aria-hidden="true" focusable="false" fill="none"
      stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
      {children}
    </svg>
  );
}

/** Ranked list. */
function RankIcon() {
  return (
    <Glyph>
      <path d="M2.5 4h11M2.5 8h8M2.5 12h5" />
    </Glyph>
  );
}

/** A single name: candlestick. */
function StockIcon() {
  return (
    <Glyph>
      <path d="M5 2.5v11M11 2.5v11" />
      <rect x="3" y="5" width="4" height="5" rx="1" />
      <rect x="9" y="7" width="4" height="4.5" rx="1" />
    </Glyph>
  );
}

/** Validation: a check inside a shield. */
function ValidationIcon() {
  return (
    <Glyph>
      <path d="M8 1.8 2.8 4v4c0 3.2 2.2 5.3 5.2 6.2 3-0.9 5.2-3 5.2-6.2V4Z" />
      <path d="M5.9 7.9 7.4 9.4l2.9-3" />
    </Glyph>
  );
}

/** Data health: a pulse. */
function PulseIcon() {
  return (
    <Glyph>
      <path d="M1.5 8h3l2-4.5L9.5 12l2-4h3" />
    </Glyph>
  );
}

/** Today: a calendar. */
function CalendarIcon() {
  return (
    <Glyph>
      <rect x="2" y="3.5" width="12" height="10.5" rx="2" />
      <path d="M2 6.8h12M5.5 1.8v2.6M10.5 1.8v2.6" />
    </Glyph>
  );
}

/** Portfolio: a case. */
function BriefcaseIcon() {
  return (
    <Glyph>
      <rect x="1.8" y="4.8" width="12.4" height="9" rx="2" />
      <path d="M5.8 4.8V3.4a1.4 1.4 0 0 1 1.4-1.4h1.6a1.4 1.4 0 0 1 1.4 1.4v1.4M1.8 8.6h12.4" />
    </Glyph>
  );
}

/** Performance: a trend line. */
function TrendIcon() {
  return (
    <Glyph>
      <path d="M2 13V3M2 13h12M4.5 10.2l3-3.2 2.4 2 3.1-4" />
    </Glyph>
  );
}

/** Journal: a notebook. */
function JournalIcon() {
  return (
    <Glyph>
      <path d="M4 2h8.2a1.3 1.3 0 0 1 1.3 1.3v9.4A1.3 1.3 0 0 1 12.2 14H4" />
      <path d="M4 2a1.6 1.6 0 0 0-1.6 1.6v8.8A1.6 1.6 0 0 0 4 14M6.2 5.6h5M6.2 8.4h5" />
    </Glyph>
  );
}

/* ------------------------------------------------------------------ items */

interface NavItem {
  to: string;
  label: string;
  icon: ReactNode;
  /** Shown at the end of the row: a ticker, or "Soon". */
  tag?: string;
  /** Spoken suffix for a destination that is not built yet. */
  note?: string;
  /** `end` matches the path exactly — used for index-ish routes. */
  end?: boolean;
}

export interface SidebarProps {
  /** The ticker last opened, so "Stock detail" can point back at it. */
  lastTicker: string | null;
  asOf: string | null;
  scoredCount: number | null;
  universeSize: number | null;
}

const COMING_NEXT: NavItem[] = [
  { to: '/today', label: 'Today', icon: <CalendarIcon />, tag: 'Soon', note: 'not built yet' },
  { to: '/portfolio', label: 'Portfolio', icon: <BriefcaseIcon />, tag: 'Soon', note: 'not built yet' },
  { to: '/performance', label: 'Performance', icon: <TrendIcon />, tag: 'Soon', note: 'not built yet' },
  { to: '/journal', label: 'Research journal', icon: <JournalIcon />, tag: 'Soon', note: 'not built yet' },
];

function navClass({ isActive }: { isActive: boolean }): string {
  return isActive ? 'nav-item is-active' : 'nav-item';
}

function NavRow({ item }: { item: NavItem }) {
  const spoken = item.note ? `${item.label} — ${item.note}` : item.label;
  return (
    <li>
      <NavLink to={item.to} end={item.end} className={navClass} aria-label={spoken} title={item.label}>
        {item.icon}
        <span>{item.label}</span>
        {item.tag !== undefined && <span className="nav-item-tag">{item.tag}</span>}
      </NavLink>
    </li>
  );
}

export function Sidebar({ lastTicker, asOf, scoredCount, universeSize }: SidebarProps) {
  return (
    <nav className="sidebar" aria-label="Primary">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true" />
        <span className="brand-name">Spaid Analytics</span>
      </div>

      <div className="nav">
        <div className="nav-label" id="nav-research">
          Research
        </div>
        <ul aria-labelledby="nav-research" style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <NavRow item={{ to: '/opportunities', label: 'Opportunities', icon: <RankIcon /> }} />
          {lastTicker === null ? (
            <li>
              {/* Nothing opened yet, so there is no detail to show. Said in
                  words — the state is never implied by colour alone. */}
              <span className="nav-item" aria-disabled="true" title="Open a name from Opportunities first">
                <StockIcon />
                <span>Stock detail</span>
                <span className="nav-item-tag">Pick a name</span>
              </span>
            </li>
          ) : (
            <NavRow
              item={{
                to: `/stock/${encodeURIComponent(lastTicker)}`,
                label: 'Stock detail',
                icon: <StockIcon />,
                tag: lastTicker,
              }}
            />
          )}
          <NavRow item={{ to: '/validation', label: 'Validation', icon: <ValidationIcon /> }} />
          <NavRow item={{ to: '/health', label: 'Data health', icon: <PulseIcon /> }} />
        </ul>

        <div className="nav-label" id="nav-next">
          Coming next
        </div>
        <ul aria-labelledby="nav-next" style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {COMING_NEXT.map((item) => (
            <NavRow key={item.to} item={item} />
          ))}
        </ul>
      </div>

      <div className="sidebar-foot">
        <span>{ofTotal(scoredCount, universeSize)} scored</span>
        <span>As of {date(asOf)}</span>
      </div>
    </nav>
  );
}
