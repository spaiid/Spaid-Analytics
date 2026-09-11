/**
 * router.ts — route parsing and href construction over real URL paths.
 *
 * Paths are plain, not hashes, because the API serves the single-page app with
 * a catch-all fallback: any unknown path returns index.html and the client
 * router takes over. That keeps /stock/NVDA shareable, bookmarkable and
 * meaningful in server logs.
 *
 *   /opportunities            /stock/AAPL            /health   /validation
 *   /today  /portfolio  /performance  /journal       (placeholders)
 */

import { useCallback } from 'react';
import { useLocation, useNavigate as useRouterNavigate } from 'react-router-dom';

export type Route =
  | { name: 'opportunities' }
  | { name: 'stock'; ticker: string }
  | { name: 'validation' }
  | { name: 'health' }
  | { name: 'today' }
  | { name: 'portfolio' }
  | { name: 'performance' }
  | { name: 'journal' }
  | { name: 'not-found'; path: string };

export type RouteName = Route['name'];

/** Routes that exist in the nav but are not built yet. */
export const PLACEHOLDER_ROUTES: readonly RouteName[] = ['today', 'portfolio', 'performance', 'journal'];

export function parsePath(pathname: string): Route {
  const path = pathname.replace(/^#/, '').replace(/^\/+/, '');
  const [head = '', tail = ''] = path.split('/');

  switch (head) {
    case '':
    case 'opportunities':
      return { name: 'opportunities' };
    case 'stock': {
      const ticker = decodeURIComponent(tail).trim().toUpperCase();
      return ticker === '' ? { name: 'opportunities' } : { name: 'stock', ticker };
    }
    case 'validation':
      return { name: 'validation' };
    case 'health':
      return { name: 'health' };
    case 'today':
      return { name: 'today' };
    case 'portfolio':
      return { name: 'portfolio' };
    case 'performance':
      return { name: 'performance' };
    case 'journal':
      return { name: 'journal' };
    default:
      return { name: 'not-found', path: `/${path}` };
  }
}

/** The href for a route — use it on real <a> elements so links are real links. */
export function hrefFor(route: Route): string {
  switch (route.name) {
    case 'opportunities':
      return '/opportunities';
    case 'stock':
      return `/stock/${encodeURIComponent(route.ticker)}`;
    case 'validation':
      return '/validation';
    case 'health':
      return '/health';
    case 'not-found':
      return route.path;
    default:
      return `/${route.name}`;
  }
}

/** Stable navigate callback for event handlers, backed by react-router. */
export function useNavigate(): (route: Route) => void {
  const routerNavigate = useRouterNavigate();
  return useCallback(
    (route: Route) => routerNavigate(hrefFor(route)),
    [routerNavigate],
  );
}

/** The current route, parsed from the URL path. */
export function useRoute(): Route {
  const { pathname } = useLocation();
  return parsePath(pathname);
}
