/**
 * client.ts — the only place that talks to the API.
 *
 * Every request funnels through `apiFetch`, which converts a non-2xx response
 * into an `ApiError` carrying the server's `{detail}` string. `useApi` turns
 * that into a discriminated union so a view can never confuse "empty" with
 * "failed": `state: 'error'` always carries a message and a `reload()`.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  HealthResponse,
  OpportunitiesQuery,
  OpportunitiesResponse,
  PipelineStartResponse,
  PipelineStatus,
  StatusResponse,
  StockDetail,
} from './types';
import type { TrialRegistry, ValidationReport } from './validation-types';

/* --------------------------------------------------------------- errors */

export class ApiError extends Error {
  readonly status: number;
  readonly path: string;

  constructor(message: string, status: number, path: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.path = path;
  }
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

/** A human-readable message for anything thrown by `apiFetch`. */
export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof TypeError) {
    return 'Could not reach the API. Is the backend running on 127.0.0.1:8000?';
  }
  if (err instanceof Error) return err.message;
  return 'Unknown error.';
}

function detailFrom(body: unknown): string | null {
  if (typeof body === 'object' && body !== null && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === 'string' && detail.trim() !== '') return detail;
    if (detail !== null && detail !== undefined) return JSON.stringify(detail);
  }
  return null;
}

/* -------------------------------------------------------------- fetching */

export interface RequestOptions {
  signal?: AbortSignal;
  method?: 'GET' | 'POST';
  body?: unknown;
}

/** Typed fetch. Resolves with the parsed body or rejects with an ApiError. */
export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { signal, method = 'GET', body } = options;

  const init: RequestInit = {
    method,
    headers: body === undefined ? { Accept: 'application/json' } : { Accept: 'application/json', 'Content-Type': 'application/json' },
  };
  if (signal) init.signal = signal;
  if (body !== undefined) init.body = JSON.stringify(body);

  const response = await fetch(path, init);

  let parsed: unknown = null;
  const text = await response.text();
  if (text !== '') {
    try {
      parsed = JSON.parse(text) as unknown;
    } catch {
      parsed = null;
      if (response.ok) {
        throw new ApiError(`The API returned a non-JSON response (${response.status}).`, response.status, path);
      }
    }
  }

  if (!response.ok) {
    const detail = detailFrom(parsed);
    throw new ApiError(
      detail ?? `Request failed with ${response.status} ${response.statusText}`.trim(),
      response.status,
      path,
    );
  }

  return parsed as T;
}

/* ----------------------------------------------------------------- paths */

function queryString(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue;
    const asString = typeof value === 'number' ? String(value) : value.trim();
    if (asString === '') continue;
    search.set(key, asString);
  }
  const encoded = search.toString();
  return encoded === '' ? '' : `?${encoded}`;
}

/** Path builders — pass these to `useApi` so the hook can key on the URL. */
export const apiPaths = {
  status: (): string => '/api/status',
  opportunities: (query: OpportunitiesQuery = {}): string =>
    `/api/opportunities${queryString({
      limit: query.limit,
      sector: query.sector,
      band: query.band,
      min_score: query.min_score,
      search: query.search,
    })}`,
  stock: (ticker: string): string => `/api/stock/${encodeURIComponent(ticker)}`,
  health: (): string => '/api/health',
  pipelineRun: (): string => '/api/pipeline/run',
  pipelineStatus: (): string => '/api/pipeline/status',
  validation: (period?: string | null): string => `/api/validation${queryString({ period })}`,
  validationTrials: (limit?: number): string => `/api/validation/trials${queryString({ limit })}`,
};

/* --------------------------------------------------------- typed helpers */

export function getStatus(signal?: AbortSignal): Promise<StatusResponse> {
  return apiFetch<StatusResponse>(apiPaths.status(), { signal });
}

export function getOpportunities(
  query: OpportunitiesQuery = {},
  signal?: AbortSignal,
): Promise<OpportunitiesResponse> {
  return apiFetch<OpportunitiesResponse>(apiPaths.opportunities(query), { signal });
}

export function getStock(ticker: string, signal?: AbortSignal): Promise<StockDetail> {
  return apiFetch<StockDetail>(apiPaths.stock(ticker), { signal });
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return apiFetch<HealthResponse>(apiPaths.health(), { signal });
}

export function getValidation(period?: string | null, signal?: AbortSignal): Promise<ValidationReport> {
  return apiFetch<ValidationReport>(apiPaths.validation(period), { signal });
}

export function getValidationTrials(limit?: number, signal?: AbortSignal): Promise<TrialRegistry> {
  return apiFetch<TrialRegistry>(apiPaths.validationTrials(limit), { signal });
}

export function runPipeline(signal?: AbortSignal): Promise<PipelineStartResponse> {
  return apiFetch<PipelineStartResponse>(apiPaths.pipelineRun(), { method: 'POST', signal });
}

export function getPipelineStatus(signal?: AbortSignal): Promise<PipelineStatus> {
  return apiFetch<PipelineStatus>(apiPaths.pipelineStatus(), { signal });
}

/* ------------------------------------------------------------- useApi ---- */

export type ApiResult<T> =
  | { state: 'loading' }
  | { state: 'error'; message: string }
  | { state: 'data'; data: T };

export interface UseApiExtras {
  /** Re-fetch, keeping the current render in place while it runs. */
  reload: () => void;
  /** True while a background refresh (reload or poll) is in flight. */
  refreshing: boolean;
}

export type UseApiResult<T> = ApiResult<T> & UseApiExtras;

export interface UseApiOptions {
  /** Poll interval in ms. 0 (default) disables polling. */
  pollMs?: number;
  /** Set false to hold the request without unmounting the consumer. */
  enabled?: boolean;
}

/**
 * Fetch `path` and expose the outcome as a discriminated union.
 *
 * - `path === null` (or `enabled: false`) issues no request and stays in
 *   `{state:'loading'}` — use it for "nothing selected yet".
 * - Changing `path` resets to `loading`; `reload()` and polling keep the
 *   previous data visible and flip `refreshing` instead.
 * - In-flight requests are aborted on unmount and on path change, so there
 *   are no unhandled rejections and no setState-after-unmount.
 */
export function useApi<T>(path: string | null, options: UseApiOptions = {}): UseApiResult<T> {
  const { pollMs = 0, enabled = true } = options;

  const [result, setResult] = useState<ApiResult<T>>({ state: 'loading' });
  const [refreshing, setRefreshing] = useState(false);
  const [nonce, setNonce] = useState(0);

  // Mirrors of state for use inside the fetching effect without re-running it.
  const resultRef = useRef<ApiResult<T>>(result);
  resultRef.current = result;
  const loadedPathRef = useRef<string | null>(null);

  const reload = useCallback(() => {
    setNonce((n) => n + 1);
  }, []);

  useEffect(() => {
    if (path === null || !enabled) {
      loadedPathRef.current = null;
      setResult({ state: 'loading' });
      setRefreshing(false);
      return;
    }

    const controller = new AbortController();
    let disposed = false;

    const isSamePath = loadedPathRef.current === path;
    const keepPrevious = isSamePath && resultRef.current.state === 'data';

    if (keepPrevious) {
      setRefreshing(true);
    } else {
      setResult({ state: 'loading' });
      setRefreshing(false);
    }

    apiFetch<T>(path, { signal: controller.signal })
      .then((data) => {
        if (disposed) return;
        loadedPathRef.current = path;
        setResult({ state: 'data', data });
        setRefreshing(false);
      })
      .catch((err: unknown) => {
        if (disposed || isAbortError(err)) return;
        loadedPathRef.current = path;
        setResult({ state: 'error', message: errorMessage(err) });
        setRefreshing(false);
      });

    return () => {
      disposed = true;
      controller.abort();
    };
  }, [path, enabled, nonce]);

  useEffect(() => {
    if (pollMs <= 0 || path === null || !enabled) return;
    const id = window.setInterval(() => setNonce((n) => n + 1), pollMs);
    return () => window.clearInterval(id);
  }, [pollMs, path, enabled]);

  return { ...result, reload, refreshing } as UseApiResult<T>;
}

/**
 * One-shot mutation helper for POST endpoints (e.g. the Run pipeline button).
 * Never rejects: the outcome is returned so callers cannot leak a rejection.
 */
export type MutationOutcome<T> = { ok: true; data: T } | { ok: false; message: string };

export async function mutate<T>(run: () => Promise<T>): Promise<MutationOutcome<T>> {
  try {
    return { ok: true, data: await run() };
  } catch (err) {
    if (isAbortError(err)) return { ok: false, message: 'Cancelled.' };
    return { ok: false, message: errorMessage(err) };
  }
}
