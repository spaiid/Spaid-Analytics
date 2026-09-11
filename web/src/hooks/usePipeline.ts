/**
 * usePipeline — start a pipeline run and follow it to the end.
 *
 * The hook owns one long-lived poll of `GET /api/pipeline/status`:
 *  - it fetches once on mount, so a run started somewhere else (another tab,
 *    the CLI) is adopted rather than ignored;
 *  - while a run is in flight it polls every `pollMs`, and stops the moment the
 *    server reports `running: false`;
 *  - `start()` POSTs to `/api/pipeline/run` and never leaks a rejection. A 409
 *    ("already in progress") is not an error — it just means we should watch the
 *    run that is already going.
 *
 * Nothing here interprets the pipeline: stage, progress and error are the
 * server's words, passed through untouched.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiError, apiPaths, errorMessage, runPipeline, useApi } from '../api/client';
import type { PipelineStatus } from '../api/types';

export interface PipelineController {
  /** A run is in flight (ours or somebody else's). */
  running: boolean;
  /** The server's stage label, verbatim. */
  stage: string | null;
  /** Fraction 0-1, or null when the server does not report progress. */
  progress: number | null;
  /** The last failure: a failed start, or the error the run ended with. */
  error: string | null;
  /** The POST is in flight but the run has not been confirmed yet. */
  starting: boolean;
  /** Epoch ms when a run we were watching finished; null when nothing to report. */
  finishedAt: number | null;
  /** True when the finished run ended without an error. */
  succeeded: boolean;
  start: () => void;
  /** Hide the finished / failed notice until the next run. */
  dismiss: () => void;
}

export interface UsePipelineOptions {
  /** Poll interval while a run is in flight. */
  pollMs?: number;
  /** Called once, when a run we were watching stops. */
  onFinish?: () => void;
}

/**
 * The server flips `running` to true inside the POST handler, but a status
 * response that was already in flight can still say "not running". Ignore
 * not-running reports for this long after a start so a run is never reported
 * finished before it has begun.
 */
const START_GRACE_MS = 3000;

export function usePipeline(options: UsePipelineOptions = {}): PipelineController {
  const { pollMs = 2000, onFinish } = options;

  const [watching, setWatching] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [finishedAt, setFinishedAt] = useState<number | null>(null);
  const [finishedError, setFinishedError] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState(false);

  // The path never changes, so the last body stays on screen when polling stops.
  const poll = useApi<PipelineStatus>(apiPaths.pipelineStatus(), { pollMs: watching ? pollMs : 0 });
  const status: PipelineStatus | null = poll.state === 'data' ? poll.data : null;

  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const watchingRef = useRef(watching);
  watchingRef.current = watching;
  const startingRef = useRef(starting);
  startingRef.current = starting;
  const reloadRef = useRef(poll.reload);
  reloadRef.current = poll.reload;
  const onFinishRef = useRef(onFinish);
  useEffect(() => {
    onFinishRef.current = onFinish;
  }, [onFinish]);
  const startedAtRef = useRef(0);

  // Adopt a run that was started elsewhere.
  const serverRunning = status !== null && status.running;
  useEffect(() => {
    if (!serverRunning) return;
    setWatching(true);
    setDismissed(false);
    setFinishedAt(null);
    setFinishedError(null);
  }, [serverRunning]);

  // Notice the end of the run we are watching. `watching` flips false here, so
  // the effect settles after one pass even though `status` is a new object on
  // every poll.
  useEffect(() => {
    if (!watching || status === null || status.running) return;
    if (Date.now() - startedAtRef.current < START_GRACE_MS) return;
    setWatching(false);
    setFinishedAt(Date.now());
    setFinishedError(status.error);
    setDismissed(false);
    onFinishRef.current?.();
  }, [watching, status]);

  const start = useCallback(() => {
    if (startingRef.current || watchingRef.current) return;
    startedAtRef.current = Date.now();
    setStarting(true);
    setStartError(null);
    setFinishedAt(null);
    setFinishedError(null);
    setDismissed(false);

    void (async () => {
      try {
        const response = await runPipeline();
        if (!mountedRef.current) return;
        setStarting(false);
        if (!response.started) {
          setStartError('The API did not start a run.');
          return;
        }
        setWatching(true);
        reloadRef.current();
      } catch (error: unknown) {
        if (!mountedRef.current) return;
        setStarting(false);
        if (error instanceof ApiError && error.status === 409) {
          // Already in progress — follow that run instead of reporting a failure.
          setWatching(true);
          reloadRef.current();
          return;
        }
        setStartError(errorMessage(error));
      }
    })();
  }, []);

  const dismiss = useCallback(() => {
    setDismissed(true);
    setFinishedAt(null);
  }, []);

  const running = watching || serverRunning;
  // `status.error` survives on the server until the next run, so a failure from
  // an earlier run is still reported rather than silently dropped.
  const error = startError ?? finishedError ?? (status !== null ? status.error : null);

  return useMemo<PipelineController>(
    () => ({
      running,
      stage: status !== null ? status.stage : starting ? 'Starting' : null,
      progress: status !== null ? status.progress : null,
      error: dismissed ? null : error,
      starting,
      finishedAt: dismissed ? null : finishedAt,
      succeeded: finishedAt !== null && finishedError === null,
      start,
      dismiss,
    }),
    [running, status, starting, dismissed, error, finishedAt, finishedError, start, dismiss],
  );
}
