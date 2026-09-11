/**
 * labels.ts — palette lookup for readings the SERVER has already made.
 *
 * Nothing here decides what a number means. The API sends `band`,
 * `valuation_label`/`valuation_class` and `confidence_label` as finished text;
 * this module only answers "which status colour does that finished text wear",
 * which is styling, not analysis.
 *
 * The vocabularies below are the server's own (spaid/config/scoring.py
 * SCORE_BANDS and spaid/api/service.py VALUATION_CLASS_LABELS). Anything the
 * server sends that is NOT in them falls back to neutral, so a new or renamed
 * label can never be silently mis-coloured — and every pill ships an icon and
 * the text itself, so the colour is never the only carrier of meaning.
 */

import type { StatusTone } from '../ui';

function normalise(value: string): string {
  return value.trim().toLowerCase().replace(/[\s-]+/g, '_');
}

/** Composite-score bands, exactly as `SCORE_BANDS` labels them. */
const BAND_TONES: Record<string, StatusTone> = {
  very_attractive: 'good',
  attractive: 'good',
  neutral: 'neutral',
  unattractive: 'warning',
  very_unattractive: 'critical',
  not_scored: 'neutral',
};

export function bandTone(band: string | null | undefined): StatusTone {
  if (!band) return 'neutral';
  return BAND_TONES[normalise(band)] ?? 'neutral';
}

/** Valuation classes, keyed by the API's `valuation_class` or its label. */
const VALUATION_TONES: Record<string, StatusTone> = {
  significantly_undervalued: 'good',
  undervalued: 'good',
  fairly_valued: 'neutral',
  overvalued: 'warning',
  significantly_overvalued: 'serious',
  insufficient_confidence: 'neutral',
  insufficient_confidence_to_classify: 'neutral',
};

export function valuationTone(
  classification: string | null | undefined,
  label?: string | null,
): StatusTone {
  const key = classification ?? label;
  if (!key) return 'neutral';
  return VALUATION_TONES[normalise(key)] ?? 'neutral';
}
