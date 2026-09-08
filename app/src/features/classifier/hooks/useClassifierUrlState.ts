/**
 * Synchronises the Classifier's request-scoped knobs — top_k and
 * min_similarity — to the URL search params so a run is shareable.
 *
 * Source text and entity-type filter are NOT synced — text can be large
 * and sensitive, and the type filter is a view-time concern, not a
 * request-time one.
 *
 * Values are clamped to the backend's bounds (1–50 / 0.0–1.0) before being
 * surfaced to callers; an unparseable param falls back to the supplied
 * default.
 */

import { useCallback, useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";

export const TOP_K_PARAM = "top_k";
export const MIN_SIMILARITY_PARAM = "min_sim";

export const TOP_K_DEFAULT = 5;
export const MIN_SIMILARITY_DEFAULT = 0;
export const TOP_K_MIN = 1;
export const TOP_K_MAX = 50;
export const MIN_SIMILARITY_MIN = 0;
export const MIN_SIMILARITY_MAX = 1;

function clamp(value: number, lo: number, hi: number): number {
  if (Number.isNaN(value)) return lo;
  return Math.min(hi, Math.max(lo, value));
}

function parseInteger(raw: string | null, fallback: number): number {
  if (raw === null) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseFloat(raw: string | null, fallback: number): number {
  if (raw === null) return fallback;
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export interface ClassifierUrlState {
  topK: number;
  minSimilarity: number;
  setTopK: (value: number) => void;
  setMinSimilarity: (value: number) => void;
}

export function useClassifierUrlState(): ClassifierUrlState {
  const [searchParams, setSearchParams] = useSearchParams();

  const topK = useMemo(
    () =>
      clamp(
        parseInteger(searchParams.get(TOP_K_PARAM), TOP_K_DEFAULT),
        TOP_K_MIN,
        TOP_K_MAX,
      ),
    [searchParams],
  );
  const minSimilarity = useMemo(
    () =>
      clamp(
        parseFloat(searchParams.get(MIN_SIMILARITY_PARAM), MIN_SIMILARITY_DEFAULT),
        MIN_SIMILARITY_MIN,
        MIN_SIMILARITY_MAX,
      ),
    [searchParams],
  );

  // On mount: if the URL had an unparseable / out-of-range value, normalise
  // it so the address bar matches the clamped value the rest of the app sees.
  useEffect(() => {
    const rawTopK = searchParams.get(TOP_K_PARAM);
    const rawMin = searchParams.get(MIN_SIMILARITY_PARAM);
    const normalisedTopK = String(topK);
    const normalisedMin = minSimilarity.toFixed(2);
    if (
      (rawTopK !== null && rawTopK !== normalisedTopK) ||
      (rawMin !== null && rawMin !== normalisedMin)
    ) {
      const next = new URLSearchParams(searchParams);
      if (rawTopK !== null) next.set(TOP_K_PARAM, normalisedTopK);
      if (rawMin !== null) next.set(MIN_SIMILARITY_PARAM, normalisedMin);
      setSearchParams(next, { replace: true });
    }
    // We only need this normalisation once per param-set change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setTopK = useCallback(
    (value: number) => {
      const clamped = clamp(Math.round(value), TOP_K_MIN, TOP_K_MAX);
      setSearchParams(
        (current) => {
          const next = new URLSearchParams(current);
          next.set(TOP_K_PARAM, String(clamped));
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setMinSimilarity = useCallback(
    (value: number) => {
      const clamped = clamp(value, MIN_SIMILARITY_MIN, MIN_SIMILARITY_MAX);
      setSearchParams(
        (current) => {
          const next = new URLSearchParams(current);
          next.set(MIN_SIMILARITY_PARAM, clamped.toFixed(2));
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return { topK, minSimilarity, setTopK, setMinSimilarity };
}
