/**
 * Loads the catalog of NEL embedding models from the backend.
 *
 * Snapshot:
 * - `status: "loading"` until the request resolves.
 * - `status: "ready"` exposes the list of models.
 * - `status: "error"` carries the underlying error and an empty list.
 */

import { useContext, useEffect, useState } from "react";
import { listNelModels, type NelModel } from "@/lib/api";
import { NelModelsOverrideContext } from "./configurationOverrides";

export type NelModelsStatus = "loading" | "ready" | "error";

export interface NelModelsSnapshot {
  status: NelModelsStatus;
  models: NelModel[];
  error: Error | null;
}

export interface UseNelModelsOptions {
  /** Test seam — override the backend fetch. Defaults to the real client. */
  fetchModels?: () => Promise<NelModel[]>;
}

export function useNelModels({
  fetchModels = listNelModels,
}: UseNelModelsOptions = {}): NelModelsSnapshot {
  const override = useContext(NelModelsOverrideContext);
  const [snapshot, setSnapshot] = useState<NelModelsSnapshot>({
    status: "loading",
    models: [],
    error: null,
  });

  useEffect(() => {
    if (override) return;
    let cancelled = false;
    fetchModels()
      .then((models) => {
        if (cancelled) return;
        setSnapshot({ status: "ready", models, error: null });
      })
      .catch((caught: unknown) => {
        if (cancelled) return;
        const error =
          caught instanceof Error ? caught : new Error(String(caught));
        setSnapshot({ status: "error", models: [], error });
      });
    return () => {
      cancelled = true;
    };
    // fetchModels is intentionally outside the dep array; it's a test seam
    // that production callers never change at runtime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [override]);

  return override ?? snapshot;
}
