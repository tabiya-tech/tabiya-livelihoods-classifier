/**
 * Loads the catalog of taxonomy models from the backend. Same snapshot
 * shape as `useNelModels` — the two hooks would compose into a single
 * generic loader, but keeping them separate gives clearer call sites and
 * easier test seams per resource.
 */

import { useEffect, useState } from "react";
import { listTaxonomyModels, type TaxonomyModel } from "@/lib/api";

export type TaxonomyModelsStatus = "loading" | "ready" | "error";

export interface TaxonomyModelsSnapshot {
  status: TaxonomyModelsStatus;
  models: TaxonomyModel[];
  error: Error | null;
}

export interface UseTaxonomyModelsOptions {
  /** Test seam — override the backend fetch. Defaults to the real client. */
  fetchModels?: () => Promise<TaxonomyModel[]>;
}

export function useTaxonomyModels({
  fetchModels = listTaxonomyModels,
}: UseTaxonomyModelsOptions = {}): TaxonomyModelsSnapshot {
  const [snapshot, setSnapshot] = useState<TaxonomyModelsSnapshot>({
    status: "loading",
    models: [],
    error: null,
  });

  useEffect(() => {
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return snapshot;
}
