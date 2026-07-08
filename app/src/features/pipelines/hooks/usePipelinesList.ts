/**
 * Loads the caller's pipelines and exposes a refetch hook for callers
 * that just mutated server state (activate / clone / delete).
 */

import { useCallback, useContext, useEffect, useState } from "react";
import { listPipelines, type Pipeline } from "@/lib/api";
import { PipelinesListOverrideContext } from "./pipelinesOverrides";

export type PipelinesListStatus = "loading" | "ready" | "error";

export interface PipelinesListSnapshot {
  status: PipelinesListStatus;
  pipelines: Pipeline[];
  error: Error | null;
  /** Re-runs the list query. Resolves once state has been updated. */
  refetch: () => Promise<void>;
}

export interface UsePipelinesListOptions {
  /** Test seam — override the backend fetch. Defaults to the real client. */
  fetchPipelines?: () => Promise<{ pipelines: Pipeline[] }>;
}

export function usePipelinesList({
  fetchPipelines = listPipelines,
}: UsePipelinesListOptions = {}): PipelinesListSnapshot {
  const override = useContext(PipelinesListOverrideContext);
  const [snapshot, setSnapshot] = useState<Omit<PipelinesListSnapshot, "refetch">>({
    status: "loading",
    pipelines: [],
    error: null,
  });

  const load = useCallback(async () => {
    try {
      const response = await fetchPipelines();
      setSnapshot({ status: "ready", pipelines: response.pipelines, error: null });
    } catch (caught: unknown) {
      const error =
        caught instanceof Error ? caught : new Error(String(caught));
      setSnapshot({ status: "error", pipelines: [], error });
    }
  }, [fetchPipelines]);

  useEffect(() => {
    if (override) return;
    void load();
  }, [load, override]);

  if (override) return override;
  return { ...snapshot, refetch: load };
}
