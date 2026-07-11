/**
 * Resolves the output slot type of the last stage in the active pipeline.
 *
 * Fetches the plugin manifest for the final stage's plugin_id whenever the
 * active pipeline changes. Returns null while loading or when no pipeline is
 * selected. Falls back to null on fetch errors so the UI defaults to the full
 * linked-entities view.
 */

import { useEffect, useState } from "react";
import { getPlugin, type Pipeline, type PluginSlot } from "@/lib/api";

export type PipelineOutputSlotStatus = "loading" | "ready";

export interface PipelineOutputSlotSnapshot {
  status: PipelineOutputSlotStatus;
  outputSlotType: PluginSlot["type"] | null;
}

export interface UsePipelineOutputSlotOptions {
  fetchPlugin?: typeof getPlugin;
}

export function usePipelineOutputSlot(
  activePipeline: Pipeline | null,
  { fetchPlugin = getPlugin }: UsePipelineOutputSlotOptions = {},
): PipelineOutputSlotSnapshot {
  const [snapshot, setSnapshot] = useState<PipelineOutputSlotSnapshot>({
    status: "loading",
    outputSlotType: null,
  });

  useEffect(() => {
    if (!activePipeline) {
      setSnapshot({ status: "ready", outputSlotType: null });
      return;
    }

    const lastStage = activePipeline.stages[activePipeline.stages.length - 1];
    if (!lastStage) {
      setSnapshot({ status: "ready", outputSlotType: null });
      return;
    }

    let cancelled = false;
    setSnapshot({ status: "loading", outputSlotType: null });

    fetchPlugin(lastStage.plugin_id)
      .then((detail) => {
        if (cancelled) return;
        setSnapshot({
          status: "ready",
          outputSlotType: detail.manifest?.output_slot.type ?? null,
        });
      })
      .catch(() => {
        if (cancelled) return;
        // Fall back to null — UI shows full linked-entities view
        setSnapshot({ status: "ready", outputSlotType: null });
      });

    return () => {
      cancelled = true;
    };
  }, [activePipeline, fetchPlugin]);

  return snapshot;
}
