import { useEffect, useState } from "react";
import { listPipelines } from "@/lib/api";

/**
 * Returns a map of pipeline_id → pipeline name, loaded once on mount.
 * Used to resolve names in the classifications history table.
 */
export function usePipelineNames(): Map<string, string> {
  const [nameMap, setNameMap] = useState<Map<string, string>>(new Map());

  useEffect(() => {
    listPipelines()
      .then((response) => {
        const entries = response.pipelines.map(
          (pipeline) => [pipeline.pipeline_id, pipeline.name] as const,
        );
        setNameMap(new Map(entries));
      })
      .catch(() => {
        // Silently fail — the table falls back to pipeline_id
      });
  }, []);

  return nameMap;
}
