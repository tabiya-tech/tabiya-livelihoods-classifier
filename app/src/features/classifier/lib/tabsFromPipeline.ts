/**
 * Decides whether the results panel should render the NER-derived tabs
 * (Entities / Table / JSON) for the current run. Today all pipelines pass
 * text through NER, so this always returns true — but once source-only or
 * sink-only pipelines exist, this becomes the single gate for the panel.
 *
 * The check reads `ClassifyMetadata.pipeline`, which the backend attaches
 * to every response starting with pipeline-executor rollout. If the field
 * is missing (legacy responses), we default to true so old behaviour holds.
 */

import type { ClassifyMetadata } from "@/lib/api";

export function shouldRenderNerTabs(
  metadata: ClassifyMetadata | null | undefined,
): boolean {
  if (!metadata?.pipeline) return true;
  return metadata.pipeline.stages.some((stage) => stage.category === "core");
}
