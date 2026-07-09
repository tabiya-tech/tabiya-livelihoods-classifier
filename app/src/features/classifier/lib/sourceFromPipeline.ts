/**
 * Inspects the active pipeline's Source stage and returns which input UI
 * the SourcePane should render. Today only "text" is wired up in the UI;
 * "url" is future work for when the job-scraper plugin comes out of
 * "coming_soon" state. "unknown" means the caller should fall back to
 * something safe (a read-only stub, typically).
 *
 * We match on `plugin_id.includes(...)` rather than exact equality so
 * v2/v3 revisions of the same source plugin keep routing to the same
 * branch without a code change.
 */

import type { Pipeline } from "@/lib/api";

export type SourceKind = "text" | "url" | "unknown";

export function sourceKindFromPipeline(
  pipeline: Pipeline | null,
): SourceKind {
  if (!pipeline || pipeline.stages.length === 0) return "text";
  const sourceStage = pipeline.stages[0];
  if (sourceStage.plugin_id.includes("source.text")) return "text";
  if (sourceStage.plugin_id.includes("scraper")) return "url";
  return "unknown";
}
