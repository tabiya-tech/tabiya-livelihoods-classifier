/**
 * Extracts the classifier's runtime defaults (top_k, min_similarity,
 * entity-type filter) from the active pipeline's NER + NEL stages.
 *
 * Pipeline stage configs are `Record<string, unknown>` at the wire level
 * because plugins define their own schemas — this helper is the single
 * place we coerce those `unknown` values into the typed shape the UI
 * needs. Anything malformed silently falls back to the legacy defaults.
 */

import type { ClassifyEntityType, Pipeline } from "@/lib/api";

export interface ClassifierDefaults {
  topK: number;
  minSimilarity: number;
  entityTypes: ClassifyEntityType[] | null;
}

const DEFAULT_TOP_K = 5;
const DEFAULT_MIN_SIMILARITY = 0;

export function defaultsFromPipeline(
  pipeline: Pipeline | null,
): ClassifierDefaults {
  const nelStage = pipeline?.stages.find((stage) =>
    stage.plugin_id.includes("nel"),
  );
  const nerStage = pipeline?.stages.find((stage) =>
    stage.plugin_id.includes("ner"),
  );
  const topKRaw = nelStage?.config?.top_k;
  const minSimilarityRaw = nelStage?.config?.min_similarity;
  const entityTypesRaw = nerStage?.config?.entity_types;

  return {
    topK: typeof topKRaw === "number" ? topKRaw : DEFAULT_TOP_K,
    minSimilarity:
      typeof minSimilarityRaw === "number"
        ? minSimilarityRaw
        : DEFAULT_MIN_SIMILARITY,
    entityTypes: Array.isArray(entityTypesRaw)
      ? (entityTypesRaw.filter(
          (entityType) => typeof entityType === "string",
        ) as ClassifyEntityType[])
      : null,
  };
}
