/**
 * MSW handler for POST /v2/classify.
 *
 * Returns the canonical fixture response, optionally trimmed by the request
 * options (top_k cap on matches per entity, min_similarity cutoff, and the
 * extract_entities type filter). Real backend semantics, faked locally.
 */

import { http, HttpResponse } from "msw";
import { API_BASE_URL } from "@/lib/api";
import type {
  ClassifiedEntity,
  ClassifyRequest,
  ClassifyResponse,
} from "@/lib/api";
import { fixtureClassifyResponse } from "../fixtures/classify";

let currentResponse: ClassifyResponse = fixtureClassifyResponse;

/** Reset the in-memory response to the canonical fixture. */
export function resetClassifyHandlersStore() {
  currentResponse = fixtureClassifyResponse;
}

/** Replace the response served for the next POSTs (used in stories/tests). */
export function seedClassifyHandlersStore(seed: ClassifyResponse) {
  currentResponse = seed;
}

function applyOptions(
  entities: ClassifiedEntity[],
  options: ClassifyRequest["options"],
): ClassifiedEntity[] {
  const allowedTypes = options?.extract_entities;
  const topK = options?.top_k;
  const minSimilarity = options?.min_similarity;

  return entities
    .filter((entity) =>
      allowedTypes ? allowedTypes.includes(entity.entity_type) : true,
    )
    .map((entity) => {
      let matches = entity.matches;
      if (minSimilarity != null) {
        matches = matches.filter(
          (match) => match.similarity_score >= minSimilarity,
        );
      }
      if (topK != null) {
        matches = matches.slice(0, topK);
      }
      return { ...entity, matches };
    });
}

export const classifyHandlers = [
  http.post(`${API_BASE_URL}/v2/classify`, async ({ request }) => {
    const body = (await request.json()) as ClassifyRequest;
    const filteredEntities = applyOptions(
      currentResponse.entities,
      body.options,
    );
    return HttpResponse.json({
      entities: filteredEntities,
      metadata: currentResponse.metadata,
    });
  }),
];
