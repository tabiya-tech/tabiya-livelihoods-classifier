import { describe, expect, it } from "vitest";
import type { Pipeline } from "@/lib/api";
import {
  defaultsFromPipeline,
  type ClassifierDefaults,
} from "./defaultsFromPipeline";

function makePipeline(stages: Pipeline["stages"]): Pipeline {
  return {
    pipeline_id: "pipeline-under-test",
    user_id: "local-user",
    name: "Pipeline under test",
    stages,
    is_active: true,
    is_default: false,
    is_readonly: false,
    created_at: "2026-01-01T00:00:00.000Z",
    updated_at: "2026-01-01T00:00:00.000Z",
  };
}

describe("defaultsFromPipeline", () => {
  it("returns the legacy defaults when the pipeline is null", () => {
    // GIVEN no pipeline
    const givenPipeline = null;
    const expectedDefaults: ClassifierDefaults = {
      topK: 5,
      minSimilarity: 0,
      entityTypes: null,
    };

    // WHEN we read defaults
    const actualDefaults = defaultsFromPipeline(givenPipeline);

    // THEN they match the legacy hard-coded defaults
    expect(actualDefaults).toEqual(expectedDefaults);
  });

  it("reads top_k, min_similarity, and entity_types from a fully populated pipeline", () => {
    // GIVEN a pipeline with all three fields set
    const givenTopK = 12;
    const givenMinSimilarity = 0.4;
    const givenEntityTypes = ["occupation", "skill"];
    const givenPipeline = makePipeline([
      { plugin_id: "tabiya.source.text.v1", config: {} },
      {
        plugin_id: "tabiya.ner.v1",
        config: { entity_types: givenEntityTypes },
      },
      {
        plugin_id: "tabiya.nel.v1",
        config: {
          top_k: givenTopK,
          min_similarity: givenMinSimilarity,
        },
      },
      { plugin_id: "tabiya.sink.results.v1", config: {} },
    ]);
    const expectedDefaults: ClassifierDefaults = {
      topK: givenTopK,
      minSimilarity: givenMinSimilarity,
      entityTypes: givenEntityTypes as ClassifierDefaults["entityTypes"],
    };

    // WHEN we read defaults
    const actualDefaults = defaultsFromPipeline(givenPipeline);

    // THEN we get the pipeline's own values
    expect(actualDefaults).toEqual(expectedDefaults);
  });

  it("falls back to NER defaults when only the NEL stage carries config", () => {
    // GIVEN a pipeline where only NEL supplies values
    const givenTopK = 10;
    const givenMinSimilarity = 0.25;
    const givenPipeline = makePipeline([
      { plugin_id: "tabiya.source.text.v1", config: {} },
      {
        plugin_id: "tabiya.nel.v1",
        config: {
          top_k: givenTopK,
          min_similarity: givenMinSimilarity,
        },
      },
    ]);
    const expectedDefaults: ClassifierDefaults = {
      topK: givenTopK,
      minSimilarity: givenMinSimilarity,
      entityTypes: null,
    };

    // WHEN we read defaults
    const actualDefaults = defaultsFromPipeline(givenPipeline);

    // THEN NEL values win, entity_types stays null (no NER stage)
    expect(actualDefaults).toEqual(expectedDefaults);
  });

  it("coerces malformed config values back to the legacy defaults", () => {
    // GIVEN a pipeline with wrong-typed configs (strings where numbers expected)
    const givenPipeline = makePipeline([
      {
        plugin_id: "tabiya.ner.v1",
        config: { entity_types: "occupation,skill" },
      },
      {
        plugin_id: "tabiya.nel.v1",
        config: {
          top_k: "twelve",
          min_similarity: "half",
        },
      },
    ]);
    const expectedDefaults: ClassifierDefaults = {
      topK: 5,
      minSimilarity: 0,
      entityTypes: null,
    };

    // WHEN we read defaults
    const actualDefaults = defaultsFromPipeline(givenPipeline);

    // THEN malformed values silently fall through to defaults
    expect(actualDefaults).toEqual(expectedDefaults);
  });
});
