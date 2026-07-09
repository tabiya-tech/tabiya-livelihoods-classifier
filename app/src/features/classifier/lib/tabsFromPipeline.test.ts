import { describe, expect, it } from "vitest";
import type { ClassifyMetadata } from "@/lib/api";
import { shouldRenderNerTabs } from "./tabsFromPipeline";

const baseMetadataFields = {
  classifier_version: "2.0.0",
  ner_model: "ner",
  nel_model_id: "nel",
  taxonomy_model_id: "tax",
  processing_time_ms: 1,
} as const;

describe("shouldRenderNerTabs", () => {
  it("returns true when metadata is null (legacy response without pipeline field)", () => {
    // GIVEN a null metadata (older backend, no pipeline echo)
    const givenMetadata = null;
    const expectedShouldRender = true;

    // WHEN we ask whether tabs should render
    const actualShouldRender = shouldRenderNerTabs(givenMetadata);

    // THEN we default to rendering (legacy behaviour)
    expect(actualShouldRender).toBe(expectedShouldRender);
  });

  it("returns true when the pipeline contains a core stage", () => {
    // GIVEN metadata describing a pipeline that runs NER (a core stage)
    const givenMetadata: ClassifyMetadata = {
      ...baseMetadataFields,
      pipeline: {
        pipeline_id: "pipeline-default",
        name: "Default Tabiya",
        stages: [
          { plugin_id: "tabiya.source.text.v1", category: "source" },
          { plugin_id: "tabiya.ner.v1", category: "core" },
          { plugin_id: "tabiya.sink.results.v1", category: "sink" },
        ],
      },
    };
    const expectedShouldRender = true;

    // WHEN we check
    const actualShouldRender = shouldRenderNerTabs(givenMetadata);

    // THEN the tabs render
    expect(actualShouldRender).toBe(expectedShouldRender);
  });

  it("returns false when the pipeline only has source and sink stages", () => {
    // GIVEN metadata for a source-plus-sink pipeline (no NER)
    const givenMetadata: ClassifyMetadata = {
      ...baseMetadataFields,
      pipeline: {
        pipeline_id: "pipeline-passthrough",
        name: "Passthrough",
        stages: [
          { plugin_id: "tabiya.source.text.v1", category: "source" },
          { plugin_id: "tabiya.sink.results.v1", category: "sink" },
        ],
      },
    };
    const expectedShouldRender = false;

    // WHEN we check
    const actualShouldRender = shouldRenderNerTabs(givenMetadata);

    // THEN the tabs are hidden (no NER data to show)
    expect(actualShouldRender).toBe(expectedShouldRender);
  });
});
