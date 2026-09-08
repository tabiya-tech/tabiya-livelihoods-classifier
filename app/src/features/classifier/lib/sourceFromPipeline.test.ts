import { describe, expect, it } from "vitest";
import type { Pipeline } from "@/lib/api";
import { sourceKindFromPipeline } from "./sourceFromPipeline";

function makePipeline(pipelineId: string, sourcePluginId: string): Pipeline {
  return {
    pipeline_id: pipelineId,
    user_id: "local-user",
    name: pipelineId,
    stages: [
      { plugin_id: sourcePluginId, config: {} },
      { plugin_id: "tabiya.ner.v1", config: {} },
    ],
    is_active: true,
    is_default: false,
    is_readonly: false,
    created_at: "2026-01-01T00:00:00.000Z",
    updated_at: "2026-01-01T00:00:00.000Z",
  };
}

describe("sourceKindFromPipeline", () => {
  it("returns 'text' when the first stage is a text source plugin", () => {
    // GIVEN a pipeline whose source stage is the text plugin
    const givenPipeline = makePipeline(
      "pipeline-text",
      "tabiya.source.text.v1",
    );
    const expectedSourceKind = "text";

    // WHEN we resolve its source kind
    const actualSourceKind = sourceKindFromPipeline(givenPipeline);

    // THEN we get "text"
    expect(actualSourceKind).toBe(expectedSourceKind);
  });

  it("returns 'url' when the first stage is a scraper source plugin", () => {
    // GIVEN a pipeline whose source stage is a scraper
    const givenPipeline = makePipeline(
      "pipeline-scraper",
      "tabiya.source.scraper.v1",
    );
    const expectedSourceKind = "url";

    // WHEN we resolve its source kind
    const actualSourceKind = sourceKindFromPipeline(givenPipeline);

    // THEN we get "url"
    expect(actualSourceKind).toBe(expectedSourceKind);
  });

  it("returns 'text' when the pipeline is null or has no stages", () => {
    // GIVEN no pipeline
    const givenPipeline = null;
    const expectedSourceKind = "text";

    // WHEN we resolve its source kind
    const actualSourceKind = sourceKindFromPipeline(givenPipeline);

    // THEN we default to text so the classifier still renders its textarea
    expect(actualSourceKind).toBe(expectedSourceKind);
  });

  it("returns 'unknown' when the first stage plugin doesn't match a known source family", () => {
    // GIVEN a pipeline whose source stage is an unrecognised plugin id
    const givenPipeline = makePipeline(
      "pipeline-mystery",
      "third-party.source.rss.v1",
    );
    const expectedSourceKind = "unknown";

    // WHEN we resolve its source kind
    const actualSourceKind = sourceKindFromPipeline(givenPipeline);

    // THEN we fall through to unknown
    expect(actualSourceKind).toBe(expectedSourceKind);
  });
});
