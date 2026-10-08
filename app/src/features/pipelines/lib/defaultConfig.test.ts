import { describe, expect, it } from "vitest";
import type { PluginManifest } from "@/lib/api";
import { defaultConfigForManifest, missingRequiredFields } from "./defaultConfig";

function makeManifest(configSchema: Record<string, unknown>): PluginManifest {
  return {
    plugin_id: "tabiya.nel.v1",
    name: "Tabiya NEL",
    version: "0.1.0",
    category: "core",
    summary: "Links entities.",
    icon: "nel",
    input_slot: { type: "Entities" },
    output_slot: { type: "LinkedEntities" },
    config_schema: configSchema,
    timeout_ms: 45000,
  } as unknown as PluginManifest;
}

describe("defaultConfigForManifest", () => {
  it("seeds config from declared JSON-Schema defaults", () => {
    // GIVEN a schema with defaults on optional fields
    const givenManifest = makeManifest({
      type: "object",
      properties: {
        nel_model_id: { type: "string" },
        top_k: { type: "integer", default: 5 },
        min_similarity: { type: "number", default: 0 },
      },
      required: ["nel_model_id"],
    });
    const expectedConfig = { top_k: 5, min_similarity: 0 };

    // WHEN we build the default config
    const actualConfig = defaultConfigForManifest(givenManifest);

    // THEN only the fields with declared defaults are populated
    expect(actualConfig).toEqual(expectedConfig);
  });

  it("returns an empty object when there are no defaults", () => {
    // GIVEN a schema with no defaults
    const givenManifest = makeManifest({
      type: "object",
      properties: { nel_model_id: { type: "string" } },
      required: ["nel_model_id"],
    });

    // WHEN we build the default config
    const actualConfig = defaultConfigForManifest(givenManifest);

    // THEN it's empty
    expect(actualConfig).toEqual({});
  });
});

describe("missingRequiredFields", () => {
  it("reports required fields that are absent or empty", () => {
    // GIVEN NEL's schema and a config missing both model ids
    const givenManifest = makeManifest({
      type: "object",
      properties: {
        nel_model_id: { type: "string" },
        taxonomy_model_id: { type: "string" },
        top_k: { type: "integer", default: 5 },
      },
      required: ["nel_model_id", "taxonomy_model_id"],
    });
    const givenConfig = { top_k: 5 };
    const expectedMissing = ["nel_model_id", "taxonomy_model_id"];

    // WHEN we compute missing required fields
    const actualMissing = missingRequiredFields(givenManifest, givenConfig);

    // THEN both required-but-empty model ids are reported
    expect(actualMissing).toEqual(expectedMissing);
  });

  it("reports nothing when required fields are filled", () => {
    // GIVEN a config with both model ids set
    const givenManifest = makeManifest({
      type: "object",
      properties: {
        nel_model_id: { type: "string" },
        taxonomy_model_id: { type: "string" },
      },
      required: ["nel_model_id", "taxonomy_model_id"],
    });
    const givenConfig = { nel_model_id: "m1", taxonomy_model_id: "t1" };

    // WHEN we compute missing required fields
    const actualMissing = missingRequiredFields(givenManifest, givenConfig);

    // THEN nothing is missing
    expect(actualMissing).toEqual([]);
  });
});
