import { describe, expect, it } from "vitest";
import { areSlotsCompatible, slotColor } from "./slotCompat";
import type { PluginSlot } from "@/lib/api";

describe("areSlotsCompatible", () => {
  it("returns true when output and input slot types match exactly", () => {
    // GIVEN a matching slot type pair
    const givenOutputType: PluginSlot["type"] = "RawText";
    const givenInputType: PluginSlot["type"] = "RawText";

    // WHEN we check compatibility
    const result = areSlotsCompatible(givenOutputType, givenInputType);

    // THEN it is compatible
    expect(result).toBe(true);
  });

  it("returns true for matching Entities slot types", () => {
    // GIVEN a matching Entities pair
    const givenOutputType: PluginSlot["type"] = "Entities";
    const givenInputType: PluginSlot["type"] = "Entities";

    // WHEN we check compatibility
    const result = areSlotsCompatible(givenOutputType, givenInputType);

    // THEN it is compatible
    expect(result).toBe(true);
  });

  it("returns true for matching LinkedEntities slot types", () => {
    // GIVEN a matching LinkedEntities pair
    const givenOutputType: PluginSlot["type"] = "LinkedEntities";
    const givenInputType: PluginSlot["type"] = "LinkedEntities";

    // WHEN we check compatibility
    const result = areSlotsCompatible(givenOutputType, givenInputType);

    // THEN it is compatible
    expect(result).toBe(true);
  });

  it("returns false when output is RawText but input expects Entities", () => {
    // GIVEN a mismatched pair
    const givenOutputType: PluginSlot["type"] = "RawText";
    const givenInputType: PluginSlot["type"] = "Entities";

    // WHEN we check compatibility
    const result = areSlotsCompatible(givenOutputType, givenInputType);

    // THEN it is not compatible
    expect(result).toBe(false);
  });

  it("returns false when output is Entities but input expects LinkedEntities", () => {
    // GIVEN a mismatched Entities->LinkedEntities pair
    const givenOutputType: PluginSlot["type"] = "Entities";
    const givenInputType: PluginSlot["type"] = "LinkedEntities";

    // WHEN we check compatibility
    const result = areSlotsCompatible(givenOutputType, givenInputType);

    // THEN it is not compatible
    expect(result).toBe(false);
  });

  it("returns false when output is None and input expects RawText", () => {
    // GIVEN a None->RawText mismatch (source boundary violation)
    const givenOutputType: PluginSlot["type"] = "None";
    const givenInputType: PluginSlot["type"] = "RawText";

    // WHEN we check compatibility
    const result = areSlotsCompatible(givenOutputType, givenInputType);

    // THEN it is not compatible
    expect(result).toBe(false);
  });
});

describe("slotColor", () => {
  it("returns a hex string for a known slot type", () => {
    // GIVEN a known slot type
    const givenSlotType: PluginSlot["type"] = "RawText";

    // WHEN we ask for the color
    const resultColor = slotColor(givenSlotType);

    // THEN we get a non-empty hex-like string
    expect(resultColor).toMatch(/^#[0-9a-fA-F]{3,8}$/);
  });
});
