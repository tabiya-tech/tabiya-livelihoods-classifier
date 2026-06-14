import { describe, expect, it } from "vitest";
import { colors, ENTITY_TYPES, fontFamily } from "./theme";

describe("design tokens", () => {
  it("exposes navy as the primary brand color", () => {
    // GIVEN the colors token export

    // WHEN we read the default navy value
    const navyDefault = colors.navy.DEFAULT;

    // THEN it matches the brand navy hex from the design handoff
    expect(navyDefault).toBe("#002147");
  });

  it("defines a fg/bg pair for every entity type", () => {
    // GIVEN the set of entity types
    // WHEN we read the colors.entity entries for each type
    // THEN every entry has fg and bg hex values
    for (const type of ENTITY_TYPES) {
      const entry = colors.entity[type];
      expect(entry.fg).toMatch(/^#[0-9a-f]{6}$/i);
      expect(entry.bg).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  it("declares IBM Plex Mono as the first mono fallback", () => {
    // GIVEN the fontFamily token export

    // WHEN we read the first item of the mono stack
    const firstMonoFont = fontFamily.mono[0];

    // THEN it is IBM Plex Mono (the brand voice typeface)
    expect(firstMonoFont).toBe('"IBM Plex Mono"');
  });
});
