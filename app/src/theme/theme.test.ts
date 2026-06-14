import { describe, expect, it } from "vitest";
import { colors, ENTITY_TYPES, fontFamily } from "./theme";

describe("design tokens", () => {
  it("exposes navy as the primary brand color", () => {
    // GIVEN the brand navy hex from the design handoff
    const expectedBrandNavyHex = "#002147";

    // WHEN we read the default navy value from the tokens
    const actualNavyDefault = colors.navy.DEFAULT;

    // THEN the default navy matches the brand hex
    expect(actualNavyDefault).toBe(expectedBrandNavyHex);
  });

  it("defines a fg/bg pair for every entity type", () => {
    // GIVEN a regex that matches a 6-digit hex color
    const sixDigitHexPattern = /^#[0-9a-f]{6}$/i;

    // WHEN we read each entity type's foreground/background entries
    // THEN every entry has fg and bg as 6-digit hex values
    for (const entityType of ENTITY_TYPES) {
      const entityColorPair = colors.entity[entityType];
      expect(entityColorPair.fg).toMatch(sixDigitHexPattern);
      expect(entityColorPair.bg).toMatch(sixDigitHexPattern);
    }
  });

  it("declares IBM Plex Mono as the first mono fallback", () => {
    // GIVEN the brand voice typeface as the expected first mono fallback
    const expectedFirstMonoFontFamily = '"IBM Plex Mono"';

    // WHEN we read the first item of the mono stack
    const actualFirstMonoFont = fontFamily.mono[0];

    // THEN it matches the brand voice typeface
    expect(actualFirstMonoFont).toBe(expectedFirstMonoFontFamily);
  });
});
