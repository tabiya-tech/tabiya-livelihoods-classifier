import { describe, expect, it } from "vitest";
import { Locale } from "./constants";
import { constructLocaleResources, getPossibleLocaleNames } from "./utils";

describe("getPossibleLocaleNames", () => {
  it("expands a region-qualified locale into BCP-47 aliases", () => {
    // GIVEN a region-qualified locale
    const givenLocale = Locale.EN_US;

    // WHEN we ask for its possible alias names
    const actualAliases = getPossibleLocaleNames(givenLocale);

    // THEN the original code, language-only, lowercased, and normalized forms are all present
    expect(actualAliases).toContain("en-US");
    expect(actualAliases).toContain("en");
    expect(actualAliases).toContain("en-us");
  });

  it("deduplicates aliases that coincide for simple locales", () => {
    // GIVEN a locale with overlapping aliases
    // WHEN we expand it
    const actualAliases = getPossibleLocaleNames(Locale.EN_US);

    // THEN there are no duplicate entries
    const uniqueAliases = new Set(actualAliases);
    expect(uniqueAliases.size).toBe(actualAliases.length);
  });
});

describe("constructLocaleResources", () => {
  it("wraps a translation object under every alias of the locale", () => {
    // GIVEN a locale and a sparse resource object
    const givenLocale = Locale.EN_US;
    const givenResource = { common: { hello: "Hello" } };

    // WHEN we construct i18next resources from it
    const actualResources = constructLocaleResources(givenLocale, givenResource);

    // THEN each alias maps to { translation: resource }
    for (const alias of getPossibleLocaleNames(givenLocale)) {
      expect(actualResources[alias]).toEqual({ translation: givenResource });
    }
  });
});
