/**
 * Locale parity check.
 *
 * Walks every locale folder under src/i18n/locales/ and asserts that every
 * non-reference translation file has the exact same key shape as the
 * reference locale (en-US). Replacing leaf values with "-" lets us compare
 * structure regardless of the actual translated copy.
 *
 * Also verifies every `SupportedLocales` entry has a translation.json on
 * disk, so a registration drift between constants.ts and the locales folder
 * fails fast.
 */

import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { Locale, SupportedLocales } from "@/i18n/constants";

const REFERENCE_LOCALE = Locale.EN_US;
const localesDir = path.dirname(new URL(import.meta.url).pathname);

type TranslationLeaf = string;
// Recursive alias used for the structural comparison below.
type TranslationObject = {
  [key: string]: TranslationLeaf | TranslationObject;
};

/**
 * Recursively replaces every leaf value in a translation object with "-".
 * Used to compare the *shape* of two translation files irrespective of the
 * actual translated copy.
 */
function replaceValuesWithDash(value: TranslationObject): TranslationObject {
  const stripped: TranslationObject = {};
  for (const key of Object.keys(value)) {
    const current = value[key];
    stripped[key] =
      typeof current === "object" && current !== null
        ? replaceValuesWithDash(current)
        : "-";
  }
  return stripped;
}

function readTranslationJson(localeDir: string): TranslationObject {
  const translationPath = path.join(localesDir, localeDir, "translation.json");
  return JSON.parse(fs.readFileSync(translationPath, "utf8"));
}

describe("Feature: i18n locales consistency", () => {
  const referenceTranslations = readTranslationJson(REFERENCE_LOCALE);

  const nonReferenceLocaleDirs = fs
    .readdirSync(localesDir)
    .filter((entry) => {
      const fullPath = path.join(localesDir, entry);
      return fs.statSync(fullPath).isDirectory() && entry !== REFERENCE_LOCALE;
    });

  it.each(nonReferenceLocaleDirs)(
    "Scenario: %s should have the same keys as " + REFERENCE_LOCALE,
    (givenLocaleDirectoryName) => {
      // GIVEN the reference translation shape (en-US) and the other locale's translations
      const expectedShape = replaceValuesWithDash(referenceTranslations);

      // WHEN we strip leaf values from the other locale
      const otherLocaleTranslations = readTranslationJson(
        givenLocaleDirectoryName,
      );
      const actualShape = replaceValuesWithDash(otherLocaleTranslations);

      // THEN the shape matches the reference
      expect(actualShape).toEqual(expectedShape);
    },
  );

  it.each(SupportedLocales)(
    "supported locale %s should have a translation file",
    (givenSupportedLocale) => {
      // GIVEN a locale registered in SupportedLocales
      // WHEN we look for its translation.json
      const translationFilePath = path.join(
        localesDir,
        givenSupportedLocale,
        "translation.json",
      );

      // THEN the file exists on disk
      expect(fs.existsSync(translationFilePath)).toBe(true);
    },
  );
});
