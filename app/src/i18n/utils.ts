/**
 * Build i18next resources that cover every reasonable form of a locale code
 * (case variations + language-only fallback). Ported from compass-zambia.
 *
 * Example: Locale.EN_US ("en-US") expands to keys for "en-US", "en", "en-us".
 */

import { type Resource, type ResourceLanguage } from "i18next";
import { type Locale } from "./constants";

class ParseLocaleError extends Error {
  readonly underlyingCause: unknown;
  constructor(locale: string, cause: unknown) {
    super(`Invalid locale: ${locale}`);
    this.name = "ParseLocaleError";
    this.underlyingCause = cause;
  }
}

/**
 * Get all possible locale names for a given locale code.
 *
 * en-US → ["en-US", "en", "en-us"]
 * en → ["en"]
 */
export function getPossibleLocaleNames(locale: Locale): string[] {
  try {
    const intlLocale = new Intl.Locale(locale);
    const possibleLocaleNames = [
      locale as string,
      intlLocale.language,
      locale.toLowerCase(),
    ];

    if (intlLocale.region) {
      possibleLocaleNames.push(
        `${intlLocale.language}-${intlLocale.region.toUpperCase()}`,
      );
    }

    // Remove duplicates while preserving order.
    return Array.from(new Set(possibleLocaleNames));
  } catch (caught) {
    // Intl.Locale has been supported in all major browsers since 2019-2020.
    // This catch is for environments where Intl.Locale is unavailable.
    console.error(new ParseLocaleError(locale, caught));
    return [locale];
  }
}

/**
 * Wrap a single locale's resource JSON into i18next's expected
 * `{ [locale]: { translation: { … } } }` shape, across every alias.
 */
export function constructLocaleResources(
  locale: Locale,
  resourceLanguage: ResourceLanguage,
): Resource {
  return getPossibleLocaleNames(locale).reduce<Resource>((accumulator, alias) => {
    accumulator[alias] = { translation: resourceLanguage };
    return accumulator;
  }, {});
}
