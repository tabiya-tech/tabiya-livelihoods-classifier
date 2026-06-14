/**
 * Locale enum, human-readable labels, and the set of locales the app supports.
 *
 * Adding a new locale: add it to {@link Locale}, give it a label in
 * {@link LocalesLabels}, drop a `translation.json` under
 * `src/i18n/locales/<code>/`, and register it in `i18n.ts`. The
 * `locales/locales.test.ts` parity check will fail until the new locale has
 * the same key shape as the reference (en-US).
 */

export enum Locale {
  EN_US = "en-US",
  FR_FR = "fr-FR",
}

export const LocalesLabels = {
  [Locale.EN_US]: "English (US)",
  [Locale.FR_FR]: "Français (France)",
} as const;

export const SupportedLocales: Locale[] = [Locale.EN_US, Locale.FR_FR];

export const FALL_BACK_LOCALE: Locale = Locale.EN_US;
