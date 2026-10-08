/**
 * Initialize i18next. Imported for side effects from main.tsx (production)
 * and from .storybook/preview.tsx (Storybook). Calling code never imports
 * this module by name beyond that — they use `useTranslation()` from
 * react-i18next instead.
 */

import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import LanguageDetector from "i18next-browser-languagedetector";
import {
  FALL_BACK_LOCALE,
  Locale,
  SupportedLocales,
} from "./constants";
import { constructLocaleResources } from "./utils";
import enUsTranslations from "./locales/en-US/translation.json";
import frFrTranslations from "./locales/fr-FR/translation.json";

const resources = {
  ...constructLocaleResources(Locale.EN_US, enUsTranslations),
  ...constructLocaleResources(Locale.FR_FR, frFrTranslations),
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: FALL_BACK_LOCALE,
    supportedLngs: SupportedLocales,
    detection: {
      order: ["localStorage", "navigator"],
      lookupLocalStorage: "tabiya-classifier-locale",
      caches: ["localStorage"],
      convertDetectedLanguage: (detectedLng) => {
        const normalized = detectedLng.toLowerCase();
        const exactMatch = SupportedLocales.find(
          (supported) => supported.toLowerCase() === normalized,
        );
        if (exactMatch) return exactMatch;

        // Language-only match (e.g. "en" → "en-US"). Prefer the first locale
        // in SupportedLocales that shares the language prefix.
        const languagePrefix = normalized.split("-")[0];
        const languageMatch = SupportedLocales.find((supported) =>
          supported.toLowerCase().startsWith(`${languagePrefix}-`),
        );
        return languageMatch ?? FALL_BACK_LOCALE;
      },
    },
    interpolation: {
      // React already escapes by default.
      escapeValue: false,
    },
  });

// Defensive: if the detector somehow landed on something unsupported, snap to fallback.
if (!SupportedLocales.includes(i18n.language as Locale)) {
  console.error(
    `Detected language "${i18n.language}" is not supported; falling back to ${FALL_BACK_LOCALE}.`,
  );
  i18n.changeLanguage(FALL_BACK_LOCALE);
}

// Surface missing-key errors prominently in the console so they don't ship silently.
i18n.on("missingKey", (languages, namespace, key, fallbackValue) => {
  console.error("Missing translation for key", {
    languages,
    namespace,
    key,
    fallbackValue,
  });
});

i18n.on("failedLoading", (lng, ns, msg) => {
  console.error(`Failed to load translation for ${lng}/${ns}`, msg);
});

export default i18n;
