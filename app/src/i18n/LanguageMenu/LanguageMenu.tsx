/**
 * Language picker for the Topbar.
 *
 * Renders an IconButton (globe) that opens a small panel of locale options.
 * Hidden entirely when only one locale is shipped — the i18n machinery is
 * wired but the surface only appears once translators have a real choice
 * to offer.
 *
 * The dropdown closes on:
 * - clicking outside the menu or trigger
 * - pressing Escape
 * - picking a locale
 */

import { useEffect, useId, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { IconButton } from "@/components";
import { mergeClassNames } from "@/lib/mergeClassNames";
import {
  type Locale,
  LocalesLabels,
  SupportedLocales,
} from "@/i18n/constants";

const uniqueId = "8a3f6c4d-2b1e-4f8a-9c7b-3d6e5f4a2c1b";

export const DATA_TEST_ID = {
  CONTAINER: `language-menu-container-${uniqueId}`,
  TRIGGER: `language-menu-trigger-${uniqueId}`,
  PANEL: `language-menu-panel-${uniqueId}`,
  OPTION: `language-menu-option-${uniqueId}`,
};

export function LanguageMenu() {
  const { i18n, t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const [isOpen, setIsOpen] = useState(false);
  const panelId = useId();

  // Close on click outside the menu container.
  useEffect(() => {
    if (!isOpen) return;
    function handleDocumentClick(event: MouseEvent) {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleDocumentClick);
    return () => document.removeEventListener("mousedown", handleDocumentClick);
  }, [isOpen]);

  // Close on Escape.
  useEffect(() => {
    if (!isOpen) return;
    function handleKeydown(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") setIsOpen(false);
    }
    document.addEventListener("keydown", handleKeydown);
    return () => document.removeEventListener("keydown", handleKeydown);
  }, [isOpen]);

  if (SupportedLocales.length <= 1) {
    return null;
  }

  const currentLocale = (SupportedLocales.find(
    (locale) => locale === i18n.language,
  ) ?? SupportedLocales[0]) as Locale;

  function handlePickLocale(nextLocale: Locale) {
    setIsOpen(false);
    if (nextLocale === currentLocale) return;
    i18n.changeLanguage(nextLocale).catch((caught) => {
      console.error(`Failed to change language to ${nextLocale}`, caught);
    });
  }

  function handleTriggerKeydown(event: KeyboardEvent<HTMLButtonElement>) {
    if (
      event.key === "ArrowDown" ||
      event.key === "Enter" ||
      event.key === " "
    ) {
      event.preventDefault();
      setIsOpen(true);
    }
  }

  return (
    <div
      ref={containerRef}
      data-testid={DATA_TEST_ID.CONTAINER}
      className="relative"
    >
      <IconButton
        icon="globe"
        aria-label={t("shell.languageMenu.selectLanguage")}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={panelId}
        onClick={() => setIsOpen((wasOpen) => !wasOpen)}
        onKeyDown={handleTriggerKeydown}
        data-testid={DATA_TEST_ID.TRIGGER}
      />

      {isOpen && (
        <div
          id={panelId}
          role="listbox"
          aria-label={t("shell.languageMenu.selectLanguage")}
          data-testid={DATA_TEST_ID.PANEL}
          className={mergeClassNames(
            "absolute right-0 top-full z-50 mt-1.5 min-w-[180px] overflow-hidden",
            "rounded-md border border-line bg-paper shadow-card-2",
          )}
        >
          {SupportedLocales.map((locale) => {
            const isCurrent = locale === currentLocale;
            return (
              <button
                key={locale}
                type="button"
                role="option"
                aria-selected={isCurrent}
                data-testid={DATA_TEST_ID.OPTION}
                data-locale={locale}
                onClick={() => handlePickLocale(locale)}
                className={mergeClassNames(
                  "flex w-full items-center justify-between gap-3 px-3 py-2 text-left",
                  "font-mono text-xs transition-colors",
                  isCurrent
                    ? "bg-cream-200 text-navy"
                    : "text-ink hover:bg-cream",
                )}
              >
                <span>{LocalesLabels[locale]}</span>
                {isCurrent && (
                  <span aria-hidden className="text-[10px] text-muted">
                    ✓
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
