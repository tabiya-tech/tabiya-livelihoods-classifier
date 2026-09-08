/**
 * Type augmentation for react-i18next.
 *
 * Reads the shape of the reference locale (en-US) and exposes every dotted
 * key path as a string literal type. The `t()` function and `<Trans i18nKey>`
 * are then narrowed so typos become compile-time errors and editors get
 * autocomplete on translation keys.
 */

import type { i18n as I18nInstance, TFunction } from "i18next";
import type defaultLanguage from "src/i18n/locales/en-US/translation.json";

/**
 * Recursive helper: walk a nested object type and produce the union of all
 * dot-notation key paths. Leaves are emitted as their own keys; branches are
 * emitted as both the branch key and `branch.child` paths.
 */
type DotKeys<TNode> = {
  [K in keyof TNode & string]: TNode[K] extends Record<string, unknown>
    ? `${K}` | `${K}.${DotKeys<TNode[K]>}`
    : `${K}`;
}[keyof TNode & string];

export type TranslationKey = DotKeys<typeof defaultLanguage>;

export type TypedTFunction = TFunction<"translation"> & {
  (key: TranslationKey, options?: Record<string, unknown>): string;
};

/**
 * Module augmentation
 * -------------------
 * Keep the original return shape of useTranslation(), but narrow the type
 * of t() so only valid keys are allowed.
 */
declare module "react-i18next" {
  export function useTranslation(): {
    t: (key: TranslationKey, options?: Record<string, unknown>) => string;
    i18n: I18nInstance;
  };
}
