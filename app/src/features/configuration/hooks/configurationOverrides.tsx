/**
 * Storybook-only override plumbing for the Configuration page's hooks.
 *
 * Production code never wraps the page in these providers. Each hook checks
 * for an active override via `useContext`; when present it short-circuits the
 * real `fetch` path and returns the canned snapshot the story provided.
 *
 * Pattern mirrors `AuthOverrideProvider` in `@/lib/auth/useFirebaseAuth`.
 */

import { createContext, type ReactNode } from "react";
import type { NelModelsSnapshot } from "./useNelModels";
import type { TaxonomyModelsSnapshot } from "./useTaxonomyModels";
import type { UserConfigurationState } from "./useUserConfiguration";

export const NelModelsOverrideContext = createContext<NelModelsSnapshot | null>(
  null,
);

export const TaxonomyModelsOverrideContext =
  createContext<TaxonomyModelsSnapshot | null>(null);

export const UserConfigurationOverrideContext =
  createContext<UserConfigurationState | null>(null);

export interface ConfigurationOverridesProviderProps {
  nelModels?: NelModelsSnapshot;
  taxonomyModels?: TaxonomyModelsSnapshot;
  userConfiguration?: UserConfigurationState;
  children: ReactNode;
}

/**
 * Wrap a subtree to short-circuit every Configuration-page hook with a
 * deterministic snapshot. Intended for Storybook and exploratory harnesses —
 * production never mounts this.
 */
export function ConfigurationOverridesProvider({
  nelModels,
  taxonomyModels,
  userConfiguration,
  children,
}: ConfigurationOverridesProviderProps) {
  return (
    <NelModelsOverrideContext.Provider value={nelModels ?? null}>
      <TaxonomyModelsOverrideContext.Provider value={taxonomyModels ?? null}>
        <UserConfigurationOverrideContext.Provider
          value={userConfiguration ?? null}
        >
          {children}
        </UserConfigurationOverrideContext.Provider>
      </TaxonomyModelsOverrideContext.Provider>
    </NelModelsOverrideContext.Provider>
  );
}
