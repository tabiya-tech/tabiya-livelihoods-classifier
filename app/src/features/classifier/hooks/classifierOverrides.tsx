/**
 * Storybook-only override plumbing for the Classifier page's hooks.
 *
 * Mirrors `apiKeysOverrides.tsx` and Configuration's overrides — production
 * never mounts this. The hook checks the context; when present it returns
 * the canned snapshot the story provided.
 */

import { createContext, type ReactNode } from "react";
import type { ClassifyState } from "./useClassify";

export const ClassifyOverrideContext = createContext<ClassifyState | null>(null);

export interface ClassifierOverridesProviderProps {
  classify?: ClassifyState;
  children: ReactNode;
}

export function ClassifierOverridesProvider({
  classify,
  children,
}: ClassifierOverridesProviderProps) {
  return (
    <ClassifyOverrideContext.Provider value={classify ?? null}>
      {children}
    </ClassifyOverrideContext.Provider>
  );
}
