/**
 * Storybook-only override plumbing for the Classifier page's pipeline hook.
 *
 * Mirrors `classifierOverrides.tsx` — a story can wrap the page in this
 * provider to feed a canned pipelines snapshot without ever touching the
 * network. Production never mounts this.
 */

import { createContext, type ReactNode } from "react";
import type { ActivePipelineSnapshot } from "./useActivePipeline";

export const ActivePipelineOverrideContext =
  createContext<ActivePipelineSnapshot | null>(null);

export interface ClassifierPipelineOverridesProviderProps {
  activePipeline?: ActivePipelineSnapshot;
  children: ReactNode;
}

export function ClassifierPipelineOverridesProvider({
  activePipeline,
  children,
}: ClassifierPipelineOverridesProviderProps) {
  return (
    <ActivePipelineOverrideContext.Provider value={activePipeline ?? null}>
      {children}
    </ActivePipelineOverrideContext.Provider>
  );
}
