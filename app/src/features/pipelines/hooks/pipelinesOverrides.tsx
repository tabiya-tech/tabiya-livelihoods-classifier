/**
 * Storybook-only override plumbing for the Pipelines page's hooks.
 *
 * Production never wraps the page in these providers. Each hook reads its
 * override context; when present it short-circuits the real fetch and
 * returns the canned snapshot the story provided.
 */

import { createContext, type ReactNode } from "react";
import type { PipelinesListSnapshot } from "./usePipelinesList";
import type { ActivatePipelineState } from "./useActivatePipeline";
import type { ClonePipelineState } from "./useClonePipeline";
import type { DeletePipelineState } from "./useDeletePipeline";

export const PipelinesListOverrideContext =
  createContext<PipelinesListSnapshot | null>(null);
export const ActivatePipelineOverrideContext =
  createContext<ActivatePipelineState | null>(null);
export const ClonePipelineOverrideContext =
  createContext<ClonePipelineState | null>(null);
export const DeletePipelineOverrideContext =
  createContext<DeletePipelineState | null>(null);

export interface PipelinesOverridesProviderProps {
  pipelinesList?: PipelinesListSnapshot;
  activatePipeline?: ActivatePipelineState;
  clonePipeline?: ClonePipelineState;
  deletePipeline?: DeletePipelineState;
  children: ReactNode;
}

export function PipelinesOverridesProvider({
  pipelinesList,
  activatePipeline,
  clonePipeline,
  deletePipeline,
  children,
}: PipelinesOverridesProviderProps) {
  return (
    <PipelinesListOverrideContext.Provider value={pipelinesList ?? null}>
      <ActivatePipelineOverrideContext.Provider value={activatePipeline ?? null}>
        <ClonePipelineOverrideContext.Provider value={clonePipeline ?? null}>
          <DeletePipelineOverrideContext.Provider value={deletePipeline ?? null}>
            {children}
          </DeletePipelineOverrideContext.Provider>
        </ClonePipelineOverrideContext.Provider>
      </ActivatePipelineOverrideContext.Provider>
    </PipelinesListOverrideContext.Provider>
  );
}
