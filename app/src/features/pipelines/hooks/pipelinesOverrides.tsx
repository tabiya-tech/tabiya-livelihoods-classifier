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
import type { PluginOptionsState } from "./usePluginOptions";
import type { ValidatePipelineState } from "./useValidatePipeline";

export const PipelinesListOverrideContext =
  createContext<PipelinesListSnapshot | null>(null);
export const ActivatePipelineOverrideContext =
  createContext<ActivatePipelineState | null>(null);
export const ClonePipelineOverrideContext =
  createContext<ClonePipelineState | null>(null);
export const DeletePipelineOverrideContext =
  createContext<DeletePipelineState | null>(null);
export const PluginOptionsOverrideContext =
  createContext<PluginOptionsState | null>(null);
export const ValidatePipelineOverrideContext =
  createContext<ValidatePipelineState | null>(null);

export interface PipelinesOverridesProviderProps {
  pipelinesList?: PipelinesListSnapshot;
  activatePipeline?: ActivatePipelineState;
  clonePipeline?: ClonePipelineState;
  deletePipeline?: DeletePipelineState;
  pluginOptions?: PluginOptionsState;
  validatePipeline?: ValidatePipelineState;
  children: ReactNode;
}

export function PipelinesOverridesProvider({
  pipelinesList,
  activatePipeline,
  clonePipeline,
  deletePipeline,
  pluginOptions,
  validatePipeline,
  children,
}: PipelinesOverridesProviderProps) {
  return (
    <PipelinesListOverrideContext.Provider value={pipelinesList ?? null}>
      <ActivatePipelineOverrideContext.Provider value={activatePipeline ?? null}>
        <ClonePipelineOverrideContext.Provider value={clonePipeline ?? null}>
          <DeletePipelineOverrideContext.Provider value={deletePipeline ?? null}>
            <PluginOptionsOverrideContext.Provider value={pluginOptions ?? null}>
              <ValidatePipelineOverrideContext.Provider
                value={validatePipeline ?? null}
              >
                {children}
              </ValidatePipelineOverrideContext.Provider>
            </PluginOptionsOverrideContext.Provider>
          </DeletePipelineOverrideContext.Provider>
        </ClonePipelineOverrideContext.Provider>
      </ActivatePipelineOverrideContext.Provider>
    </PipelinesListOverrideContext.Provider>
  );
}
