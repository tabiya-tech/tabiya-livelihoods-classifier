/**
 * Reads the caller's pipelines, exposes the currently-active one, and lets
 * the Classifier page switch it.
 *
 * The switch is optimistic — we flip the "active" flag in local state as
 * soon as the user picks a new pipeline, then send activatePipeline in the
 * background. If the backend rejects, we roll back and surface the error.
 */

import { useCallback, useContext, useEffect, useState } from "react";
import {
  activatePipeline as defaultActivatePipeline,
  listPipelines as defaultListPipelines,
  type ListPipelinesResponse,
  type Pipeline,
} from "@/lib/api";
import { ActivePipelineOverrideContext } from "./classifierPipelineOverrides";

export type ActivePipelineStatus = "loading" | "ready" | "error";

export interface ActivePipelineSnapshot {
  status: ActivePipelineStatus;
  pipelines: Pipeline[];
  activePipeline: Pipeline | null;
  error: Error | null;
  /**
   * Optimistically switches active pipeline; calls activatePipeline behind
   * the scenes. Rejects (and rolls back) if the backend errors out.
   */
  setActivePipeline: (pipelineId: string) => Promise<void>;
}

export interface UseActivePipelineOptions {
  /** Test seam — override the backend list fetch. Defaults to the real client. */
  fetchPipelines?: () => Promise<ListPipelinesResponse>;
  /** Test seam — override the backend activate mutation. Defaults to the real client. */
  activateFn?: (pipelineId: string) => Promise<Pipeline>;
}

interface InternalState {
  status: ActivePipelineStatus;
  pipelines: Pipeline[];
  error: Error | null;
}

function pickActive(pipelines: Pipeline[]): Pipeline | null {
  return pipelines.find((pipeline) => pipeline.is_active) ?? null;
}

function applyActiveSwitch(
  pipelines: Pipeline[],
  activePipelineId: string,
): Pipeline[] {
  return pipelines.map((pipeline) => ({
    ...pipeline,
    is_active: pipeline.pipeline_id === activePipelineId,
  }));
}

export function useActivePipeline({
  fetchPipelines = defaultListPipelines,
  activateFn = defaultActivatePipeline,
}: UseActivePipelineOptions = {}): ActivePipelineSnapshot {
  const override = useContext(ActivePipelineOverrideContext);

  const [state, setState] = useState<InternalState>({
    status: "loading",
    pipelines: [],
    error: null,
  });

  useEffect(() => {
    if (override) return;
    let cancelled = false;
    (async () => {
      try {
        const response = await fetchPipelines();
        if (cancelled) return;
        setState({
          status: "ready",
          pipelines: response.pipelines,
          error: null,
        });
      } catch (caught: unknown) {
        if (cancelled) return;
        const caughtError =
          caught instanceof Error ? caught : new Error(String(caught));
        setState({ status: "error", pipelines: [], error: caughtError });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [fetchPipelines, override]);

  const setActivePipeline = useCallback(
    async (pipelineId: string) => {
      // Capture the pipelines BEFORE the optimistic flip so we can roll back
      // if the backend rejects. Reading via functional setState avoids stale
      // closures if the caller invokes this repeatedly.
      let priorPipelines: Pipeline[] = [];
      setState((previous) => {
        priorPipelines = previous.pipelines;
        return {
          ...previous,
          pipelines: applyActiveSwitch(previous.pipelines, pipelineId),
        };
      });
      try {
        await activateFn(pipelineId);
      } catch (caught: unknown) {
        const caughtError =
          caught instanceof Error ? caught : new Error(String(caught));
        setState((previous) => ({
          ...previous,
          pipelines: priorPipelines,
          error: caughtError,
        }));
        throw caughtError;
      }
    },
    [activateFn],
  );

  if (override) return override;

  return {
    status: state.status,
    pipelines: state.pipelines,
    activePipeline: pickActive(state.pipelines),
    error: state.error,
    setActivePipeline,
  };
}
