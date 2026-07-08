/**
 * "Activate pipeline" lifecycle. Tracks which pipeline id is in-flight so
 * the row can disable its toggle while the request is pending.
 */

import { useCallback, useContext, useState } from "react";
import { activatePipeline, type Pipeline } from "@/lib/api";
import { ActivatePipelineOverrideContext } from "./pipelinesOverrides";

export type ActivatePipelineStatus = "idle" | "submitting" | "error";

export interface ActivatePipelineState {
  status: ActivatePipelineStatus;
  error: Error | null;
  /** pipeline_id currently being activated, or null. */
  pendingId: string | null;
  /** Submit an activate. Rejects on backend failure. */
  activate: (pipelineId: string) => Promise<Pipeline>;
}

export interface UseActivatePipelineOptions {
  /** Test seam — override the backend mutation. Defaults to the real client. */
  activateFn?: (pipelineId: string) => Promise<Pipeline>;
  /** Optional side-effect after a successful activate — typically a list refetch. */
  onSuccess?: (pipeline: Pipeline) => void | Promise<void>;
}

export function useActivatePipeline({
  activateFn = activatePipeline,
  onSuccess,
}: UseActivatePipelineOptions = {}): ActivatePipelineState {
  const override = useContext(ActivatePipelineOverrideContext);
  const [status, setStatus] = useState<ActivatePipelineStatus>("idle");
  const [error, setError] = useState<Error | null>(null);
  const [pendingId, setPendingId] = useState<string | null>(null);

  const activate = useCallback(
    async (pipelineId: string) => {
      setStatus("submitting");
      setPendingId(pipelineId);
      setError(null);
      try {
        const pipeline = await activateFn(pipelineId);
        setStatus("idle");
        setPendingId(null);
        if (onSuccess) await onSuccess(pipeline);
        return pipeline;
      } catch (caught: unknown) {
        const caughtError =
          caught instanceof Error ? caught : new Error(String(caught));
        setError(caughtError);
        setStatus("error");
        setPendingId(null);
        throw caughtError;
      }
    },
    [activateFn, onSuccess],
  );

  if (override) return override;
  return { status, error, pendingId, activate };
}
