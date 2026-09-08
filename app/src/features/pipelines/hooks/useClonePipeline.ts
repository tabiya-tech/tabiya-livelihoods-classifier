/**
 * "Clone pipeline" lifecycle. Tracks which pipeline id is in-flight.
 * The cloned pipeline is passed to the onSuccess callback so callers
 * can act on it (e.g. navigate to the editor, trigger a list refetch).
 */

import { useCallback, useContext, useState } from "react";
import { clonePipeline, type Pipeline } from "@/lib/api";
import { ClonePipelineOverrideContext } from "./pipelinesOverrides";

export type ClonePipelineStatus = "idle" | "submitting" | "error";

export interface ClonePipelineState {
  status: ClonePipelineStatus;
  error: Error | null;
  /** pipeline_id currently being cloned, or null. */
  pendingId: string | null;
  /** Submit a clone. Resolves with the new pipeline. Rejects on backend failure. */
  clone: (pipelineId: string) => Promise<Pipeline>;
}

export interface UseClonePipelineOptions {
  /** Test seam — override the backend mutation. Defaults to the real client. */
  cloneFn?: (pipelineId: string) => Promise<Pipeline>;
  /** Optional side-effect after a successful clone — typically a list refetch. */
  onSuccess?: (clonedPipeline: Pipeline) => void | Promise<void>;
}

export function useClonePipeline({
  cloneFn = clonePipeline,
  onSuccess,
}: UseClonePipelineOptions = {}): ClonePipelineState {
  const override = useContext(ClonePipelineOverrideContext);
  const [status, setStatus] = useState<ClonePipelineStatus>("idle");
  const [error, setError] = useState<Error | null>(null);
  const [pendingId, setPendingId] = useState<string | null>(null);

  const clone = useCallback(
    async (pipelineId: string) => {
      setStatus("submitting");
      setPendingId(pipelineId);
      setError(null);
      try {
        const clonedPipeline = await cloneFn(pipelineId);
        setStatus("idle");
        setPendingId(null);
        if (onSuccess) await onSuccess(clonedPipeline);
        return clonedPipeline;
      } catch (caught: unknown) {
        const caughtError =
          caught instanceof Error ? caught : new Error(String(caught));
        setError(caughtError);
        setStatus("error");
        setPendingId(null);
        throw caughtError;
      }
    },
    [cloneFn, onSuccess],
  );

  if (override) return override;
  return { status, error, pendingId, clone };
}
