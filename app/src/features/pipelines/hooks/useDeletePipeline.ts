/**
 * "Delete pipeline" lifecycle. Tracks which pipeline id is in-flight so
 * the row can disable its actions while the request is pending.
 */

import { useCallback, useContext, useState } from "react";
import { deletePipeline } from "@/lib/api";
import { DeletePipelineOverrideContext } from "./pipelinesOverrides";

export type DeletePipelineStatus = "idle" | "submitting" | "error";

export interface DeletePipelineState {
  status: DeletePipelineStatus;
  error: Error | null;
  /** pipeline_id currently being deleted, or null. */
  pendingId: string | null;
  /** Submit a delete. Rejects on backend failure. */
  delete: (pipelineId: string) => Promise<void>;
}

export interface UseDeletePipelineOptions {
  /** Test seam — override the backend mutation. Defaults to the real client. */
  deleteFn?: (pipelineId: string) => Promise<void>;
  /** Optional side-effect after a successful delete — typically a list refetch. */
  onSuccess?: (pipelineId: string) => void | Promise<void>;
}

export function useDeletePipeline({
  deleteFn = deletePipeline,
  onSuccess,
}: UseDeletePipelineOptions = {}): DeletePipelineState {
  const override = useContext(DeletePipelineOverrideContext);
  const [status, setStatus] = useState<DeletePipelineStatus>("idle");
  const [error, setError] = useState<Error | null>(null);
  const [pendingId, setPendingId] = useState<string | null>(null);

  const deleteById = useCallback(
    async (pipelineId: string) => {
      setStatus("submitting");
      setPendingId(pipelineId);
      setError(null);
      try {
        await deleteFn(pipelineId);
        setStatus("idle");
        setPendingId(null);
        if (onSuccess) await onSuccess(pipelineId);
      } catch (caught: unknown) {
        const caughtError =
          caught instanceof Error ? caught : new Error(String(caught));
        setError(caughtError);
        setStatus("error");
        setPendingId(null);
        throw caughtError;
      }
    },
    [deleteFn, onSuccess],
  );

  if (override) return override;
  return { status, error, pendingId, delete: deleteById };
}
