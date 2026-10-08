/**
 * Owns the lifecycle for a single Classifier run. One round-trip to
 * POST /v2/classify; transitions through idle → running → done | error.
 *
 * The page re-runs by calling {@link ClassifyState.run} again. Each call
 * supersedes the previous in-flight request — only the latest response wins
 * if two runs overlap (the older one's resolution is ignored).
 */

import { useCallback, useContext, useRef, useState } from "react";
import {
  classify,
  type ClassifyRequest,
  type ClassifyResponse,
} from "@/lib/api";
import { ClassifyOverrideContext } from "./classifierOverrides";

export type ClassifyStatus = "idle" | "running" | "done" | "error";

export interface ClassifyState {
  status: ClassifyStatus;
  /** The latest successful response, if any. */
  response: ClassifyResponse | null;
  /** The most recent error, if any. */
  error: Error | null;
  /**
   * Fire a classify run. Resolves with the response or rejects on failure.
   *
   * Callers may pass a `pipelineId` — the hook merges it into the payload
   * as `pipeline_id`. Left alone, the backend uses the caller's active
   * pipeline. If the payload already carries a `pipeline_id`, the caller's
   * explicit value wins.
   */
  run: (
    payload: ClassifyRequest,
    pipelineId?: string,
  ) => Promise<ClassifyResponse>;
  /** Drop state back to idle, clearing any prior response/error. */
  reset: () => void;
}

export interface UseClassifyOptions {
  /** Test seam — override the backend call. Defaults to the real client. */
  classifyImpl?: (payload: ClassifyRequest) => Promise<ClassifyResponse>;
}

export function useClassify({
  classifyImpl = classify,
}: UseClassifyOptions = {}): ClassifyState {
  const override = useContext(ClassifyOverrideContext);

  const [status, setStatus] = useState<ClassifyStatus>("idle");
  const [response, setResponse] = useState<ClassifyResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);

  // Generation counter — guards against an older run resolving after a
  // newer one and overwriting the latest result.
  const runGenerationRef = useRef(0);

  const run = useCallback(
    async (payload: ClassifyRequest, pipelineId?: string) => {
      const myGeneration = ++runGenerationRef.current;
      setStatus("running");
      setError(null);
      // Merge in pipeline_id only when the caller passed one AND the payload
      // does not already carry an explicit value. This keeps callers that
      // already build the full payload themselves in control.
      const payloadWithPipeline: ClassifyRequest =
        pipelineId != null && payload.pipeline_id == null
          ? { ...payload, pipeline_id: pipelineId }
          : payload;
      try {
        const result = await classifyImpl(payloadWithPipeline);
        if (myGeneration === runGenerationRef.current) {
          setResponse(result);
          setStatus("done");
        }
        return result;
      } catch (caught: unknown) {
        const caughtError =
          caught instanceof Error ? caught : new Error(String(caught));
        if (myGeneration === runGenerationRef.current) {
          setError(caughtError);
          setStatus("error");
        }
        throw caughtError;
      }
    },
    [classifyImpl],
  );

  const reset = useCallback(() => {
    // Bump the generation so any in-flight request's resolution is ignored.
    runGenerationRef.current += 1;
    setStatus("idle");
    setResponse(null);
    setError(null);
  }, []);

  if (override) return override;
  return { status, response, error, run, reset };
}
