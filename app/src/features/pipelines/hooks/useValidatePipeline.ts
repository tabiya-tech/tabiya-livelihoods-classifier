/**
 * Validates a pipeline's stages against the backend, with a 300 ms debounce.
 *
 * - Stays "idle" when stages array is empty.
 * - Moves to "checking" after the debounce fires, then to "valid" / "invalid" /
 *   "error" once the response arrives.
 * - Supports an override context so Storybook stories can inject canned state.
 */

import { useContext, useEffect, useState } from "react";
import {
  validatePipeline,
  type PipelineStage,
  type PipelineValidationIssue,
  type ValidatePipelineRequest,
  type ValidatePipelineResponse,
} from "@/lib/api";
import { ValidatePipelineOverrideContext } from "./pipelinesOverrides";

export type ValidatePipelineStatus =
  | "idle"
  | "checking"
  | "valid"
  | "invalid"
  | "error";

export interface ValidatePipelineState {
  status: ValidatePipelineStatus;
  issues: PipelineValidationIssue[];
  error: Error | null;
}

export interface UseValidatePipelineOptions {
  /** Test / Storybook seam — override the backend mutation. Defaults to the real client. */
  validateFn?: (
    payload: ValidatePipelineRequest,
  ) => Promise<ValidatePipelineResponse>;
}

const DEBOUNCE_MS = 300;

export function useValidatePipeline(
  stages: PipelineStage[],
  { validateFn = validatePipeline }: UseValidatePipelineOptions = {},
): ValidatePipelineState {
  const override = useContext(ValidatePipelineOverrideContext);
  const [state, setState] = useState<ValidatePipelineState>({
    status: "idle",
    issues: [],
    error: null,
  });

  useEffect(() => {
    if (override) return;
    if (stages.length === 0) {
      setState({ status: "idle", issues: [], error: null });
      return;
    }

    const debounceTimer = setTimeout(() => {
      setState({ status: "checking", issues: [], error: null });

      validateFn({ stages })
        .then((response) => {
          setState({
            status: response.valid ? "valid" : "invalid",
            issues: response.issues,
            error: null,
          });
        })
        .catch((caught: unknown) => {
          const validationError =
            caught instanceof Error ? caught : new Error(String(caught));
          setState({ status: "error", issues: [], error: validationError });
        });
    }, DEBOUNCE_MS);

    return () => {
      clearTimeout(debounceTimer);
    };
  }, [stages, validateFn, override]);

  if (override) return override;
  return state;
}
