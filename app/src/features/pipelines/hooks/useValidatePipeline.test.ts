import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { createElement } from "react";
import type {
  PipelineStage,
  PipelineValidationIssue,
  ValidatePipelineRequest,
  ValidatePipelineResponse,
} from "@/lib/api";
import { ValidatePipelineOverrideContext } from "./pipelinesOverrides";
import { useValidatePipeline } from "./useValidatePipeline";
import type { ValidatePipelineState } from "./useValidatePipeline";

const givenValidStages: PipelineStage[] = [
  { plugin_id: "tabiya.source.text.v1", config: { text: "" } },
  { plugin_id: "tabiya.ner.v1", config: {} },
  {
    plugin_id: "tabiya.nel.v1",
    config: {
      nel_model_id: "all-MiniLM-L6-v2",
      taxonomy_model_id: "esco-v1.2",
      top_k: 5,
      min_similarity: 0.0,
    },
  },
  { plugin_id: "tabiya.sink.results.v1", config: {} },
];

const givenTooFewStages: PipelineStage[] = [
  { plugin_id: "tabiya.ner.v1", config: {} },
];

describe("useValidatePipeline", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("stays 'idle' when the stages array is empty", async () => {
    // GIVEN an empty stages array
    const givenEmptyStages: PipelineStage[] = [];
    const validateFn = vi.fn();

    // WHEN the hook is rendered with empty stages
    const { result } = renderHook(() =>
      useValidatePipeline(givenEmptyStages, { validateFn }),
    );

    // AND the debounce timer fires
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    // THEN it stays idle and never calls validate
    expect(result.current.status).toBe("idle");
    expect(validateFn).not.toHaveBeenCalled();
  });

  it("transitions to 'valid' with zero issues after debounce when stages pass validation", async () => {
    // GIVEN a validate function that returns valid with no issues
    const expectedIssues: PipelineValidationIssue[] = [];
    const validateFn = vi.fn(
      async (_payload: ValidatePipelineRequest): Promise<ValidatePipelineResponse> => ({
        valid: true,
        issues: expectedIssues,
      }),
    );

    // AND the hook is rendered with valid stages
    const { result } = renderHook(() =>
      useValidatePipeline(givenValidStages, { validateFn }),
    );

    // WHEN the debounce timer fires and the async validate resolves
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    // THEN it transitions to valid with no issues
    expect(result.current.status).toBe("valid");
    expect(result.current.issues).toEqual(expectedIssues);
    expect(result.current.error).toBeNull();
  });

  it("transitions to 'invalid' with issues after debounce when stages fail validation", async () => {
    // GIVEN a validate function that returns invalid with one issue
    const expectedIssue: PipelineValidationIssue = {
      code: "too_few_stages",
      message: "A pipeline needs at least a Source and a Sink stage.",
      stage_index: null,
      plugin_id: null,
    };
    const validateFn = vi.fn(
      async (_payload: ValidatePipelineRequest): Promise<ValidatePipelineResponse> => ({
        valid: false,
        issues: [expectedIssue],
      }),
    );

    // AND the hook is rendered with a single-stage pipeline (invalid)
    const { result } = renderHook(() =>
      useValidatePipeline(givenTooFewStages, { validateFn }),
    );

    // WHEN the debounce timer fires and the async validate resolves
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    // THEN it transitions to invalid with the reported issue
    expect(result.current.status).toBe("invalid");
    expect(result.current.issues).toHaveLength(1);
    expect(result.current.issues[0]).toEqual(expectedIssue);
    expect(result.current.error).toBeNull();
  });

  it("returns the override state without validating when wrapped in ValidatePipelineOverrideContext", async () => {
    // GIVEN an override state and a validate spy that should never be called
    const givenOverrideState: ValidatePipelineState = {
      status: "valid",
      issues: [],
      error: null,
    };
    const validateFn = vi.fn();

    // AND the hook is rendered inside the override context
    const wrapper = ({ children }: { children: React.ReactNode }) =>
      createElement(
        ValidatePipelineOverrideContext.Provider,
        { value: givenOverrideState },
        children,
      );

    // WHEN the hook is rendered
    const { result } = renderHook(
      () => useValidatePipeline(givenValidStages, { validateFn }),
      { wrapper },
    );

    // AND the debounce timer fires
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    // THEN it returns the override state without ever calling validate
    expect(result.current.status).toBe(givenOverrideState.status);
    expect(result.current.issues).toEqual(givenOverrideState.issues);
    expect(result.current.error).toBeNull();
    expect(validateFn).not.toHaveBeenCalled();
  });
});
