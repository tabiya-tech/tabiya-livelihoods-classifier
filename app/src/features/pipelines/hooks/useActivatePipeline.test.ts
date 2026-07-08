import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { Pipeline } from "@/lib/api";
import { useActivatePipeline } from "./useActivatePipeline";

const givenActivatedPipeline: Pipeline = {
  pipeline_id: "pipeline-001",
  user_id: "local-user",
  name: "Default Tabiya",
  stages: [],
  is_active: true,
  is_default: true,
  is_readonly: true,
  created_at: "2026-01-01T00:00:00.000Z",
  updated_at: "2026-06-15T00:00:00.000Z",
};

describe("useActivatePipeline", () => {
  it("transitions to idle and calls onSuccess with the pipeline on success", async () => {
    // GIVEN an activate function that resolves and a success spy
    const givenPipelineId = givenActivatedPipeline.pipeline_id;
    const activateFn = vi.fn(async () => givenActivatedPipeline);
    const onSuccess = vi.fn();

    // WHEN we render the hook and call activate
    const { result } = renderHook(() =>
      useActivatePipeline({ activateFn, onSuccess }),
    );
    await act(async () => {
      await result.current.activate(givenPipelineId);
    });

    // THEN status returns to idle and onSuccess was called with the result
    expect(result.current.status).toBe("idle");
    expect(result.current.pendingId).toBeNull();
    expect(onSuccess).toHaveBeenCalledWith(givenActivatedPipeline);
  });

  it("sets pendingId while the activate request is in-flight", async () => {
    // GIVEN an activate function that we can intercept mid-flight
    const givenPipelineId = "pipeline-001";
    let resolveFlight!: (pipeline: Pipeline) => void;
    const activateFn = vi.fn(
      () =>
        new Promise<Pipeline>((resolve) => {
          resolveFlight = resolve;
        }),
    );

    // WHEN we render the hook and start the activate
    const { result } = renderHook(() => useActivatePipeline({ activateFn }));
    act(() => {
      void result.current.activate(givenPipelineId);
    });

    // THEN pendingId is set while in-flight
    await waitFor(() =>
      expect(result.current.pendingId).toBe(givenPipelineId),
    );
    expect(result.current.status).toBe("submitting");

    // AND pendingId clears after the request resolves
    await act(async () => {
      resolveFlight(givenActivatedPipeline);
    });
    expect(result.current.pendingId).toBeNull();
  });

  it("transitions to error and rejects when the activate function throws", async () => {
    // GIVEN an activate function that rejects
    const givenPipelineId = "pipeline-001";
    const givenErrorMessage = "Backend unreachable";
    const activateFn = vi.fn(async () => {
      throw new Error(givenErrorMessage);
    });

    // WHEN we render the hook and call activate
    const { result } = renderHook(() => useActivatePipeline({ activateFn }));
    await act(async () => {
      await expect(
        result.current.activate(givenPipelineId),
      ).rejects.toThrow(givenErrorMessage);
    });

    // THEN status is error and the error is set
    expect(result.current.status).toBe("error");
    expect(result.current.error?.message).toBe(givenErrorMessage);
    expect(result.current.pendingId).toBeNull();
  });
});
