import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { Pipeline } from "@/lib/api";
import { useClonePipeline } from "./useClonePipeline";

const givenSourcePipeline: Pipeline = {
  pipeline_id: "pipeline-001",
  user_id: "local-user",
  name: "Recruiter tuning",
  stages: [],
  is_active: false,
  is_default: false,
  is_readonly: false,
  created_at: "2026-01-01T00:00:00.000Z",
  updated_at: "2026-06-15T00:00:00.000Z",
};

const givenClonedPipeline: Pipeline = {
  ...givenSourcePipeline,
  pipeline_id: "pipeline-001-clone",
  name: "Recruiter tuning (copy)",
};

describe("useClonePipeline", () => {
  it("transitions to idle and calls onSuccess with the cloned pipeline on success", async () => {
    // GIVEN a clone function that resolves with the cloned pipeline and a success spy
    const givenPipelineId = givenSourcePipeline.pipeline_id;
    const cloneFn = vi.fn(async () => givenClonedPipeline);
    const onSuccess = vi.fn();

    // WHEN we render the hook and call clone
    const { result } = renderHook(() => useClonePipeline({ cloneFn, onSuccess }));
    await act(async () => {
      await result.current.clone(givenPipelineId);
    });

    // THEN status returns to idle and onSuccess was called with the cloned pipeline
    expect(result.current.status).toBe("idle");
    expect(result.current.pendingId).toBeNull();
    expect(onSuccess).toHaveBeenCalledWith(givenClonedPipeline);
  });

  it("sets pendingId while the clone request is in-flight", async () => {
    // GIVEN a clone function that we can intercept mid-flight
    const givenPipelineId = givenSourcePipeline.pipeline_id;
    let resolveFlight!: (pipeline: Pipeline) => void;
    const cloneFn = vi.fn(
      () =>
        new Promise<Pipeline>((resolve) => {
          resolveFlight = resolve;
        }),
    );

    // WHEN we render the hook and start the clone
    const { result } = renderHook(() => useClonePipeline({ cloneFn }));
    act(() => {
      void result.current.clone(givenPipelineId);
    });

    // THEN pendingId is set while in-flight
    await waitFor(() =>
      expect(result.current.pendingId).toBe(givenPipelineId),
    );
    expect(result.current.status).toBe("submitting");

    // AND pendingId clears after the request resolves
    await act(async () => {
      resolveFlight(givenClonedPipeline);
    });
    expect(result.current.pendingId).toBeNull();
  });

  it("transitions to error and rejects when the clone function throws", async () => {
    // GIVEN a clone function that rejects
    const givenPipelineId = givenSourcePipeline.pipeline_id;
    const givenErrorMessage = "Clone failed";
    const cloneFn = vi.fn(async () => {
      throw new Error(givenErrorMessage);
    });

    // WHEN we render the hook and call clone
    const { result } = renderHook(() => useClonePipeline({ cloneFn }));
    await act(async () => {
      await expect(
        result.current.clone(givenPipelineId),
      ).rejects.toThrow(givenErrorMessage);
    });

    // THEN status is error and the error is set
    expect(result.current.status).toBe("error");
    expect(result.current.error?.message).toBe(givenErrorMessage);
    expect(result.current.pendingId).toBeNull();
  });
});
