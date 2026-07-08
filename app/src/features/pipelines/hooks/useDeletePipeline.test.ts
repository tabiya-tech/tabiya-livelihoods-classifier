import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useDeletePipeline } from "./useDeletePipeline";

describe("useDeletePipeline", () => {
  it("transitions to idle and calls onSuccess with the pipeline_id on success", async () => {
    // GIVEN a delete function that resolves and a success spy
    const givenPipelineId = "pipeline-001";
    const deleteFn = vi.fn(async () => undefined);
    const onSuccess = vi.fn();

    // WHEN we render the hook and call delete
    const { result } = renderHook(() =>
      useDeletePipeline({ deleteFn, onSuccess }),
    );
    await act(async () => {
      await result.current.delete(givenPipelineId);
    });

    // THEN status returns to idle and onSuccess was called with the pipeline_id
    expect(result.current.status).toBe("idle");
    expect(result.current.pendingId).toBeNull();
    expect(onSuccess).toHaveBeenCalledWith(givenPipelineId);
  });

  it("sets pendingId while the delete request is in-flight", async () => {
    // GIVEN a delete function that we can intercept mid-flight
    const givenPipelineId = "pipeline-001";
    let resolveFlight!: () => void;
    const deleteFn = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveFlight = resolve;
        }),
    );

    // WHEN we render the hook and start the delete
    const { result } = renderHook(() => useDeletePipeline({ deleteFn }));
    act(() => {
      void result.current.delete(givenPipelineId);
    });

    // THEN pendingId is set while in-flight
    await waitFor(() =>
      expect(result.current.pendingId).toBe(givenPipelineId),
    );
    expect(result.current.status).toBe("submitting");

    // AND pendingId clears after the request resolves
    await act(async () => {
      resolveFlight();
    });
    expect(result.current.pendingId).toBeNull();
  });

  it("transitions to error and rejects when the delete function throws", async () => {
    // GIVEN a delete function that rejects
    const givenPipelineId = "pipeline-001";
    const givenErrorMessage = "Delete failed";
    const deleteFn = vi.fn(async () => {
      throw new Error(givenErrorMessage);
    });

    // WHEN we render the hook and call delete
    const { result } = renderHook(() => useDeletePipeline({ deleteFn }));
    await act(async () => {
      await expect(
        result.current.delete(givenPipelineId),
      ).rejects.toThrow(givenErrorMessage);
    });

    // THEN status is error and the error is set
    expect(result.current.status).toBe("error");
    expect(result.current.error?.message).toBe(givenErrorMessage);
    expect(result.current.pendingId).toBeNull();
  });
});
