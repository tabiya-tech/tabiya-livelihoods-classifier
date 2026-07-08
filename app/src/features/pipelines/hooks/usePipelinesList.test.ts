import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { Pipeline } from "@/lib/api";
import { usePipelinesList } from "./usePipelinesList";

const givenPipeline: Pipeline = {
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

describe("usePipelinesList", () => {
  it("transitions to ready with the pipeline list on success", async () => {
    // GIVEN a fetch function that resolves with one pipeline
    const givenPipelines = [givenPipeline];
    const fetchPipelines = vi.fn(async () => ({ pipelines: givenPipelines }));

    // WHEN we render the hook
    const { result } = renderHook(() => usePipelinesList({ fetchPipelines }));

    // THEN it starts loading and transitions to ready with the data
    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.pipelines).toEqual(givenPipelines);
    expect(result.current.error).toBeNull();
  });

  it("transitions to error with the caught error on fetch failure", async () => {
    // GIVEN a fetch function that rejects
    const givenErrorMessage = "Network error";
    const fetchPipelines = vi.fn(async () => {
      throw new Error(givenErrorMessage);
    });

    // WHEN we render the hook
    const { result } = renderHook(() => usePipelinesList({ fetchPipelines }));

    // THEN it transitions to error with the error set
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.pipelines).toEqual([]);
    expect(result.current.error?.message).toBe(givenErrorMessage);
  });

  it("re-runs the fetcher when refetch is called", async () => {
    // GIVEN a fetch function that resolves
    const fetchPipelines = vi.fn(async () => ({ pipelines: [givenPipeline] }));

    // WHEN we render the hook and wait for ready, then call refetch
    const { result } = renderHook(() => usePipelinesList({ fetchPipelines }));
    await waitFor(() => expect(result.current.status).toBe("ready"));
    const callCountBeforeRefetch = fetchPipelines.mock.calls.length;

    await act(async () => {
      await result.current.refetch();
    });

    // THEN the fetcher was called again
    expect(fetchPipelines.mock.calls.length).toBeGreaterThan(
      callCountBeforeRefetch,
    );
  });
});
