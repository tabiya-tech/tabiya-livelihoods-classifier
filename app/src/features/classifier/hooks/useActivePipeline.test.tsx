import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { Pipeline } from "@/lib/api";
import {
  ActivePipelineOverrideContext,
} from "./classifierPipelineOverrides";
import {
  useActivePipeline,
  type ActivePipelineSnapshot,
} from "./useActivePipeline";

const givenActivePipeline: Pipeline = {
  pipeline_id: "pipeline-default",
  user_id: "local-user",
  name: "Default Tabiya",
  stages: [],
  is_active: true,
  is_default: true,
  is_readonly: true,
  created_at: "2026-01-01T00:00:00.000Z",
  updated_at: "2026-01-01T00:00:00.000Z",
};

const givenInactivePipeline: Pipeline = {
  pipeline_id: "pipeline-recruiter",
  user_id: "local-user",
  name: "Recruiter tuning",
  stages: [],
  is_active: false,
  is_default: false,
  is_readonly: false,
  created_at: "2026-02-01T00:00:00.000Z",
  updated_at: "2026-02-01T00:00:00.000Z",
};

describe("useActivePipeline", () => {
  it("loads pipelines and exposes the active one on success", async () => {
    // GIVEN a fetch that returns two pipelines, one active
    const givenPipelines = [givenActivePipeline, givenInactivePipeline];
    const fetchPipelines = vi.fn(async () => ({ pipelines: givenPipelines }));
    const activateFn = vi.fn(async () => givenActivePipeline);

    // WHEN we render the hook
    const { result } = renderHook(() =>
      useActivePipeline({ fetchPipelines, activateFn }),
    );

    // THEN it starts loading, then becomes ready with the pipelines and active pick
    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.pipelines).toEqual(givenPipelines);
    expect(result.current.activePipeline).toEqual(givenActivePipeline);
    expect(result.current.error).toBeNull();
  });

  it("transitions to error state when the fetch rejects", async () => {
    // GIVEN a fetch that rejects
    const givenErrorMessage = "Backend unreachable";
    const fetchPipelines = vi.fn(async () => {
      throw new Error(givenErrorMessage);
    });
    const activateFn = vi.fn(async () => givenActivePipeline);

    // WHEN we render the hook
    const { result } = renderHook(() =>
      useActivePipeline({ fetchPipelines, activateFn }),
    );

    // THEN it transitions to error and reports the failure
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.pipelines).toEqual([]);
    expect(result.current.activePipeline).toBeNull();
    expect(result.current.error?.message).toBe(givenErrorMessage);
  });

  it("optimistically flips active pipeline and calls activateFn on setActivePipeline", async () => {
    // GIVEN a hook that has loaded two pipelines
    const givenPipelines = [givenActivePipeline, givenInactivePipeline];
    const fetchPipelines = vi.fn(async () => ({ pipelines: givenPipelines }));
    const activateFn = vi.fn(async () => ({
      ...givenInactivePipeline,
      is_active: true,
    }));
    const { result } = renderHook(() =>
      useActivePipeline({ fetchPipelines, activateFn }),
    );
    await waitFor(() => expect(result.current.status).toBe("ready"));

    // WHEN the user switches to the other pipeline
    const targetPipelineId = givenInactivePipeline.pipeline_id;
    await act(async () => {
      await result.current.setActivePipeline(targetPipelineId);
    });

    // THEN the active pipeline flipped optimistically AND the backend was called
    expect(result.current.activePipeline?.pipeline_id).toBe(targetPipelineId);
    expect(activateFn).toHaveBeenCalledWith(targetPipelineId);
  });

  it("returns the override snapshot instead of fetching when the override context is set", async () => {
    // GIVEN an override snapshot provided via context
    const expectedActivePipeline = givenInactivePipeline;
    const overrideSnapshot: ActivePipelineSnapshot = {
      status: "ready",
      pipelines: [givenActivePipeline, givenInactivePipeline],
      activePipeline: expectedActivePipeline,
      error: null,
      setActivePipeline: vi.fn(async () => undefined),
    };
    const fetchPipelines = vi.fn(async () => ({ pipelines: [] }));
    const activateFn = vi.fn(async () => givenActivePipeline);

    function Wrapper({ children }: { children: ReactNode }) {
      return (
        <ActivePipelineOverrideContext.Provider value={overrideSnapshot}>
          {children}
        </ActivePipelineOverrideContext.Provider>
      );
    }

    // WHEN the hook renders under that provider
    const { result } = renderHook(
      () => useActivePipeline({ fetchPipelines, activateFn }),
      { wrapper: Wrapper },
    );

    // THEN the override is returned verbatim and the real fetch was never hit
    expect(result.current).toBe(overrideSnapshot);
    expect(fetchPipelines).not.toHaveBeenCalled();
  });
});
