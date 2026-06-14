import { describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import type { NelModel } from "@/lib/api";
import { useNelModels } from "./useNelModels";

describe("useNelModels", () => {
  it("starts in loading state, then transitions to ready with the fetched models", async () => {
    // GIVEN a fetcher that returns a known list of models
    const givenModels: NelModel[] = [
      { model_id: "a", dimensions: 384, description: "fast" },
      { model_id: "b", dimensions: 768, description: "balanced" },
    ];
    const fetchModels = vi.fn(async () => givenModels);

    // WHEN we mount the hook
    const { result } = renderHook(() => useNelModels({ fetchModels }));

    // THEN the initial snapshot is loading
    expect(result.current.status).toBe("loading");

    // AND once the request resolves, the snapshot exposes the fetched models
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.models).toEqual(givenModels);
    expect(result.current.error).toBeNull();
  });

  it("transitions to error when the fetcher rejects", async () => {
    // GIVEN a fetcher that throws
    const givenError = new Error("backend offline");
    const fetchModels = vi.fn(async () => {
      throw givenError;
    });

    // WHEN we mount the hook
    const { result } = renderHook(() => useNelModels({ fetchModels }));

    // THEN the snapshot eventually reports the error and an empty list
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toBe(givenError);
    expect(result.current.models).toEqual([]);
  });
});
