import { describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import type { TaxonomyModel } from "@/lib/api";
import { useTaxonomyModels } from "./useTaxonomyModels";

describe("useTaxonomyModels", () => {
  it("starts loading and exposes the fetched taxonomy models on success", async () => {
    // GIVEN a fetcher that returns a known list of taxonomy models
    const givenModels: TaxonomyModel[] = [
      {
        id: "esco-1.2.0",
        name: "ESCO",
        version: "v1.2.0",
        description: "current release",
        released: true,
      },
    ];
    const fetchModels = vi.fn(async () => givenModels);

    // WHEN we mount the hook
    const { result } = renderHook(() => useTaxonomyModels({ fetchModels }));

    // THEN the initial snapshot is loading and eventually transitions to ready
    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.models).toEqual(givenModels);
  });

  it("reports error status and an empty model list when the fetcher rejects", async () => {
    // GIVEN a fetcher that throws
    const givenError = new Error("taxonomy API offline");
    const fetchModels = vi.fn(async () => {
      throw givenError;
    });

    // WHEN we mount the hook
    const { result } = renderHook(() => useTaxonomyModels({ fetchModels }));

    // THEN the snapshot reports the error
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toBe(givenError);
    expect(result.current.models).toEqual([]);
  });
});
