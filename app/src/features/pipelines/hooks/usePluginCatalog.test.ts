import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { PluginSummary } from "@/lib/api";
import { usePluginCatalog } from "./usePluginCatalog";

const givenPlugins: PluginSummary[] = [
  {
    plugin_id: "tabiya.ner.v1",
    name: "Tabiya NER",
    version: "0.1.0",
    category: "core",
    summary: "Named-entity recognition over job-ad prose.",
    detail: null,
    icon: "ner",
    status: "enabled",
    coming_soon: false,
    last_error: null,
  },
];

describe("usePluginCatalog", () => {
  it("resolves to status='ready' with the fetched plugins", async () => {
    // GIVEN a fetcher that resolves to one plugin
    const fetchPlugins = vi.fn(async () => ({ plugins: givenPlugins }));

    // WHEN we render the hook
    const { result } = renderHook(() => usePluginCatalog({ fetchPlugins }));

    // THEN we end in ready with the supplied plugins
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.plugins).toEqual(givenPlugins);
    expect(result.current.error).toBeNull();
  });

  it("transitions to status='error' on fetch failure", async () => {
    // GIVEN a fetcher that rejects
    const givenError = new Error("network error");
    const fetchPlugins = vi.fn(async () => {
      throw givenError;
    });

    // WHEN we render the hook
    const { result } = renderHook(() => usePluginCatalog({ fetchPlugins }));

    // THEN we end in error with that exception surfaced
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.plugins).toEqual([]);
    expect(result.current.error).toBe(givenError);
  });

  it("re-runs the fetcher when refetch is invoked", async () => {
    // GIVEN a fetcher we can inspect for invocation count
    const fetchPlugins = vi.fn(async () => ({ plugins: givenPlugins }));
    const { result } = renderHook(() => usePluginCatalog({ fetchPlugins }));
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(fetchPlugins).toHaveBeenCalledTimes(1);

    // WHEN refetch is called
    await act(async () => {
      await result.current.refetch();
    });

    // THEN the fetcher fires again
    expect(fetchPlugins).toHaveBeenCalledTimes(2);
  });
});
