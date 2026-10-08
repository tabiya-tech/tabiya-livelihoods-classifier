import { createElement, type ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import type { PluginDetail } from "@/lib/api";
import { fixtureNerManifest } from "@/mocks/fixtures/plugins";
import {
  PluginDetailOverrideContext,
  usePluginDetail,
  type PluginDetailSnapshot,
} from "./usePluginDetail";

const givenNerDetail: PluginDetail = {
  plugin_id: fixtureNerManifest.plugin_id,
  status: "enabled",
  coming_soon: false,
  last_error: null,
  manifest: fixtureNerManifest,
};

describe("usePluginDetail", () => {
  it("resolves to status='ready' with the fetched detail", async () => {
    // GIVEN a fetcher that resolves to the NER plugin detail
    const givenPluginId = fixtureNerManifest.plugin_id;
    const fetchPlugin = vi.fn(async () => givenNerDetail);

    // WHEN we render the hook with that pluginId
    const { result } = renderHook(() =>
      usePluginDetail(givenPluginId, { fetchPlugin }),
    );

    // THEN the hook eventually reports ready with the returned detail
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.detail).toEqual(givenNerDetail);
    expect(result.current.error).toBeNull();
    expect(fetchPlugin).toHaveBeenCalledWith(givenPluginId);
  });

  it("transitions to status='error' when the fetch rejects", async () => {
    // GIVEN a fetcher that rejects with a known error
    const givenPluginId = fixtureNerManifest.plugin_id;
    const givenError = new Error("plugin not found");
    const fetchPlugin = vi.fn(async () => {
      throw givenError;
    });

    // WHEN we render the hook
    const { result } = renderHook(() =>
      usePluginDetail(givenPluginId, { fetchPlugin }),
    );

    // THEN we end in error and the error is surfaced verbatim
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.detail).toBeNull();
    expect(result.current.error).toBe(givenError);
  });

  it("short-circuits to the override context snapshot when provided", () => {
    // GIVEN an override snapshot in ready state
    const givenOverrideSnapshot: PluginDetailSnapshot = {
      status: "ready",
      detail: givenNerDetail,
      error: null,
    };
    const fetchPluginShouldNotFire = vi.fn(async () => {
      throw new Error("fetch must not be called when an override is set");
    });

    function OverrideWrapper({ children }: { children: ReactNode }) {
      return createElement(
        PluginDetailOverrideContext.Provider,
        { value: givenOverrideSnapshot },
        children,
      );
    }

    // WHEN we render the hook inside the override provider
    const { result } = renderHook(
      () =>
        usePluginDetail(fixtureNerManifest.plugin_id, {
          fetchPlugin: fetchPluginShouldNotFire,
        }),
      { wrapper: OverrideWrapper },
    );

    // THEN we get the override snapshot back and the fetcher was never called
    expect(result.current).toEqual(givenOverrideSnapshot);
    expect(fetchPluginShouldNotFire).not.toHaveBeenCalled();
  });
});
