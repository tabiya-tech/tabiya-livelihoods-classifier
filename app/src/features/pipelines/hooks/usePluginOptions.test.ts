import { describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { createElement } from "react";
import type { PluginOptionItem, PluginOptionsResponse } from "@/lib/api";
import { PluginOptionsOverrideContext } from "./pipelinesOverrides";
import { usePluginOptions } from "./usePluginOptions";
import type { PluginOptionsState } from "./usePluginOptions";

describe("usePluginOptions", () => {
  it("fetches options and transitions to 'ready' on success", async () => {
    // GIVEN a plugin id, field, and expected options
    const givenPluginId = "tabiya.ner.v1";
    const givenField = "model_id";
    const expectedOptions: PluginOptionItem[] = [
      { value: "tabiya/roberta-base-job-ner", label: "Roberta base (job NER)" },
      {
        value: "tabiya/roberta-large-job-ner",
        label: "Roberta large (job NER)",
      },
    ];
    const fetchOptions = vi.fn(
      async (): Promise<PluginOptionsResponse> => ({
        field: givenField,
        options: expectedOptions,
      }),
    );

    // WHEN the hook is rendered with that plugin id and field
    const { result } = renderHook(() =>
      usePluginOptions(givenPluginId, givenField, { fetchOptions }),
    );

    // THEN it transitions from loading to ready with the expected options
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.options).toEqual(expectedOptions);
    expect(result.current.error).toBeNull();
  });

  it("transitions to 'error' when the fetch throws", async () => {
    // GIVEN a fetch function that rejects with a known error message
    const givenPluginId = "tabiya.ner.v1";
    const givenField = "model_id";
    const givenErrorMessage = "Network failure";
    const fetchOptions = vi.fn(async (): Promise<PluginOptionsResponse> => {
      throw new Error(givenErrorMessage);
    });

    // WHEN the hook is rendered
    const { result } = renderHook(() =>
      usePluginOptions(givenPluginId, givenField, { fetchOptions }),
    );

    // THEN it transitions to error with the error populated
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.options).toEqual([]);
    expect(result.current.error?.message).toBe(givenErrorMessage);
  });

  it("stays 'idle' when pluginId is an empty string", async () => {
    // GIVEN an empty pluginId
    const givenPluginId = "";
    const givenField = "model_id";
    const fetchOptions = vi.fn();

    // WHEN the hook is rendered with an empty pluginId
    const { result } = renderHook(() =>
      usePluginOptions(givenPluginId, givenField, { fetchOptions }),
    );

    // THEN it stays idle and never calls fetch
    expect(result.current.status).toBe("idle");
    expect(result.current.options).toEqual([]);
    expect(fetchOptions).not.toHaveBeenCalled();
  });

  it("returns the override state without fetching when wrapped in PluginOptionsOverrideContext", async () => {
    // GIVEN an override state and a fetch spy that should never be called
    const givenOverrideState: PluginOptionsState = {
      status: "ready",
      options: [{ value: "override-value", label: "Override Label" }],
      error: null,
    };
    const fetchOptions = vi.fn();

    // AND the hook is rendered inside the override context
    const wrapper = ({ children }: { children: React.ReactNode }) =>
      createElement(
        PluginOptionsOverrideContext.Provider,
        { value: givenOverrideState },
        children,
      );

    // WHEN the hook is rendered
    const { result } = renderHook(
      () => usePluginOptions("tabiya.ner.v1", "model_id", { fetchOptions }),
      { wrapper },
    );

    // THEN it immediately returns the override state and never calls fetch
    expect(result.current.status).toBe(givenOverrideState.status);
    expect(result.current.options).toEqual(givenOverrideState.options);
    expect(result.current.error).toBeNull();
    expect(fetchOptions).not.toHaveBeenCalled();
  });
});
