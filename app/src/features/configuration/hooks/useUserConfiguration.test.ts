import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { V2UserConfig } from "@/lib/api";
import { useUserConfiguration } from "./useUserConfiguration";

const givenInitialConfig: V2UserConfig = {
  nel_model_id: "mpnet-base-v2",
  taxonomy_model_id: "esco-1.2.0",
};

const givenNextConfig: V2UserConfig = {
  nel_model_id: "tabiya-job-bge",
  taxonomy_model_id: "esco-1.2.0",
};

describe("useUserConfiguration", () => {
  it("loads the persisted config and seeds the draft from it", async () => {
    // GIVEN a fetcher that resolves with the initial config
    const fetchConfig = vi.fn(async () => givenInitialConfig);
    const persistConfig = vi.fn(async (next: V2UserConfig) => next);

    // WHEN we mount the hook
    const { result } = renderHook(() =>
      useUserConfiguration({ fetchConfig, persistConfig }),
    );

    // THEN it starts in loading state, then transitions to ready
    expect(result.current.loadStatus).toBe("loading");
    await waitFor(() => expect(result.current.loadStatus).toBe("ready"));

    // AND the saved and draft snapshots both match the fetched config
    expect(result.current.saved).toEqual(givenInitialConfig);
    expect(result.current.draft).toEqual(givenInitialConfig);
    expect(result.current.isDirty).toBe(false);
  });

  it("marks the state dirty when the draft diverges from the saved config", async () => {
    // GIVEN a loaded hook
    const fetchConfig = vi.fn(async () => givenInitialConfig);
    const persistConfig = vi.fn(async (next: V2UserConfig) => next);
    const { result } = renderHook(() =>
      useUserConfiguration({ fetchConfig, persistConfig }),
    );
    await waitFor(() => expect(result.current.loadStatus).toBe("ready"));

    // WHEN the caller updates the draft NEL model
    act(() => {
      result.current.setDraft({ nel_model_id: givenNextConfig.nel_model_id });
    });

    // THEN the snapshot reports dirty and the draft reflects the change
    expect(result.current.isDirty).toBe(true);
    expect(result.current.draft?.nel_model_id).toBe(
      givenNextConfig.nel_model_id,
    );
    // AND the persisted config is still the original
    expect(result.current.saved?.nel_model_id).toBe(
      givenInitialConfig.nel_model_id,
    );
  });

  it("rolls back the draft when discard() is called", async () => {
    // GIVEN a hook with a pending change
    const fetchConfig = vi.fn(async () => givenInitialConfig);
    const persistConfig = vi.fn(async (next: V2UserConfig) => next);
    const { result } = renderHook(() =>
      useUserConfiguration({ fetchConfig, persistConfig }),
    );
    await waitFor(() => expect(result.current.loadStatus).toBe("ready"));
    act(() => {
      result.current.setDraft({ nel_model_id: givenNextConfig.nel_model_id });
    });
    expect(result.current.isDirty).toBe(true);

    // WHEN the caller discards
    act(() => result.current.discard());

    // THEN the draft is rolled back and the dirty flag clears
    expect(result.current.draft).toEqual(givenInitialConfig);
    expect(result.current.isDirty).toBe(false);
  });

  it("persists the draft on save() and updates the saved snapshot", async () => {
    // GIVEN a hook with a pending change
    const fetchConfig = vi.fn(async () => givenInitialConfig);
    const persistConfig = vi.fn(async (next: V2UserConfig) => next);
    const { result } = renderHook(() =>
      useUserConfiguration({ fetchConfig, persistConfig }),
    );
    await waitFor(() => expect(result.current.loadStatus).toBe("ready"));
    act(() => {
      result.current.setDraft({ nel_model_id: givenNextConfig.nel_model_id });
    });

    // WHEN the caller saves
    await act(async () => {
      await result.current.save();
    });

    // THEN persist is called with the draft, saveStatus is 'saved', and the
    // saved snapshot reflects the new value
    expect(persistConfig).toHaveBeenCalledWith({
      ...givenInitialConfig,
      nel_model_id: givenNextConfig.nel_model_id,
    });
    expect(result.current.saveStatus).toBe("saved");
    expect(result.current.saved?.nel_model_id).toBe(
      givenNextConfig.nel_model_id,
    );
    expect(result.current.isDirty).toBe(false);
    expect(result.current.savedAt).toBeTypeOf("number");
  });

  it("keeps the dirty flag and reports saveStatus='error' when persistence fails", async () => {
    // GIVEN a hook where persistConfig rejects
    const givenSaveError = new Error("backend offline");
    const fetchConfig = vi.fn(async () => givenInitialConfig);
    const persistConfig = vi.fn(async () => {
      throw givenSaveError;
    });
    const { result } = renderHook(() =>
      useUserConfiguration({ fetchConfig, persistConfig }),
    );
    await waitFor(() => expect(result.current.loadStatus).toBe("ready"));
    act(() => {
      result.current.setDraft({ nel_model_id: givenNextConfig.nel_model_id });
    });

    // WHEN the caller attempts to save
    await act(async () => {
      await expect(result.current.save()).rejects.toBe(givenSaveError);
    });

    // THEN saveStatus is 'error', the error is surfaced, and dirty stays true
    expect(result.current.saveStatus).toBe("error");
    expect(result.current.saveError).toBe(givenSaveError);
    expect(result.current.isDirty).toBe(true);
    expect(result.current.draft?.nel_model_id).toBe(
      givenNextConfig.nel_model_id,
    );
  });

  it("reports loadStatus='error' when the initial fetch rejects", async () => {
    // GIVEN a fetcher that rejects
    const givenLoadError = new Error("not authorized");
    const fetchConfig = vi.fn(async () => {
      throw givenLoadError;
    });
    const persistConfig = vi.fn();

    // WHEN we mount the hook
    const { result } = renderHook(() =>
      useUserConfiguration({ fetchConfig, persistConfig }),
    );

    // THEN it surfaces the error and keeps saved/draft empty
    await waitFor(() => expect(result.current.loadStatus).toBe("error"));
    expect(result.current.loadError).toBe(givenLoadError);
    expect(result.current.saved).toBeNull();
    expect(result.current.draft).toBeNull();
  });

  it("returns saveStatus to 'idle' when the user edits after a recent save", async () => {
    // GIVEN a hook that has just saved successfully
    const fetchConfig = vi.fn(async () => givenInitialConfig);
    const persistConfig = vi.fn(async (next: V2UserConfig) => next);
    const { result } = renderHook(() =>
      useUserConfiguration({ fetchConfig, persistConfig }),
    );
    await waitFor(() => expect(result.current.loadStatus).toBe("ready"));
    act(() => result.current.setDraft({ nel_model_id: "tabiya-job-bge" }));
    await act(async () => {
      await result.current.save();
    });
    expect(result.current.saveStatus).toBe("saved");

    // WHEN the user edits the draft again
    act(() => result.current.setDraft({ taxonomy_model_id: "esco-1.1.1" }));

    // THEN saveStatus resets to idle so the SaveBar can show "Unsaved changes" again
    expect(result.current.saveStatus).toBe("idle");
    expect(result.current.isDirty).toBe(true);
  });
});
