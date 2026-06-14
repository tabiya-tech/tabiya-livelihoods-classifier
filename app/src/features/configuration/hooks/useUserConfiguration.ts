/**
 * Owns the draft/save lifecycle for the Configuration page.
 *
 * - On mount: fetches the persisted user config via GET /v2/nel/user/config
 *   and seeds the draft from the response.
 * - The page mutates the draft via {@link setDraft}.
 * - {@link isDirty} compares draft vs. last-saved.
 * - {@link save} PUTs the draft; on success it updates the saved snapshot and
 *   transitions through `saving → saved`. {@link savedAt} timestamps the last
 *   success so the UI can fade out the "Saved" state.
 * - {@link discard} rolls the draft back to the last-saved snapshot.
 *
 * Errors don't clear the dirty flag — the caller (Configuration page) shows
 * a toast and the user can retry.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getV2UserConfig,
  saveV2UserConfig,
  type V2UserConfig,
} from "@/lib/api";

export type ConfigurationLoadStatus = "loading" | "ready" | "error";
export type ConfigurationSaveStatus =
  | "idle"
  | "saving"
  | "saved"
  | "error";

export interface UserConfigurationState {
  /** Status of the initial GET. */
  loadStatus: ConfigurationLoadStatus;
  /** Last error from the initial GET, if any. */
  loadError: Error | null;
  /** Status of the most recent PUT (or "idle" if none yet). */
  saveStatus: ConfigurationSaveStatus;
  /** Last error from the most recent PUT, if any. */
  saveError: Error | null;
  /** The currently-persisted config (last successful GET / PUT). */
  saved: V2UserConfig | null;
  /** The in-flight draft the UI is editing. */
  draft: V2UserConfig | null;
  /** True when draft and saved differ. */
  isDirty: boolean;
  /** Timestamp of the most recent successful save, if any. */
  savedAt: number | null;
  /** Replace the draft (partial — merges with current). */
  setDraft: (patch: Partial<V2UserConfig>) => void;
  /** Roll the draft back to the saved snapshot. */
  discard: () => void;
  /** PUT the draft. Resolves on success, rejects on failure. */
  save: () => Promise<void>;
}

export interface UseUserConfigurationOptions {
  /** Test seam — override the GET. Defaults to the real client. */
  fetchConfig?: () => Promise<V2UserConfig>;
  /** Test seam — override the PUT. Defaults to the real client. */
  persistConfig?: (config: V2UserConfig) => Promise<V2UserConfig>;
}

function configsEqual(a: V2UserConfig | null, b: V2UserConfig | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.nel_model_id === b.nel_model_id &&
    a.taxonomy_model_id === b.taxonomy_model_id
  );
}

export function useUserConfiguration({
  fetchConfig = getV2UserConfig,
  persistConfig = saveV2UserConfig,
}: UseUserConfigurationOptions = {}): UserConfigurationState {
  const [loadStatus, setLoadStatus] =
    useState<ConfigurationLoadStatus>("loading");
  const [loadError, setLoadError] = useState<Error | null>(null);
  const [saved, setSaved] = useState<V2UserConfig | null>(null);
  const [draft, setDraftState] = useState<V2UserConfig | null>(null);
  const [saveStatus, setSaveStatus] =
    useState<ConfigurationSaveStatus>("idle");
  const [saveError, setSaveError] = useState<Error | null>(null);
  const [savedAt, setSavedAt] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchConfig()
      .then((config) => {
        if (cancelled) return;
        setSaved(config);
        setDraftState(config);
        setLoadStatus("ready");
      })
      .catch((caught: unknown) => {
        if (cancelled) return;
        const error =
          caught instanceof Error ? caught : new Error(String(caught));
        setLoadError(error);
        setLoadStatus("error");
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setDraft = useCallback((patch: Partial<V2UserConfig>) => {
    setDraftState((previous) => {
      if (!previous) return previous;
      return { ...previous, ...patch };
    });
    // Editing while a previous "saved" indicator is still showing returns the
    // status to "idle" so the SaveBar swaps back to the dirty state.
    setSaveStatus((previousStatus) =>
      previousStatus === "saved" ? "idle" : previousStatus,
    );
  }, []);

  const discard = useCallback(() => {
    setDraftState(saved);
    setSaveStatus("idle");
    setSaveError(null);
  }, [saved]);

  const save = useCallback(async () => {
    if (!draft) return;
    setSaveStatus("saving");
    setSaveError(null);
    try {
      const persisted = await persistConfig(draft);
      setSaved(persisted);
      setDraftState(persisted);
      setSavedAt(Date.now());
      setSaveStatus("saved");
    } catch (caught) {
      const error =
        caught instanceof Error ? caught : new Error(String(caught));
      setSaveError(error);
      setSaveStatus("error");
      throw error;
    }
  }, [draft, persistConfig]);

  const isDirty = useMemo(() => !configsEqual(saved, draft), [saved, draft]);

  return {
    loadStatus,
    loadError,
    saveStatus,
    saveError,
    saved,
    draft,
    isDirty,
    savedAt,
    setDraft,
    discard,
    save,
  };
}
