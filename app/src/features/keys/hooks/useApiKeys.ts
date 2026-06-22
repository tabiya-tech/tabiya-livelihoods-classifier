/**
 * Loads the caller's active API keys and exposes a refetch hook for callers
 * that just mutated server state (create / revoke).
 */

import { useCallback, useContext, useEffect, useState } from "react";
import { listApiKeys, type ApiKeyMetadata } from "@/lib/api";
import { ApiKeysOverrideContext } from "./apiKeysOverrides";

export type ApiKeysStatus = "loading" | "ready" | "error";

export interface ApiKeysSnapshot {
  status: ApiKeysStatus;
  keys: ApiKeyMetadata[];
  error: Error | null;
  /** Re-runs the list query. Resolves once state has been updated. */
  refetch: () => Promise<void>;
}

export interface UseApiKeysOptions {
  /** Test seam — override the backend fetch. Defaults to the real client. */
  fetchKeys?: () => Promise<{ keys: ApiKeyMetadata[] }>;
}

export function useApiKeys({
  fetchKeys = listApiKeys,
}: UseApiKeysOptions = {}): ApiKeysSnapshot {
  const override = useContext(ApiKeysOverrideContext);
  const [snapshot, setSnapshot] = useState<Omit<ApiKeysSnapshot, "refetch">>({
    status: "loading",
    keys: [],
    error: null,
  });

  const load = useCallback(async () => {
    try {
      const response = await fetchKeys();
      setSnapshot({ status: "ready", keys: response.keys, error: null });
    } catch (caught: unknown) {
      const error =
        caught instanceof Error ? caught : new Error(String(caught));
      setSnapshot({ status: "error", keys: [], error });
    }
  }, [fetchKeys]);

  useEffect(() => {
    if (override) return;
    void load();
  }, [load, override]);

  if (override) return override;
  return { ...snapshot, refetch: load };
}
