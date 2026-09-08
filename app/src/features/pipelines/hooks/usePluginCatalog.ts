import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { listPlugins, type PluginSummary } from "@/lib/api";

export type PluginCatalogStatus = "loading" | "ready" | "error";

export interface PluginCatalogSnapshot {
  status: PluginCatalogStatus;
  plugins: PluginSummary[];
  error: Error | null;
  /** Re-runs the list query. Resolves once state has been updated. */
  refetch: () => Promise<void>;
}

export const PluginCatalogOverrideContext =
  createContext<PluginCatalogSnapshot | null>(null);

export interface UsePluginCatalogOptions {
  /** Test seam — override the backend fetch. Defaults to the real client. */
  fetchPlugins?: () => Promise<{ plugins: PluginSummary[] }>;
}

export function usePluginCatalog({
  fetchPlugins = listPlugins,
}: UsePluginCatalogOptions = {}): PluginCatalogSnapshot {
  const override = useContext(PluginCatalogOverrideContext);
  const [snapshot, setSnapshot] = useState<Omit<PluginCatalogSnapshot, "refetch">>({
    status: "loading",
    plugins: [],
    error: null,
  });

  const load = useCallback(async () => {
    try {
      const response = await fetchPlugins();
      setSnapshot({ status: "ready", plugins: response.plugins, error: null });
    } catch (caught: unknown) {
      const error =
        caught instanceof Error ? caught : new Error(String(caught));
      setSnapshot({ status: "error", plugins: [], error });
    }
  }, [fetchPlugins]);

  useEffect(() => {
    if (override) return;
    void load();
  }, [load, override]);

  if (override) return override;
  return { ...snapshot, refetch: load };
}
