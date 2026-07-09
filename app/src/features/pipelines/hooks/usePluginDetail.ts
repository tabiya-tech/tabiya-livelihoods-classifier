/**
 * Loads a single plugin's full manifest via `GET /v2/plugins/{plugin_id}`.
 * Mirrors the shape of {@link usePluginCatalog} — override-context aware so
 * Storybook / tests can inject a canned detail without hitting the network.
 *
 * Stays in "idle" while the caller has no selection (empty pluginId), so the
 * hook can safely be mounted at page level and only fires when needed.
 */

import { createContext, useContext, useEffect, useState } from "react";
import { getPlugin, type PluginDetail } from "@/lib/api";

export type PluginDetailStatus = "idle" | "loading" | "ready" | "error";

export interface PluginDetailSnapshot {
  status: PluginDetailStatus;
  detail: PluginDetail | null;
  error: Error | null;
}

export const PluginDetailOverrideContext =
  createContext<PluginDetailSnapshot | null>(null);

export interface UsePluginDetailOptions {
  /** Test seam — override the backend fetch. Defaults to the real client. */
  fetchPlugin?: (pluginId: string) => Promise<PluginDetail>;
}

export function usePluginDetail(
  pluginId: string,
  { fetchPlugin = getPlugin }: UsePluginDetailOptions = {},
): PluginDetailSnapshot {
  const override = useContext(PluginDetailOverrideContext);
  const [snapshot, setSnapshot] = useState<PluginDetailSnapshot>({
    status: "idle",
    detail: null,
    error: null,
  });

  useEffect(() => {
    if (override) return;

    if (!pluginId) {
      setSnapshot({ status: "idle", detail: null, error: null });
      return;
    }

    let cancelled = false;
    setSnapshot({ status: "loading", detail: null, error: null });

    fetchPlugin(pluginId)
      .then((detail) => {
        if (cancelled) return;
        setSnapshot({ status: "ready", detail, error: null });
      })
      .catch((caught: unknown) => {
        if (cancelled) return;
        const caughtError =
          caught instanceof Error ? caught : new Error(String(caught));
        setSnapshot({ status: "error", detail: null, error: caughtError });
      });

    return () => {
      cancelled = true;
    };
  }, [fetchPlugin, override, pluginId]);

  if (override) return override;
  return snapshot;
}
