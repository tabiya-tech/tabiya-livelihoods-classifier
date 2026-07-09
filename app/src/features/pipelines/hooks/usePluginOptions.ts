/**
 * Fetches the dynamic option list for one field of a plugin configuration.
 *
 * - Stays "idle" when pluginId or field are empty strings.
 * - Transitions through "loading" → "ready" (or "error") once deps are set.
 * - Supports an override context so Storybook stories can inject canned state.
 */

import { useContext, useEffect, useState } from "react";
import {
  getPluginOptions,
  type PluginOptionItem,
  type PluginOptionsResponse,
} from "@/lib/api";
import { PluginOptionsOverrideContext } from "./pipelinesOverrides";

export type PluginOptionsStatus = "idle" | "loading" | "ready" | "error";

export interface PluginOptionsState {
  status: PluginOptionsStatus;
  options: PluginOptionItem[];
  error: Error | null;
}

export interface UsePluginOptionsOptions {
  /** Test / Storybook seam — override the backend fetch. Defaults to the real client. */
  fetchOptions?: (
    pluginId: string,
    field: string,
  ) => Promise<PluginOptionsResponse>;
}

export function usePluginOptions(
  pluginId: string,
  field: string,
  { fetchOptions = getPluginOptions }: UsePluginOptionsOptions = {},
): PluginOptionsState {
  const override = useContext(PluginOptionsOverrideContext);
  const [state, setState] = useState<PluginOptionsState>({
    status: "idle",
    options: [],
    error: null,
  });

  useEffect(() => {
    if (override) return;
    if (!pluginId || !field) {
      setState({ status: "idle", options: [], error: null });
      return;
    }

    let cancelled = false;
    setState({ status: "loading", options: [], error: null });

    fetchOptions(pluginId, field)
      .then((response) => {
        if (!cancelled) {
          setState({ status: "ready", options: response.options, error: null });
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          const fetchError =
            caught instanceof Error ? caught : new Error(String(caught));
          setState({ status: "error", options: [], error: fetchError });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [pluginId, field, fetchOptions, override]);

  if (override) return override;
  return state;
}
