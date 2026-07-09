/**
 * Loads a single pipeline by ID (or starts empty for a new pipeline) and
 * fetches the plugin manifests for every active plugin in the catalog.
 *
 * Both fetches run concurrently; the hook moves to "ready" only after both
 * resolve. A 404 on the pipeline fetch propagates as an error.
 */

import { useEffect, useState } from "react";
import {
  getPipeline,
  listPlugins,
  getPlugin,
  type ListPluginsResponse,
  type Pipeline,
  type PluginDetail,
  type PluginManifest,
  type PluginSummary,
} from "@/lib/api";

export type PipelineEditorStatus = "loading" | "ready" | "error";

export interface PipelineEditorSnapshot {
  status: PipelineEditorStatus;
  /** null for a new pipeline (no pipelineId provided). */
  pipeline: Pipeline | null;
  /** Manifests keyed by plugin_id for all non-coming-soon plugins. */
  manifests: Record<string, PluginManifest>;
  pluginSummaries: PluginSummary[];
  error: Error | null;
}

export interface UsePipelineEditorOptions {
  /** Undefined means "new pipeline" — no pipeline load will be attempted. */
  pipelineId?: string;
  /** Test seam — defaults to the real API client. */
  fetchPipeline?: (id: string) => Promise<Pipeline>;
  /** Test seam — defaults to the real API client. */
  fetchPlugins?: () => Promise<ListPluginsResponse>;
  /** Test seam — defaults to the real API client. */
  fetchPluginDetail?: (pluginId: string) => Promise<PluginDetail>;
}

export function usePipelineEditor({
  pipelineId,
  fetchPipeline = getPipeline,
  fetchPlugins = listPlugins,
  fetchPluginDetail = getPlugin,
}: UsePipelineEditorOptions = {}): PipelineEditorSnapshot {
  const [snapshot, setSnapshot] = useState<PipelineEditorSnapshot>({
    status: "loading",
    pipeline: null,
    manifests: {},
    pluginSummaries: [],
    error: null,
  });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        // Run pipeline load (if editing) and plugin catalog concurrently.
        const [pipelineResult, pluginsResponse] = await Promise.all([
          pipelineId ? fetchPipeline(pipelineId) : Promise.resolve(null),
          fetchPlugins(),
        ]);

        if (cancelled) return;

        const summaries = pluginsResponse.plugins;

        // Fetch manifests for all non-coming-soon plugins concurrently.
        const detailResults = await Promise.all(
          summaries
            .filter((summary) => !summary.coming_soon)
            .map((summary) =>
              fetchPluginDetail(summary.plugin_id).catch(() => null),
            ),
        );

        if (cancelled) return;

        const manifestsMap: Record<string, PluginManifest> = {};
        for (const detail of detailResults) {
          if (detail?.manifest) {
            manifestsMap[detail.plugin_id] = detail.manifest;
          }
        }

        setSnapshot({
          status: "ready",
          pipeline: pipelineResult,
          manifests: manifestsMap,
          pluginSummaries: summaries,
          error: null,
        });
      } catch (caught: unknown) {
        if (cancelled) return;
        const loadError =
          caught instanceof Error ? caught : new Error(String(caught));
        setSnapshot({
          status: "error",
          pipeline: null,
          manifests: {},
          pluginSummaries: [],
          error: loadError,
        });
      }
    }

    void load();

    return () => {
      cancelled = true;
    };
  }, [pipelineId, fetchPipeline, fetchPlugins, fetchPluginDetail]);

  return snapshot;
}
