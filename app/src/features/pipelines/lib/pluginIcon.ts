/**
 * Resolves the icon for a pipeline plugin to a concrete frontend {@link IconName}.
 *
 * A node must show a sensible icon even before its manifest has loaded (and
 * coming-soon plugins never load one), so this derives an icon from THREE
 * signals, most specific first:
 *   1. the manifest's declared `icon` (mapped, or used directly if it's already
 *      a valid frontend icon name),
 *   2. the plugin_id (well-known plugins + the `.source.`/`.sink.`/`.transform.`
 *      segment),
 *   3. the category.
 * Only when none of those resolve do we fall back to the neutral gear icon —
 * so the "sun on every node" happens only when we truly know nothing.
 */

import type { IconName } from "@/components/Icon/Icon.types";
import type { PluginCategory } from "@/lib/api";

/** The frontend icon set — used to accept a manifest icon value verbatim. */
const VALID_ICON_NAMES: ReadonlySet<string> = new Set<IconName>([
  "classify",
  "dashboard",
  "config",
  "key",
  "docs",
  "history",
  "copy",
  "arrowRight",
  "external",
  "plus",
  "trash",
  "check",
  "close",
  "filter",
  "download",
  "upload",
  "spark",
  "search",
  "globe",
  "pipelines",
]);

/** Map domain-specific manifest icon values to frontend icons. */
const ICON_BY_MANIFEST_VALUE: Record<string, IconName> = {
  ner: "search",
  nel: "pipelines",
  text: "docs",
  json: "docs",
  results: "check",
  scraper: "globe",
  stopwords: "filter",
  language_router: "globe",
  database: "download",
};

/** Match a plugin_id to an icon by well-known id or its category segment. */
function iconFromPluginId(pluginId: string | undefined): IconName | null {
  if (!pluginId) return null;
  const id = pluginId.toLowerCase();
  if (id.includes(".ner.")) return "search";
  if (id.includes(".nel.")) return "pipelines";
  if (id.includes("json")) return "docs";
  if (id.includes("scraper")) return "globe";
  if (id.includes("stopword")) return "filter";
  if (id.includes("language_router") || id.includes("router")) return "globe";
  if (id.includes("database")) return "download";
  if (id.includes(".source.")) return "upload";
  if (id.includes(".sink.")) return "download";
  if (id.includes(".transform.")) return "filter";
  return null;
}

/** Fallback icon per plugin category. */
const ICON_BY_CATEGORY: Record<PluginCategory, IconName> = {
  source: "upload",
  core: "classify",
  transform: "filter",
  sink: "download",
};

const DEFAULT_ICON: IconName = "config";

/**
 * Resolve the icon for a plugin. `manifestIcon`/`category` come from the loaded
 * manifest (when available); `pluginId` is always present on a node and lets us
 * pick a good icon even with no manifest.
 */
export function pluginIcon(
  manifestIcon: string | undefined,
  category: PluginCategory | undefined,
  pluginId?: string,
): IconName {
  if (manifestIcon) {
    if (manifestIcon in ICON_BY_MANIFEST_VALUE) {
      return ICON_BY_MANIFEST_VALUE[manifestIcon];
    }
    // The manifest may already declare a valid frontend icon (e.g. "globe").
    if (VALID_ICON_NAMES.has(manifestIcon)) {
      return manifestIcon as IconName;
    }
  }
  const fromId = iconFromPluginId(pluginId);
  if (fromId) return fromId;
  if (category && category in ICON_BY_CATEGORY) {
    return ICON_BY_CATEGORY[category];
  }
  return DEFAULT_ICON;
}
