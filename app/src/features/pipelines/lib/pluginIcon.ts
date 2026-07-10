/**
 * Maps a plugin's backend-declared `icon` string (and its category) to a
 * concrete frontend {@link IconName}.
 *
 * Backend manifests declare domain icon names ("ner", "nel", "text",
 * "results", "docs", …) that don't all exist in the frontend icon set. Before
 * this map, StageNode hard-coded `"config"` for every node — which renders as
 * a gear/sun glyph, so every node looked identical ("the sun icon"). Here we
 * translate to the real icons, falling back by category, then to a neutral
 * default.
 */

import type { IconName } from "@/components/Icon/Icon.types";
import type { PluginCategory } from "@/lib/api";

/** Direct map from a manifest `icon` value to a frontend icon. */
const ICON_BY_MANIFEST_VALUE: Record<string, IconName> = {
  ner: "search",
  nel: "pipelines",
  text: "docs",
  json: "docs",
  results: "check",
  docs: "docs",
  classify: "classify",
  filter: "filter",
  scraper: "globe",
  language_router: "globe",
};

/** Fallback icon per plugin category when the manifest icon is unmapped. */
const ICON_BY_CATEGORY: Record<PluginCategory, IconName> = {
  source: "upload",
  core: "classify",
  transform: "filter",
  sink: "download",
};

const DEFAULT_ICON: IconName = "config";

/**
 * Resolve the icon for a plugin. Prefers the manifest's declared icon, then
 * the category default, then a neutral fallback.
 */
export function pluginIcon(
  manifestIcon: string | undefined,
  category: PluginCategory | undefined,
): IconName {
  if (manifestIcon && manifestIcon in ICON_BY_MANIFEST_VALUE) {
    return ICON_BY_MANIFEST_VALUE[manifestIcon];
  }
  if (category && category in ICON_BY_CATEGORY) {
    return ICON_BY_CATEGORY[category];
  }
  return DEFAULT_ICON;
}
