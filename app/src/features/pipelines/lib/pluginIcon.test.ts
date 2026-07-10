import { describe, expect, it } from "vitest";
import { pluginIcon } from "./pluginIcon";

describe("pluginIcon", () => {
  it("maps a known manifest icon value to a concrete frontend icon", () => {
    // GIVEN the NER plugin's manifest icon value
    const givenManifestIcon = "ner";
    const expectedIcon = "search";

    // WHEN we resolve the icon
    const actualIcon = pluginIcon(givenManifestIcon, "core");

    // THEN it maps to the concrete icon, not the gear/sun default
    expect(actualIcon).toBe(expectedIcon);
  });

  it("falls back to the category icon when the manifest icon is unmapped", () => {
    // GIVEN an unmapped manifest icon on a source plugin
    const givenManifestIcon = "totally-unknown";
    const expectedIcon = "upload";

    // WHEN we resolve the icon
    const actualIcon = pluginIcon(givenManifestIcon, "source");

    // THEN it falls back to the source-category icon
    expect(actualIcon).toBe(expectedIcon);
  });

  it("derives an icon from the plugin_id when no manifest has loaded", () => {
    // GIVEN a NER node with no manifest and no category (manifest not loaded)
    const givenPluginId = "tabiya.ner.v1";
    const expectedIcon = "search";

    // WHEN we resolve using only the plugin_id
    const actualIcon = pluginIcon(undefined, undefined, givenPluginId);

    // THEN we still get a meaningful icon, NOT the gear/sun default
    expect(actualIcon).toBe(expectedIcon);
  });

  it("derives a source icon from the plugin_id's category segment", () => {
    // GIVEN an unknown source plugin id with no manifest
    const givenPluginId = "tabiya.source.somethingnew.v1";
    const expectedIcon = "upload";

    // WHEN we resolve using only the plugin_id
    const actualIcon = pluginIcon(undefined, undefined, givenPluginId);

    // THEN the `.source.` segment yields the source icon
    expect(actualIcon).toBe(expectedIcon);
  });

  it("accepts a manifest icon that is already a valid frontend icon name", () => {
    // GIVEN a manifest icon that is itself a valid icon (e.g. scraper's globe)
    const givenManifestIcon = "globe";
    const expectedIcon = "globe";

    // WHEN we resolve
    const actualIcon = pluginIcon(givenManifestIcon, "source");

    // THEN it's used verbatim
    expect(actualIcon).toBe(expectedIcon);
  });

  it("falls back to the gear icon only when nothing is known", () => {
    // GIVEN no manifest icon, no category, and no plugin_id
    const expectedIcon = "config";

    // WHEN we resolve
    const actualIcon = pluginIcon(undefined, undefined, undefined);

    // THEN the neutral default is the true last resort
    expect(actualIcon).toBe(expectedIcon);
  });
});
