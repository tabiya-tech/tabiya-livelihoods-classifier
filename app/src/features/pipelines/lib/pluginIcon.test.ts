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

  it("falls back to a neutral default when icon and category are both unknown", () => {
    // GIVEN no manifest icon and no category
    const expectedIcon = "config";

    // WHEN we resolve the icon
    const actualIcon = pluginIcon(undefined, undefined);

    // THEN we get the neutral default
    expect(actualIcon).toBe(expectedIcon);
  });
});
