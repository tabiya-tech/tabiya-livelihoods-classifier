import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import type { PluginSummary } from "@/lib/api";
import { fixturePluginSummaries } from "@/mocks/fixtures/plugins";
import {
  PluginCatalogOverrideContext,
  type PluginCatalogSnapshot,
} from "../../hooks/usePluginCatalog";
import { PluginPalette, DATA_TEST_ID } from "./PluginPalette";

const givenReadySnapshot: PluginCatalogSnapshot = {
  status: "ready",
  plugins: fixturePluginSummaries,
  error: null,
  refetch: async () => {},
};

const givenLoadingSnapshot: PluginCatalogSnapshot = {
  status: "loading",
  plugins: [],
  error: null,
  refetch: async () => {},
};

const givenErrorSnapshot: PluginCatalogSnapshot = {
  status: "error",
  plugins: [],
  error: new Error("Failed to fetch plugins"),
  refetch: async () => {},
};

function renderWithOverride(snapshot: PluginCatalogSnapshot) {
  return render(
    <PluginCatalogOverrideContext.Provider value={snapshot}>
      <PluginPalette />
    </PluginCatalogOverrideContext.Provider>,
  );
}

describe("PluginPalette", () => {
  it("renders one row per plugin in the catalog", () => {
    // GIVEN a catalog of 6 plugins
    const expectedRowCount = fixturePluginSummaries.length;

    // WHEN we render with the ready snapshot
    renderWithOverride(givenReadySnapshot);

    // THEN there is one row per plugin
    const pluginRows = screen.getAllByTestId(DATA_TEST_ID.PLUGIN_ROW);
    expect(pluginRows).toHaveLength(expectedRowCount);
  });

  it("renders category sections with category badges", () => {
    // GIVEN a catalog with plugins in multiple categories

    // WHEN we render
    renderWithOverride(givenReadySnapshot);

    // THEN category sections are present
    const categorySections = screen.getAllByTestId(DATA_TEST_ID.CATEGORY_SECTION);
    expect(categorySections.length).toBeGreaterThan(0);
  });

  it("shows the coming soon pill for coming_soon plugins", () => {
    // GIVEN the fixture catalog which has 2 coming_soon plugins
    const givenComingSoonPlugins: PluginSummary[] = fixturePluginSummaries.filter(
      (plugin) => plugin.coming_soon,
    );
    const expectedComingSoonCount = givenComingSoonPlugins.length;

    // WHEN we render
    renderWithOverride(givenReadySnapshot);

    // THEN there is one coming-soon pill per coming_soon plugin
    const comingSoonPills = screen.getAllByTestId(DATA_TEST_ID.COMING_SOON_PILL);
    expect(comingSoonPills).toHaveLength(expectedComingSoonCount);
  });

  it("does not show coming soon pill for available plugins", () => {
    // GIVEN a single available plugin (not coming_soon)
    const givenAvailablePlugin: PluginSummary = {
      plugin_id: "tabiya.ner.v1",
      name: "Tabiya NER",
      version: "0.1.0",
      category: "core",
      summary: "Named-entity recognition.",
      detail: null,
      icon: "ner",
      status: "enabled",
      coming_soon: false,
      last_error: null,
    };
    const snapshotWithOnePlugin = {
      ...givenReadySnapshot,
      plugins: [givenAvailablePlugin],
    };

    // WHEN we render
    renderWithOverride(snapshotWithOnePlugin);

    // THEN no coming-soon pill is rendered
    const comingSoonPills = screen.queryAllByTestId(DATA_TEST_ID.COMING_SOON_PILL);
    expect(comingSoonPills).toHaveLength(0);
  });

  it("renders the plugin name and summary in each row", () => {
    // GIVEN one known plugin
    const givenPlugin: PluginSummary = fixturePluginSummaries[0];
    const expectedName = givenPlugin.name;
    const expectedSummary = givenPlugin.summary;

    // WHEN we render
    renderWithOverride(givenReadySnapshot);

    // THEN that plugin's row contains its name and summary
    const allRows = screen.getAllByTestId(DATA_TEST_ID.PLUGIN_ROW);
    const matchingRow = allRows.find(
      (row) => row.getAttribute("data-plugin-id") === givenPlugin.plugin_id,
    );
    expect(matchingRow).toBeDefined();
    expect(within(matchingRow!).getByText(expectedName)).toBeInTheDocument();
    expect(within(matchingRow!).getByText(expectedSummary)).toBeInTheDocument();
  });

  it("shows a loading indicator while status is loading", () => {
    // GIVEN a loading snapshot

    // WHEN we render
    renderWithOverride(givenLoadingSnapshot);

    // THEN the loading element is shown
    expect(screen.getByTestId(DATA_TEST_ID.LOADING)).toBeInTheDocument();
  });

  it("shows an error message when status is error", () => {
    // GIVEN an error snapshot with a specific message
    const expectedErrorText = "Failed to fetch plugins";

    // WHEN we render
    renderWithOverride(givenErrorSnapshot);

    // THEN the error element shows the message
    const errorElement = screen.getByTestId(DATA_TEST_ID.ERROR);
    expect(errorElement).toHaveTextContent(expectedErrorText);
  });
});
