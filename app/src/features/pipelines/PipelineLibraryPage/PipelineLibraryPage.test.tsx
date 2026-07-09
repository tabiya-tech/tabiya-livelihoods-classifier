import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { ReactFlowProvider } from "reactflow";
import type { ListPluginsResponse, PluginDetail } from "@/lib/api";
import {
  fixtureNerManifest,
  fixturePluginSummaries,
} from "@/mocks/fixtures/plugins";

// Mock the api module so listPlugins resolves without going through the
// auth-guarded fetcher (which rejects with "Not authenticated" in jsdom).
const apiMocks = vi.hoisted(() => ({
  listPlugins: vi.fn<() => Promise<ListPluginsResponse>>(),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    listPlugins: apiMocks.listPlugins,
  };
});

// Imports below intentionally follow the vi.mock call above.
import {
  PluginDetailOverrideContext,
  type PluginDetailSnapshot,
} from "../hooks/usePluginDetail";
import { DATA_TEST_ID as PLUGIN_PALETTE_DATA_TEST_ID } from "../components/PluginPalette/PluginPalette";
import { DATA_TEST_ID, PipelineLibraryPage } from "./PipelineLibraryPage";

const givenNerDetail: PluginDetail = {
  plugin_id: fixtureNerManifest.plugin_id,
  status: "enabled",
  coming_soon: false,
  last_error: null,
  manifest: fixtureNerManifest,
};

const givenReadyDetailSnapshot: PluginDetailSnapshot = {
  status: "ready",
  detail: givenNerDetail,
  error: null,
};

interface RenderLibraryPageOptions {
  detailOverride?: PluginDetailSnapshot;
}

function renderLibraryPage({
  detailOverride,
}: RenderLibraryPageOptions = {}) {
  return render(
    <MemoryRouter>
      <ReactFlowProvider>
        <PluginDetailOverrideContext.Provider value={detailOverride ?? null}>
          <PipelineLibraryPage />
        </PluginDetailOverrideContext.Provider>
      </ReactFlowProvider>
    </MemoryRouter>,
  );
}

describe("PipelineLibraryPage", () => {
  beforeEach(() => {
    apiMocks.listPlugins.mockResolvedValue({
      plugins: fixturePluginSummaries,
    });
  });

  it("renders one palette card per plugin fixture", async () => {
    // GIVEN the six-plugin fixture catalog
    const expectedPluginCount = fixturePluginSummaries.length;

    // WHEN we render the page
    renderLibraryPage();

    // THEN one palette row is present per plugin
    await waitFor(() => {
      const paletteRows = screen.getAllByTestId(
        PLUGIN_PALETTE_DATA_TEST_ID.PLUGIN_ROW,
      );
      expect(paletteRows).toHaveLength(expectedPluginCount);
    });
  });

  it("shows the empty prompt when no plugin has been selected", async () => {
    // GIVEN a freshly rendered library page (no selection yet)

    // WHEN we render
    renderLibraryPage();

    // THEN the empty-prompt element is visible
    await waitFor(() =>
      expect(
        screen.getByTestId(DATA_TEST_ID.DETAIL_EMPTY_PROMPT),
      ).toBeInTheDocument(),
    );
  });

  it("populates the detail panel with the plugin name when a card is clicked", async () => {
    // GIVEN an override that returns the NER detail whenever the hook fires
    const expectedPluginName = fixtureNerManifest.name;
    const user = userEvent.setup();
    renderLibraryPage({ detailOverride: givenReadyDetailSnapshot });

    // Wait for the palette to hydrate.
    await waitFor(() =>
      expect(
        screen.getAllByTestId(PLUGIN_PALETTE_DATA_TEST_ID.PLUGIN_ROW).length,
      ).toBeGreaterThan(0),
    );

    // WHEN we click the NER row
    const paletteRows = screen.getAllByTestId(
      PLUGIN_PALETTE_DATA_TEST_ID.PLUGIN_ROW,
    );
    const nerRow = paletteRows.find(
      (row) =>
        row.getAttribute("data-plugin-id") === fixtureNerManifest.plugin_id,
    );
    expect(nerRow).toBeDefined();
    await user.click(nerRow!);

    // THEN the detail panel shows the NER plugin name
    const detailPanel = screen.getByTestId(DATA_TEST_ID.DETAIL_PANEL);
    await waitFor(() =>
      expect(detailPanel).toHaveTextContent(expectedPluginName),
    );
  });

  it("lists the plugin's config schema fields after selection", async () => {
    // GIVEN the NER detail override (schema has `model_id` and `entity_types`)
    const expectedFieldName = "model_id";
    const user = userEvent.setup();
    renderLibraryPage({ detailOverride: givenReadyDetailSnapshot });

    await waitFor(() =>
      expect(
        screen.getAllByTestId(PLUGIN_PALETTE_DATA_TEST_ID.PLUGIN_ROW).length,
      ).toBeGreaterThan(0),
    );

    // WHEN we click the NER row
    const paletteRows = screen.getAllByTestId(
      PLUGIN_PALETTE_DATA_TEST_ID.PLUGIN_ROW,
    );
    const nerRow = paletteRows.find(
      (row) =>
        row.getAttribute("data-plugin-id") === fixtureNerManifest.plugin_id,
    );
    await user.click(nerRow!);

    // THEN a schema-field entry exists for `model_id`
    await waitFor(() => {
      const detailFields = screen.getAllByTestId(DATA_TEST_ID.DETAIL_FIELD);
      const matchingField = detailFields.find(
        (fieldElement) =>
          fieldElement.getAttribute("data-field-name") === expectedFieldName,
      );
      expect(matchingField).toBeDefined();
    });
  });

  it("does not populate the detail panel when a coming_soon plugin is clicked", async () => {
    // GIVEN the fixture catalog contains coming_soon plugins whose clicks
    // must be ignored (they only render as read-only cards)
    const givenComingSoonPlugin = fixturePluginSummaries.find(
      (plugin) => plugin.coming_soon,
    );
    expect(givenComingSoonPlugin).toBeDefined();
    const user = userEvent.setup();
    renderLibraryPage({ detailOverride: givenReadyDetailSnapshot });

    await waitFor(() =>
      expect(
        screen.getAllByTestId(PLUGIN_PALETTE_DATA_TEST_ID.PLUGIN_ROW).length,
      ).toBeGreaterThan(0),
    );

    // WHEN we click the coming_soon row
    const paletteRows = screen.getAllByTestId(
      PLUGIN_PALETTE_DATA_TEST_ID.PLUGIN_ROW,
    );
    const comingSoonRow = paletteRows.find(
      (row) =>
        row.getAttribute("data-plugin-id") ===
        givenComingSoonPlugin!.plugin_id,
    );
    expect(comingSoonRow).toBeDefined();
    await user.click(comingSoonRow!);

    // THEN the empty prompt still shows — no manifest was rendered
    expect(
      screen.getByTestId(DATA_TEST_ID.DETAIL_EMPTY_PROMPT),
    ).toBeInTheDocument();
  });
});
