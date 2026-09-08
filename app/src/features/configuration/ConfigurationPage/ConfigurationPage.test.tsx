import { beforeEach, describe, expect, it, vi } from "vitest";

// The real fetcher reads `auth.currentUser.getIdToken()` to mint the bearer
// token. In jsdom there is no Firebase session, so we mock the fetcher's
// firebase import and the v1 firebase module path. The MSW handlers then
// serve the network responses themselves.
// Mock the v2 API functions directly so the page hooks resolve against an
// in-memory store rather than the real fetcher (which requires a Firebase
// session). This keeps the test focused on the page's composition; the
// underlying hooks have their own dedicated tests with their fetch seams.
const apiMocks = vi.hoisted(() => {
  const fixtureUser = {
    nel_model_id: "mpnet-base-v2",
    taxonomy_model_id: "esco-1.2.0",
  };
  let currentConfig = { ...fixtureUser };
  return {
    fixtureUser,
    listNelModels: vi.fn(async () => [
      {
        model_id: "all-MiniLM-L6-v2",
        dimensions: 384,
        description: "Fast general-purpose sentence embedder.",
      },
      {
        model_id: "mpnet-base-v2",
        dimensions: 768,
        description: "Higher quality embeddings; ~2× slower.",
      },
      {
        model_id: "tabiya-job-bge",
        dimensions: 1024,
        description: "Fine-tuned on job-ad corpora.",
      },
    ]),
    listTaxonomyModels: vi.fn(async () => [
      {
        id: "esco-1.1.1",
        name: "ESCO",
        version: "v1.1.1",
        description: "Default.",
        released: true,
      },
      {
        id: "esco-1.2.0",
        name: "ESCO",
        version: "v1.2.0",
        description: "Latest ESCO release.",
        released: true,
      },
      {
        id: "isco-08",
        name: "ISCO",
        version: "08",
        description: "ILO classification.",
        released: true,
      },
    ]),
    getV2UserConfig: vi.fn(async () => ({ ...currentConfig })),
    saveV2UserConfig: vi.fn(async (config) => {
      currentConfig = { ...config };
      return { ...currentConfig };
    }),
    resetStore: () => {
      currentConfig = { ...fixtureUser };
    },
  };
});

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    listNelModels: apiMocks.listNelModels,
    listTaxonomyModels: apiMocks.listTaxonomyModels,
    getV2UserConfig: apiMocks.getV2UserConfig,
    saveV2UserConfig: apiMocks.saveV2UserConfig,
  };
});

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";

import { ToastProvider } from "@/components";
import { NavigationGuardProvider } from "@/lib/navigationGuard";
import {
  ConfigurationPage,
  DATA_TEST_ID,
} from "./ConfigurationPage";
import {
  DATA_TEST_ID as SAVE_BAR_DATA_TEST_ID,
} from "../components/SaveBar/SaveBar";
import {
  DATA_TEST_ID as STAGE_RAIL_DATA_TEST_ID,
} from "../components/StageRail/StageRail";
import { RADIO_CARD_DATA_TEST_ID } from "@/components";

function renderConfigurationPage() {
  return render(
    <ToastProvider>
      <NavigationGuardProvider>
        <ConfigurationPage />
      </NavigationGuardProvider>
    </ToastProvider>,
  );
}

describe("ConfigurationPage", () => {
  beforeEach(() => {
    apiMocks.resetStore();
    apiMocks.listNelModels.mockClear();
    apiMocks.listTaxonomyModels.mockClear();
    apiMocks.getV2UserConfig.mockClear();
    apiMocks.saveV2UserConfig.mockClear();
  });

  it("renders the page header and resolves the loading skeleton once data is fetched", async () => {
    // GIVEN the expected page header copy
    const expectedTitle = i18n.t("configuration.title");
    const expectedIntro = i18n.t("configuration.intro");

    // WHEN we render the page
    renderConfigurationPage();

    // THEN the header copy is present immediately
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(
      expectedTitle,
    );
    expect(screen.getByTestId(DATA_TEST_ID.INTRO)).toHaveTextContent(
      expectedIntro,
    );

    // AND the loading indicator clears once models + config land
    await waitFor(() =>
      expect(
        screen.queryByTestId(DATA_TEST_ID.LOADING),
      ).not.toBeInTheDocument(),
    );
  });

  it("seeds the NEL selection from the persisted user config", async () => {
    // GIVEN the fixture user config picks the second NEL model
    const givenSelectedNelModelId = apiMocks.fixtureUser.nel_model_id;

    // WHEN we render the page and wait for ready state
    renderConfigurationPage();
    await screen.findByTestId(DATA_TEST_ID.OPTION_LIST);

    // THEN the corresponding ModelOption is rendered selected
    const renderedOptions = screen.getAllByTestId(
      RADIO_CARD_DATA_TEST_ID.CONTAINER,
    );
    const selectedOption = renderedOptions.find(
      (option) => option.getAttribute("data-model-id") === givenSelectedNelModelId,
    );
    expect(selectedOption).toBeDefined();
    expect(selectedOption).toHaveAttribute("aria-checked", "true");
  });

  it("shows the save bar with the dirty copy after changing the NEL selection", async () => {
    // GIVEN a model id that differs from the persisted selection
    const allNelModels = await apiMocks.listNelModels();
    const givenOtherNelModelId = allNelModels.find(
      (model) => model.model_id !== apiMocks.fixtureUser.nel_model_id,
    )!.model_id;
    const expectedDirtyTitle = i18n.t("configuration.saveBar.dirtyTitle");

    // WHEN we render and click that other model
    renderConfigurationPage();
    await screen.findByTestId(DATA_TEST_ID.OPTION_LIST);
    const otherOption = screen
      .getAllByTestId(RADIO_CARD_DATA_TEST_ID.CONTAINER)
      .find(
        (option) => option.getAttribute("data-model-id") === givenOtherNelModelId,
      );
    expect(otherOption).toBeDefined();
    await userEvent.click(otherOption!);

    // THEN the save bar appears with the unsaved-changes copy
    const renderedSaveBar = await screen.findByTestId(
      SAVE_BAR_DATA_TEST_ID.CONTAINER,
    );
    expect(
      within(renderedSaveBar).getByTestId(SAVE_BAR_DATA_TEST_ID.TITLE),
    ).toHaveTextContent(expectedDirtyTitle);
  });

  it("switches the right-hand panel when a different stage is selected", async () => {
    // GIVEN the page in its ready state
    const expectedTaxonomyPanelTitle = i18n.t(
      "configuration.stages.taxonomy.panelTitle",
    );

    // WHEN we render and click the Taxonomy stage in the rail
    renderConfigurationPage();
    await screen.findByTestId(DATA_TEST_ID.OPTION_LIST);
    const taxonomyStage = screen
      .getAllByTestId(STAGE_RAIL_DATA_TEST_ID.ITEM)
      .find((stageButton) => stageButton.getAttribute("data-stage-id") === "taxonomy");
    expect(taxonomyStage).toBeDefined();
    await userEvent.click(taxonomyStage!);

    // THEN the panel title swaps to the Taxonomy panel
    expect(screen.getByTestId(DATA_TEST_ID.PANEL_TITLE)).toHaveTextContent(
      expectedTaxonomyPanelTitle,
    );
    // AND every taxonomy model surfaces as a radio option
    const renderedTaxonomyOptions = screen.getAllByTestId(
      RADIO_CARD_DATA_TEST_ID.CONTAINER,
    );
    const allTaxonomyModels = await apiMocks.listTaxonomyModels();
    expect(renderedTaxonomyOptions).toHaveLength(allTaxonomyModels.length);
  });

  it("persists the draft via PUT and transitions the save bar to 'saved'", async () => {
    // GIVEN a NEL model that differs from the persisted selection
    const allNelModels = await apiMocks.listNelModels();
    const givenOtherNelModelId = allNelModels.find(
      (model) => model.model_id !== apiMocks.fixtureUser.nel_model_id,
    )!.model_id;
    const expectedSavedTitle = i18n.t("configuration.saveBar.savedTitle");

    // WHEN we render, change the selection, and click Save
    renderConfigurationPage();
    await screen.findByTestId(DATA_TEST_ID.OPTION_LIST);
    const otherOption = screen
      .getAllByTestId(RADIO_CARD_DATA_TEST_ID.CONTAINER)
      .find(
        (option) => option.getAttribute("data-model-id") === givenOtherNelModelId,
      );
    await userEvent.click(otherOption!);
    await userEvent.click(
      await screen.findByTestId(SAVE_BAR_DATA_TEST_ID.SAVE_BUTTON),
    );

    // THEN the save bar transitions to the "saved" copy
    await waitFor(() => {
      expect(screen.getByTestId(SAVE_BAR_DATA_TEST_ID.TITLE)).toHaveTextContent(
        expectedSavedTitle,
      );
    });
    expect(
      screen.getByTestId(SAVE_BAR_DATA_TEST_ID.SAVED_INDICATOR),
    ).toBeInTheDocument();
  });
});
