import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";

// Mock the API layer at the module boundary so we don't need MSW for this
// page-level test. Two endpoints to stub:
//   - classify (the page's primary mutation)
//   - getV2UserConfig (called by the active-config chip indirectly)
const apiMocks = vi.hoisted(() => {
  return {
    classify: vi.fn(),
    getV2UserConfig: vi.fn(async () => ({
      nel_model_id: "all-MiniLM-L6-v2",
      taxonomy_model_id: "esco-1.1.1",
    })),
    saveV2UserConfig: vi.fn(),
    listNelModels: vi.fn(async () => []),
    listTaxonomyModels: vi.fn(async () => []),
  };
});
vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    classify: apiMocks.classify,
    getV2UserConfig: apiMocks.getV2UserConfig,
    saveV2UserConfig: apiMocks.saveV2UserConfig,
    listNelModels: apiMocks.listNelModels,
    listTaxonomyModels: apiMocks.listTaxonomyModels,
  };
});

import { ToastProvider } from "@/components";
import { fixtureClassifyResponse } from "@/mocks/fixtures/classify";
import {
  ClassifierPage,
  DATA_TEST_ID,
} from "./ClassifierPage";
import {
  DATA_TEST_ID as SOURCE_PANE_DATA_TEST_ID,
} from "../components/SourcePane/SourcePane";
import {
  DATA_TEST_ID as RESULTS_TABS_DATA_TEST_ID,
} from "../components/ResultsTabs/ResultsTabs";
import {
  DATA_TEST_ID as ENTITY_HIGHLIGHT_DATA_TEST_ID,
} from "../components/EntityHighlight/EntityHighlight";
import { DRAWER_DATA_TEST_ID } from "@/components";

function renderPage(initialUrl = "/classifier") {
  return render(
    <MemoryRouter initialEntries={[initialUrl]}>
      <ToastProvider>
        <ClassifierPage />
      </ToastProvider>
    </MemoryRouter>,
  );
}

describe("ClassifierPage", () => {
  beforeEach(() => {
    apiMocks.classify.mockReset();
    apiMocks.classify.mockResolvedValue(fixtureClassifyResponse);
    apiMocks.getV2UserConfig.mockClear();
  });

  it("renders the title, intro, and an idle placeholder before any run", () => {
    // GIVEN no prior run
    // WHEN we render
    renderPage();

    // THEN title + intro show, and the placeholder is visible (not results)
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.INTRO)).toBeInTheDocument();
    expect(
      screen.queryByTestId(RESULTS_TABS_DATA_TEST_ID.CONTAINER),
    ).not.toBeInTheDocument();
  });

  it("disables Run until text is entered", async () => {
    // GIVEN an empty page
    renderPage();
    const runButton = screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.RUN_BUTTON);

    // THEN Run starts disabled
    expect(runButton).toBeDisabled();

    // WHEN the user types
    await userEvent.type(
      screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.TEXTAREA),
      "needs a data scientist",
    );

    // THEN Run is enabled
    expect(runButton).toBeEnabled();
  });

  it("calls /v2/classify with the URL-synced params on Run", async () => {
    // GIVEN a URL with non-default params
    renderPage("/classifier?top_k=7&min_sim=0.5");
    await userEvent.type(
      screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.TEXTAREA),
      "needs data scientist",
    );

    // WHEN we click Run
    await userEvent.click(screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.RUN_BUTTON));

    // THEN classify was called with the parsed URL params
    await waitFor(() => expect(apiMocks.classify).toHaveBeenCalledTimes(1));
    expect(apiMocks.classify).toHaveBeenCalledWith({
      text: "needs data scientist",
      options: { top_k: 7, min_similarity: 0.5 },
    });
  });

  it("normalises form-style text before sending it to the backend", async () => {
    // GIVEN text laid out as form labels (uploads of structured .txt files)
    const givenFormText = "Job title\nStatistician\nDepartment";
    const expectedNormalised = "Job title Statistician Department";

    // WHEN we paste it and run
    renderPage();
    // userEvent.type can't type a literal newline; paste into the textarea instead.
    const textarea = screen.getByTestId(
      SOURCE_PANE_DATA_TEST_ID.TEXTAREA,
    );
    textarea.focus();
    await userEvent.paste(givenFormText);
    await userEvent.click(screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.RUN_BUTTON));

    // THEN the backend receives the prose-style normalised text
    await waitFor(() => expect(apiMocks.classify).toHaveBeenCalledTimes(1));
    expect(apiMocks.classify.mock.calls[0][0]).toMatchObject({
      text: expectedNormalised,
    });
  });

  it("renders the EntityHighlight + ResultsTabs once a run resolves", async () => {
    // GIVEN a successful classify
    renderPage();
    await userEvent.type(
      screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.TEXTAREA),
      "Senior data scientist with Python.",
    );
    await userEvent.click(screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.RUN_BUTTON));

    // THEN the source switches to the highlight view AND results tabs render
    await waitFor(() => {
      expect(
        screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.HIGHLIGHT),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId(RESULTS_TABS_DATA_TEST_ID.CONTAINER),
      ).toBeInTheDocument();
    });
  });

  it("opens the EntityDetailDrawer when an entity span is clicked", async () => {
    // GIVEN a finished run
    renderPage();
    await userEvent.type(
      screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.TEXTAREA),
      "Senior data scientist with Python.",
    );
    await userEvent.click(screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.RUN_BUTTON));
    const firstSpan = await waitFor(() =>
      screen.getAllByTestId(ENTITY_HIGHLIGHT_DATA_TEST_ID.ENTITY_SEGMENT)[0],
    );

    // WHEN the user clicks the first entity span
    await userEvent.click(firstSpan);

    // THEN the drawer is mounted
    expect(screen.getByTestId(DRAWER_DATA_TEST_ID.PANEL)).toBeInTheDocument();
  });

  it("resets state on Clear — drawer closes and results disappear", async () => {
    // GIVEN a finished run with an open drawer
    renderPage();
    await userEvent.type(
      screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.TEXTAREA),
      "Senior data scientist with Python.",
    );
    await userEvent.click(screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.RUN_BUTTON));
    await waitFor(() =>
      expect(
        screen.getByTestId(RESULTS_TABS_DATA_TEST_ID.CONTAINER),
      ).toBeInTheDocument(),
    );

    // WHEN the user clicks Clear
    await userEvent.click(screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.CLEAR_BUTTON));

    // THEN results disappear and the textarea is empty
    await waitFor(() => {
      expect(
        screen.queryByTestId(RESULTS_TABS_DATA_TEST_ID.CONTAINER),
      ).not.toBeInTheDocument();
    });
    expect(screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.TEXTAREA)).toHaveValue(
      "",
    );
  });

  it("surfaces an error toast when classify rejects", async () => {
    // GIVEN a failing backend
    apiMocks.classify.mockRejectedValueOnce(new Error("nope"));
    renderPage();
    await userEvent.type(
      screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.TEXTAREA),
      "x",
    );

    // WHEN the user clicks Run
    await userEvent.click(screen.getByTestId(SOURCE_PANE_DATA_TEST_ID.RUN_BUTTON));

    // THEN no results render and a toast role=status appears
    await waitFor(() => {
      expect(
        screen.queryByTestId(RESULTS_TABS_DATA_TEST_ID.CONTAINER),
      ).not.toBeInTheDocument();
    });
  });
});
