import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { ReactFlowProvider } from "reactflow";
import i18n from "@/i18n/i18n";
import { ToastProvider } from "@/components";
import { NavigationGuardProvider } from "@/lib/navigationGuard";
import type { ListPluginsResponse, Pipeline, PluginDetail } from "@/lib/api";
import {
  fixtureRecruiterTuningPipeline,
  fixtureDefaultTabiyaPipeline,
} from "@/mocks/fixtures/pipelines";
import {
  fixturePluginSummaries,
  fixturePluginDetails,
} from "@/mocks/fixtures/plugins";
import { DATA_TEST_ID, PipelineEditorPage } from "./PipelineEditorPage";
import { DATA_TEST_ID as SAVE_BAR_DATA_TEST_ID } from "../components/PipelineSaveBar/PipelineSaveBar";

// Mock the API module so we can control what is returned without hitting MSW.
const apiMocks = vi.hoisted(() => {
  return {
    listPlugins: vi.fn<() => Promise<ListPluginsResponse>>(),
    getPlugin: vi.fn<(pluginId: string) => Promise<PluginDetail>>(),
    getPipeline: vi.fn<(pipelineId: string) => Promise<Pipeline>>(),
    createPipeline: vi.fn(),
    updatePipeline: vi.fn(),
    validatePipeline: vi.fn(async () => ({ valid: true, issues: [] })),
  };
});

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    listPlugins: apiMocks.listPlugins,
    getPlugin: apiMocks.getPlugin,
    getPipeline: apiMocks.getPipeline,
    createPipeline: apiMocks.createPipeline,
    updatePipeline: apiMocks.updatePipeline,
    validatePipeline: apiMocks.validatePipeline,
  };
});

// Mock useNavigate so we can assert navigate calls without triggering the
// data-router AbortSignal/undici conflict that occurs when the router actually
// navigates in jsdom+MSW. The unsaved-changes guard uses the app's
// NavigationGuardProvider (below), not react-router's useBlocker.
const navigateMock = vi.fn();
vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    useNavigate: () => navigateMock,
  };
});

/**
 * Renders the PipelineEditorPage inside a MemoryRouter + NavigationGuardProvider
 * (the editor's unsaved-changes guard reads from that provider). useNavigate is
 * mocked at the module level so navigate calls can be asserted via navigateMock.
 */
function renderEditorAtPath(initialPath: string) {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <NavigationGuardProvider>
        <ToastProvider>
          <ReactFlowProvider>
            <Routes>
              <Route path="/pipelines/new" element={<PipelineEditorPage />} />
              <Route path="/pipelines/:pipelineId" element={<PipelineEditorPage />} />
              <Route path="/pipelines" element={<div data-testid="pipelines-page">Pipelines</div>} />
            </Routes>
          </ReactFlowProvider>
        </ToastProvider>
      </NavigationGuardProvider>
    </MemoryRouter>,
  );
}

describe("PipelineEditorPage", () => {
  beforeEach(() => {
    // Default: plugins always resolve successfully.
    apiMocks.listPlugins.mockResolvedValue({
      plugins: fixturePluginSummaries,
    });
    apiMocks.getPlugin.mockImplementation(async (pluginId: string) => {
      const detail = fixturePluginDetails[pluginId];
      if (!detail) throw new Error(`Unknown plugin: ${pluginId}`);
      return detail;
    });
    // Default: pipeline fetches reject (only seeded per test)
    apiMocks.getPipeline.mockRejectedValue(new Error("Not found"));
    apiMocks.createPipeline.mockResolvedValue({
      ...fixtureRecruiterTuningPipeline,
      pipeline_id: "pipeline-new-001",
    });
    apiMocks.updatePipeline.mockResolvedValue(fixtureRecruiterTuningPipeline);
  });

  it("shows the editor with the empty canvas prompt for a new pipeline", async () => {
    // GIVEN we navigate to the new pipeline route
    const givenPath = "/pipelines/new";
    const expectedEmptyPrompt = i18n.t("pipelines.editor.emptyPrompt");
    const expectedTitle = i18n.t("pipelines.editor.title.new");

    // WHEN we render the editor
    renderEditorAtPath(givenPath);

    // THEN the empty prompt appears once loading clears and initialization completes
    await waitFor(() =>
      expect(screen.getByTestId(DATA_TEST_ID.EMPTY_PROMPT)).toBeInTheDocument(),
    );
    expect(screen.getByText(expectedTitle)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.EMPTY_PROMPT)).toHaveTextContent(
      expectedEmptyPrompt,
    );
  });

  it("loads an existing pipeline and shows the pipeline name in the save bar", async () => {
    // GIVEN an existing editable pipeline returned by the API
    const givenPipeline = fixtureRecruiterTuningPipeline;
    apiMocks.getPipeline.mockResolvedValue(givenPipeline);
    const givenPath = `/pipelines/${givenPipeline.pipeline_id}`;
    const expectedName = givenPipeline.name;

    // WHEN we render the editor for the existing pipeline
    renderEditorAtPath(givenPath);

    // THEN loading clears and the pipeline name appears in the name input
    await waitFor(() =>
      expect(screen.queryByTestId(DATA_TEST_ID.LOADING)).not.toBeInTheDocument(),
    );
    const nameInput = await screen.findByTestId(SAVE_BAR_DATA_TEST_ID.NAME_INPUT);
    expect(nameInput).toHaveValue(expectedName);
  });

  it("disables the save button for a readonly pipeline", async () => {
    // GIVEN a readonly pipeline returned by the API
    const givenPipeline = fixtureDefaultTabiyaPipeline;
    apiMocks.getPipeline.mockResolvedValue(givenPipeline);
    const givenPath = `/pipelines/${givenPipeline.pipeline_id}`;

    // WHEN we render the editor for the readonly pipeline
    renderEditorAtPath(givenPath);

    // THEN loading clears and the save button is disabled (readonly pipeline)
    await waitFor(() =>
      expect(screen.queryByTestId(DATA_TEST_ID.LOADING)).not.toBeInTheDocument(),
    );
    const saveButton = screen.getByTestId(SAVE_BAR_DATA_TEST_ID.SAVE_BUTTON);
    expect(saveButton).toBeDisabled();
  });

  it("reflects a changed name in the save bar input (dirty state)", async () => {
    // GIVEN an existing editable pipeline returned by the API
    const givenPipeline = fixtureRecruiterTuningPipeline;
    apiMocks.getPipeline.mockResolvedValue(givenPipeline);
    const givenPath = `/pipelines/${givenPipeline.pipeline_id}`;
    const givenNewNameSuffix = " updated";

    // AND the editor has loaded
    renderEditorAtPath(givenPath);
    await waitFor(() =>
      expect(screen.queryByTestId(DATA_TEST_ID.LOADING)).not.toBeInTheDocument(),
    );

    // WHEN the user appends text to the name input
    const nameInput = screen.getByTestId(SAVE_BAR_DATA_TEST_ID.NAME_INPUT);
    await userEvent.type(nameInput, givenNewNameSuffix);

    // THEN the input reflects the updated value
    expect(nameInput).toHaveValue(`${givenPipeline.name}${givenNewNameSuffix}`);
  });

  it("renders all structural containers on the new pipeline route", async () => {
    // GIVEN we are at the new pipeline path
    const givenPath = "/pipelines/new";

    // WHEN we render the editor
    renderEditorAtPath(givenPath);

    // AND loading completes
    await waitFor(() =>
      expect(screen.queryByTestId(DATA_TEST_ID.LOADING)).not.toBeInTheDocument(),
    );

    // THEN the container, palette, canvas, and save bar are all present
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.PALETTE_CONTAINER)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.CANVAS_CONTAINER)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.SAVE_BAR_CONTAINER)).toBeInTheDocument();
  });

  it("calls navigate to the pipelines list when the Cancel button is clicked", async () => {
    // GIVEN the new pipeline route
    const givenPath = "/pipelines/new";
    const expectedNavigationTarget = "/pipelines";
    navigateMock.mockClear();

    // WHEN we render and wait for the editor to load
    renderEditorAtPath(givenPath);
    await waitFor(() =>
      expect(screen.queryByTestId(DATA_TEST_ID.LOADING)).not.toBeInTheDocument(),
    );

    // AND we click Cancel
    const cancelButton = screen.getByTestId(SAVE_BAR_DATA_TEST_ID.CANCEL_BUTTON);
    await userEvent.click(cancelButton);

    // THEN navigate is called with the pipelines list route
    expect(navigateMock).toHaveBeenCalledWith(expectedNavigationTarget);
  });

  it("shows the error state when the pipeline cannot be loaded", async () => {
    // GIVEN the API rejects with an error
    const givenError = new Error("Not found");
    apiMocks.getPipeline.mockRejectedValue(givenError);
    const givenPath = "/pipelines/pipeline-does-not-exist";
    const expectedErrorText = i18n.t("pipelines.editor.toasts.loadError");

    // WHEN we render the editor for a non-existent pipeline
    renderEditorAtPath(givenPath);

    // THEN the error state is shown with the correct message
    await waitFor(() =>
      expect(screen.getByTestId(DATA_TEST_ID.ERROR)).toBeInTheDocument(),
    );
    expect(screen.getByTestId(DATA_TEST_ID.ERROR)).toHaveTextContent(
      expectedErrorText,
    );
  });
});
