import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";

const apiMocks = vi.hoisted(() => {
  const store: Array<{
    pipeline_id: string;
    user_id: string;
    name: string;
    stages: unknown[];
    is_active: boolean;
    is_default: boolean;
    is_readonly: boolean;
    created_at: string;
    updated_at: string;
  }> = [];

  return {
    store,
    listPipelines: vi.fn(async () => ({ pipelines: [...store] })),
    activatePipeline: vi.fn(async (pipelineId: string) => {
      const found = store.find((row) => row.pipeline_id === pipelineId);
      if (!found) throw new Error("Not found");
      return { ...found, is_active: true };
    }),
    clonePipeline: vi.fn(async (pipelineId: string) => {
      const found = store.find((row) => row.pipeline_id === pipelineId);
      if (!found) throw new Error("Not found");
      const cloned = {
        ...found,
        pipeline_id: `${pipelineId}-clone`,
        name: `${found.name} (copy)`,
        is_active: false,
        is_readonly: false,
      };
      store.push(cloned);
      return cloned;
    }),
    deletePipeline: vi.fn(async (pipelineId: string) => {
      const index = store.findIndex((row) => row.pipeline_id === pipelineId);
      if (index >= 0) store.splice(index, 1);
    }),
    seed: (
      rows: Array<{
        pipeline_id: string;
        user_id: string;
        name: string;
        stages: unknown[];
        is_active: boolean;
        is_default: boolean;
        is_readonly: boolean;
        created_at: string;
        updated_at: string;
      }>,
    ) => {
      store.length = 0;
      store.push(...rows);
    },
    reset: () => {
      store.length = 0;
    },
  };
});

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    listPipelines: apiMocks.listPipelines,
    activatePipeline: apiMocks.activatePipeline,
    clonePipeline: apiMocks.clonePipeline,
    deletePipeline: apiMocks.deletePipeline,
  };
});

import { MemoryRouter } from "react-router-dom";
import { ToastProvider } from "@/components";
import { DATA_TEST_ID, PipelinesPage } from "./PipelinesPage";
import { DATA_TEST_ID as PIPELINES_TABLE_DATA_TEST_ID } from "../components/PipelinesTable/PipelinesTable";

const givenDefaultPipeline = {
  pipeline_id: "pipeline-default",
  user_id: "local-user",
  name: "Default Tabiya",
  stages: [],
  is_active: true,
  is_default: true,
  is_readonly: true,
  created_at: "2026-01-01T00:00:00.000Z",
  updated_at: "2026-06-15T00:00:00.000Z",
};

const givenEditablePipeline = {
  pipeline_id: "pipeline-editable",
  user_id: "local-user",
  name: "Recruiter tuning",
  stages: [],
  is_active: false,
  is_default: false,
  is_readonly: false,
  created_at: "2026-02-01T00:00:00.000Z",
  updated_at: "2026-07-01T00:00:00.000Z",
};

function renderPipelinesPage() {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <PipelinesPage />
      </ToastProvider>
    </MemoryRouter>,
  );
}

describe("PipelinesPage", () => {
  beforeEach(() => {
    apiMocks.reset();
    apiMocks.listPipelines.mockClear();
    apiMocks.activatePipeline.mockClear();
    apiMocks.clonePipeline.mockClear();
    apiMocks.deletePipeline.mockClear();
  });

  it("renders the page header and resolves the loading state", async () => {
    // GIVEN the expected title from i18n
    const expectedTitle = i18n.t("pipelines.list.title");

    // WHEN we render
    renderPipelinesPage();

    // THEN the title is present and loading clears once the list resolves
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(
      expectedTitle,
    );
    await waitFor(() =>
      expect(
        screen.queryByTestId(DATA_TEST_ID.LOADING),
      ).not.toBeInTheDocument(),
    );
  });

  it("shows the empty state when no pipelines are returned", async () => {
    // GIVEN an empty list response (store is empty)

    // WHEN we render
    renderPipelinesPage();

    // THEN the empty-state element appears once loading completes
    await waitFor(() =>
      expect(
        screen.getByTestId(DATA_TEST_ID.EMPTY_STATE),
      ).toBeInTheDocument(),
    );
  });

  it("renders the table with pipeline rows when pipelines exist", async () => {
    // GIVEN two pipelines in the store
    apiMocks.seed([givenDefaultPipeline, givenEditablePipeline]);
    const expectedRowCount = 2;

    // WHEN we render
    renderPipelinesPage();

    // THEN the table renders with two rows
    await waitFor(() => {
      const renderedRows = screen.getAllByTestId(
        PIPELINES_TABLE_DATA_TEST_ID.ROW,
      );
      expect(renderedRows).toHaveLength(expectedRowCount);
    });
  });

  it("shows the error state when the list fetch fails", async () => {
    // GIVEN the list fetch rejects
    apiMocks.listPipelines.mockRejectedValueOnce(new Error("Network error"));
    const expectedErrorTitle = i18n.t("pipelines.list.toasts.loadError");

    // WHEN we render
    renderPipelinesPage();

    // THEN the error container appears
    await waitFor(() =>
      expect(
        screen.getByTestId(DATA_TEST_ID.LOAD_ERROR),
      ).toBeInTheDocument(),
    );
    expect(screen.getByTestId(DATA_TEST_ID.LOAD_ERROR)).toHaveTextContent(
      expectedErrorTitle,
    );
  });

  it("calls activatePipeline when the toggle is clicked", async () => {
    // GIVEN one editable pipeline
    apiMocks.seed([givenEditablePipeline]);

    // WHEN we render and click the active toggle
    renderPipelinesPage();
    await waitFor(() =>
      expect(
        screen.getByTestId(PIPELINES_TABLE_DATA_TEST_ID.ROW),
      ).toBeInTheDocument(),
    );
    await userEvent.click(
      screen.getByTestId(PIPELINES_TABLE_DATA_TEST_ID.ACTIVE_TOGGLE),
    );

    // THEN activatePipeline was called with the correct id
    expect(apiMocks.activatePipeline).toHaveBeenCalledWith(
      givenEditablePipeline.pipeline_id,
    );
  });

  it("calls clonePipeline when the clone button is clicked", async () => {
    // GIVEN one editable pipeline
    apiMocks.seed([givenEditablePipeline]);

    // WHEN we render and click Clone
    renderPipelinesPage();
    await waitFor(() =>
      expect(
        screen.getByTestId(PIPELINES_TABLE_DATA_TEST_ID.ROW),
      ).toBeInTheDocument(),
    );
    await userEvent.click(
      screen.getByTestId(PIPELINES_TABLE_DATA_TEST_ID.CLONE_BUTTON),
    );

    // THEN clonePipeline was called with the correct id
    expect(apiMocks.clonePipeline).toHaveBeenCalledWith(
      givenEditablePipeline.pipeline_id,
    );
  });

  it("calls deletePipeline when the delete button is clicked", async () => {
    // GIVEN one editable pipeline
    apiMocks.seed([givenEditablePipeline]);

    // WHEN we render and click Delete
    renderPipelinesPage();
    await waitFor(() =>
      expect(
        screen.getByTestId(PIPELINES_TABLE_DATA_TEST_ID.ROW),
      ).toBeInTheDocument(),
    );
    await userEvent.click(
      screen.getByTestId(PIPELINES_TABLE_DATA_TEST_ID.DELETE_BUTTON),
    );

    // THEN deletePipeline was called with the correct id
    expect(apiMocks.deletePipeline).toHaveBeenCalledWith(
      givenEditablePipeline.pipeline_id,
    );
  });

  it("disables clone and delete buttons for readonly pipelines", async () => {
    // GIVEN only the readonly default pipeline
    apiMocks.seed([givenDefaultPipeline]);

    // WHEN we render
    renderPipelinesPage();
    await waitFor(() =>
      expect(
        screen.getByTestId(PIPELINES_TABLE_DATA_TEST_ID.ROW),
      ).toBeInTheDocument(),
    );

    // THEN clone and delete are disabled on the readonly row
    expect(
      screen.getByTestId(PIPELINES_TABLE_DATA_TEST_ID.CLONE_BUTTON),
    ).toBeDisabled();
    expect(
      screen.getByTestId(PIPELINES_TABLE_DATA_TEST_ID.DELETE_BUTTON),
    ).toBeDisabled();
  });
});
