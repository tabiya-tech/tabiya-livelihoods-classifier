import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import type { Pipeline } from "@/lib/api";
import { DATA_TEST_ID, PipelinesTable } from "./PipelinesTable";

const givenActivePipeline: Pipeline = {
  pipeline_id: "pipeline-001",
  user_id: "u",
  name: "Default Tabiya",
  stages: [],
  is_active: true,
  is_default: true,
  is_readonly: true,
  created_at: "2026-01-01T00:00:00.000Z",
  updated_at: "2026-06-15T00:00:00.000Z",
};

const givenEditablePipeline: Pipeline = {
  pipeline_id: "pipeline-002",
  user_id: "u",
  name: "Recruiter tuning",
  stages: [],
  is_active: false,
  is_default: false,
  is_readonly: false,
  created_at: "2026-02-01T00:00:00.000Z",
  updated_at: "2026-07-01T00:00:00.000Z",
};

describe("PipelinesTable", () => {
  it("renders one row per pipeline with name and formatted date", () => {
    // GIVEN two pipelines
    const givenPipelines = [givenActivePipeline, givenEditablePipeline];

    // WHEN we render
    render(
      <PipelinesTable
        pipelines={givenPipelines}
        pendingId={null}
        onActivate={() => {}}
        onClone={() => {}}
        onDelete={() => {}}
      />,
    );

    // THEN the table has two rows, one per pipeline
    const renderedRows = screen.getAllByTestId(DATA_TEST_ID.ROW);
    expect(renderedRows).toHaveLength(2);
    expect(renderedRows[0]).toHaveAttribute(
      "data-pipeline-id",
      givenActivePipeline.pipeline_id,
    );
    expect(
      within(renderedRows[0]).getByTestId(DATA_TEST_ID.NAME_CELL),
    ).toHaveTextContent(givenActivePipeline.name);
  });

  it("shows the default badge when is_default is true", () => {
    // GIVEN the expected default badge text
    const expectedDefaultBadge = i18n.t("pipelines.list.badges.default");

    // WHEN we render a default pipeline
    render(
      <PipelinesTable
        pipelines={[givenActivePipeline]}
        pendingId={null}
        onActivate={() => {}}
        onClone={() => {}}
        onDelete={() => {}}
      />,
    );

    // THEN the default badge is present
    expect(screen.getByTestId(DATA_TEST_ID.DEFAULT_BADGE)).toHaveTextContent(
      expectedDefaultBadge,
    );
  });

  it("formats the updated_at ISO timestamp as a localized date", () => {
    // GIVEN a pipeline with a known updated_at
    const givenPipeline: Pipeline = {
      ...givenEditablePipeline,
      updated_at: "2026-07-01T00:00:00.000Z",
    };

    // WHEN we render
    render(
      <PipelinesTable
        pipelines={[givenPipeline]}
        pendingId={null}
        onActivate={() => {}}
        onClone={() => {}}
        onDelete={() => {}}
      />,
    );

    // THEN the updated cell shows a non-empty localized date string
    const updatedCell = screen.getByTestId(DATA_TEST_ID.UPDATED_CELL);
    expect(updatedCell.textContent?.trim().length).toBeGreaterThan(0);
  });

  it("calls onActivate with the pipeline_id when the toggle is clicked", async () => {
    // GIVEN an activate spy and one editable pipeline
    const onActivate = vi.fn();

    // WHEN we render and click the toggle
    render(
      <PipelinesTable
        pipelines={[givenEditablePipeline]}
        pendingId={null}
        onActivate={onActivate}
        onClone={() => {}}
        onDelete={() => {}}
      />,
    );
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.ACTIVE_TOGGLE));

    // THEN onActivate fires with the correct pipeline_id
    expect(onActivate).toHaveBeenCalledWith(givenEditablePipeline.pipeline_id);
  });

  it("calls onClone with the pipeline_id when the clone button is clicked", async () => {
    // GIVEN a clone spy and one editable pipeline
    const onClone = vi.fn();

    // WHEN we render and click Clone
    render(
      <PipelinesTable
        pipelines={[givenEditablePipeline]}
        pendingId={null}
        onActivate={() => {}}
        onClone={onClone}
        onDelete={() => {}}
      />,
    );
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CLONE_BUTTON));

    // THEN onClone fires with the correct pipeline_id
    expect(onClone).toHaveBeenCalledWith(givenEditablePipeline.pipeline_id);
  });

  it("calls onDelete with the pipeline_id when the delete button is clicked", async () => {
    // GIVEN a delete spy and one editable pipeline
    const onDelete = vi.fn();

    // WHEN we render and click Delete
    render(
      <PipelinesTable
        pipelines={[givenEditablePipeline]}
        pendingId={null}
        onActivate={() => {}}
        onClone={() => {}}
        onDelete={onDelete}
      />,
    );
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.DELETE_BUTTON));

    // THEN onDelete fires with the correct pipeline_id
    expect(onDelete).toHaveBeenCalledWith(givenEditablePipeline.pipeline_id);
  });

  it("disables clone and delete buttons for readonly pipelines", () => {
    // GIVEN a readonly pipeline
    const givenPipelines = [givenActivePipeline, givenEditablePipeline];

    // WHEN we render both pipelines
    render(
      <PipelinesTable
        pipelines={givenPipelines}
        pendingId={null}
        onActivate={() => {}}
        onClone={() => {}}
        onDelete={() => {}}
      />,
    );

    // THEN the readonly row's clone and delete buttons are disabled
    const rows = screen.getAllByTestId(DATA_TEST_ID.ROW);
    const readonlyRow = rows[0];
    const editableRow = rows[1];
    expect(
      within(readonlyRow).getByTestId(DATA_TEST_ID.CLONE_BUTTON),
    ).toBeDisabled();
    expect(
      within(readonlyRow).getByTestId(DATA_TEST_ID.DELETE_BUTTON),
    ).toBeDisabled();

    // AND the editable row's buttons are enabled
    expect(
      within(editableRow).getByTestId(DATA_TEST_ID.CLONE_BUTTON),
    ).toBeEnabled();
    expect(
      within(editableRow).getByTestId(DATA_TEST_ID.DELETE_BUTTON),
    ).toBeEnabled();
  });

  it("disables all controls on the pending row while a request is in-flight", () => {
    // GIVEN a pendingId matching the editable pipeline
    const givenPipelines = [givenActivePipeline, givenEditablePipeline];
    const givenPendingId = givenEditablePipeline.pipeline_id;

    // WHEN we render with pendingId set
    render(
      <PipelinesTable
        pipelines={givenPipelines}
        pendingId={givenPendingId}
        onActivate={() => {}}
        onClone={() => {}}
        onDelete={() => {}}
      />,
    );

    // THEN the pending row's toggle and action buttons are all disabled
    const rows = screen.getAllByTestId(DATA_TEST_ID.ROW);
    const pendingRow = rows[1];
    expect(
      within(pendingRow).getByTestId(DATA_TEST_ID.ACTIVE_TOGGLE),
    ).toBeDisabled();
    expect(
      within(pendingRow).getByTestId(DATA_TEST_ID.CLONE_BUTTON),
    ).toBeDisabled();
    expect(
      within(pendingRow).getByTestId(DATA_TEST_ID.DELETE_BUTTON),
    ).toBeDisabled();
  });
});
