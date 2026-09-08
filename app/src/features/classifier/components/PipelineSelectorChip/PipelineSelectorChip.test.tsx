import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import type { Pipeline } from "@/lib/api";
import { routerPaths } from "@/routes/routerPaths";
import {
  DATA_TEST_ID,
  PipelineSelectorChip,
  type PipelineSelectorChipProps,
} from "./PipelineSelectorChip";

const givenActivePipeline: Pipeline = {
  pipeline_id: "pipeline-default",
  user_id: "local-user",
  name: "Default Tabiya",
  stages: [],
  is_active: true,
  is_default: true,
  is_readonly: true,
  created_at: "2026-01-01T00:00:00.000Z",
  updated_at: "2026-01-01T00:00:00.000Z",
};

const givenSecondPipeline: Pipeline = {
  pipeline_id: "pipeline-recruiter",
  user_id: "local-user",
  name: "Recruiter tuning",
  stages: [],
  is_active: false,
  is_default: false,
  is_readonly: false,
  created_at: "2026-02-01T00:00:00.000Z",
  updated_at: "2026-02-01T00:00:00.000Z",
};

function renderChip(overrides: Partial<PipelineSelectorChipProps> = {}) {
  const defaultProps: PipelineSelectorChipProps = {
    pipelines: [givenActivePipeline, givenSecondPipeline],
    selectedPipelineId: givenActivePipeline.pipeline_id,
    onPipelineChange: vi.fn(),
    isLoading: false,
  };
  const finalProps: PipelineSelectorChipProps = { ...defaultProps, ...overrides };
  const utils = render(
    <MemoryRouter>
      <PipelineSelectorChip {...finalProps} />
    </MemoryRouter>,
  );
  return { ...utils, props: finalProps };
}

describe("PipelineSelectorChip", () => {
  it("renders the loading label and disables the trigger while pipelines load", () => {
    // GIVEN a chip in loading state
    const expectedLoadingLabel = "Loading pipelines…";
    renderChip({
      pipelines: [],
      selectedPipelineId: null,
      isLoading: true,
    });

    // WHEN we inspect the trigger
    const trigger = screen.getByTestId(DATA_TEST_ID.TRIGGER);

    // THEN it shows the loading label and is disabled
    expect(trigger).toHaveTextContent(expectedLoadingLabel);
    expect(trigger).toBeDisabled();
  });

  it("renders the selected pipeline's name inside the trigger", () => {
    // GIVEN pipelines with one selected
    const expectedName = givenActivePipeline.name;
    renderChip({ selectedPipelineId: givenActivePipeline.pipeline_id });

    // WHEN we look at the trigger
    const trigger = screen.getByTestId(DATA_TEST_ID.TRIGGER);

    // THEN it shows that pipeline's name
    expect(trigger).toHaveTextContent(expectedName);
  });

  it("opens the popover with one row per pipeline when the trigger is clicked", async () => {
    // GIVEN a closed chip with two pipelines
    const expectedRowCount = 2;
    renderChip();

    // WHEN the user clicks the trigger
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TRIGGER));

    // THEN the popover opens with one row per pipeline
    expect(screen.getByTestId(DATA_TEST_ID.POPOVER)).toBeInTheDocument();
    const rows = screen.getAllByTestId(DATA_TEST_ID.PIPELINE_ROW);
    expect(rows).toHaveLength(expectedRowCount);
  });

  it("calls onPipelineChange with the row's pipeline id when the user clicks it", async () => {
    // GIVEN a chip with a click handler
    const givenChangeHandler = vi.fn();
    const expectedPipelineId = givenSecondPipeline.pipeline_id;
    renderChip({ onPipelineChange: givenChangeHandler });

    // WHEN the user opens the popover and clicks the second pipeline
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TRIGGER));
    const rows = screen.getAllByTestId(DATA_TEST_ID.PIPELINE_ROW);
    const secondRow = rows.find(
      (element) => element.getAttribute("data-pipeline-id") === expectedPipelineId,
    );
    if (!secondRow) throw new Error("second pipeline row not found");
    await userEvent.click(secondRow);

    // THEN the handler was called with that pipeline's id
    expect(givenChangeHandler).toHaveBeenCalledWith(expectedPipelineId);
  });

  it("routes the Edit link to routerPaths.PIPELINES", async () => {
    // GIVEN an open popover
    const expectedHref = routerPaths.PIPELINES;
    renderChip({ defaultOpen: true });

    // WHEN we inspect the edit link
    const editLink = screen.getByTestId(DATA_TEST_ID.EDIT_LINK);

    // THEN it points to /pipelines
    expect(editLink).toHaveAttribute("href", expectedHref);
  });

  it("closes the popover when the user clicks outside", async () => {
    // GIVEN an open chip alongside an unrelated sibling
    const outsideTestId = "outside-node-for-click";
    render(
      <MemoryRouter>
        <div>
          <PipelineSelectorChip
            pipelines={[givenActivePipeline, givenSecondPipeline]}
            selectedPipelineId={givenActivePipeline.pipeline_id}
            onPipelineChange={vi.fn()}
            isLoading={false}
          />
          <button type="button" data-testid={outsideTestId}>outside</button>
        </div>
      </MemoryRouter>,
    );
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TRIGGER));
    expect(screen.getByTestId(DATA_TEST_ID.POPOVER)).toBeInTheDocument();

    // WHEN the user clicks outside the chip
    await userEvent.click(screen.getByTestId(outsideTestId));

    // THEN the popover closes
    expect(screen.queryByTestId(DATA_TEST_ID.POPOVER)).not.toBeInTheDocument();
  });
});
