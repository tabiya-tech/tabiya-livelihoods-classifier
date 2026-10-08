import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { fixtureNerManifest } from "@/mocks/fixtures/plugins";
import type { PipelineStage, PipelineValidationIssue } from "@/lib/api";
import type { PluginOptionsState } from "../../hooks/usePluginOptions";
import { PluginOptionsOverrideContext } from "../../hooks/pipelinesOverrides";
import { DATA_TEST_ID as DRAWER_DATA_TEST_ID } from "@/components/Drawer/Drawer";
import { DATA_TEST_ID, StageDetailDrawer } from "./StageDetailDrawer";

// ── Fixtures ─────────────────────────────────────────────────────────────────

const givenNerStage: PipelineStage = {
  plugin_id: fixtureNerManifest.plugin_id,
  config: { model_id: "tabiya/roberta-base-job-ner", entity_types: ["occupation"] },
};

/** Wrap with a ready options context so x-source fields don't trigger real fetches. */
function renderWithReadyOptions(children: React.ReactNode) {
  const givenReadyOptions: PluginOptionsState = {
    status: "ready",
    options: [],
    error: null,
  };
  return render(
    <PluginOptionsOverrideContext.Provider value={givenReadyOptions}>
      {children}
    </PluginOptionsOverrideContext.Provider>,
  );
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("StageDetailDrawer", () => {
  it("does not render the drawer backdrop when open is false", () => {
    // GIVEN the drawer is closed
    const givenOpen = false;

    // WHEN the component is rendered
    render(
      <StageDetailDrawer
        open={givenOpen}
        onClose={vi.fn()}
        onChange={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    // THEN the backdrop is not in the document
    expect(
      screen.queryByTestId(DRAWER_DATA_TEST_ID.BACKDROP),
    ).not.toBeInTheDocument();
  });

  it("renders the plugin name in the header and slot pills when open with a stage", () => {
    // GIVEN the drawer is open with the NER stage and manifest
    const givenOpen = true;
    const givenStageIndex = 1;
    const expectedPluginName = fixtureNerManifest.name;
    const expectedInputSlot = fixtureNerManifest.input_slot.type;
    const expectedOutputSlot = fixtureNerManifest.output_slot.type;

    // WHEN the component is rendered
    renderWithReadyOptions(
      <StageDetailDrawer
        open={givenOpen}
        onClose={vi.fn()}
        stage={givenNerStage}
        stageIndex={givenStageIndex}
        manifest={fixtureNerManifest}
        onChange={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    // THEN the drawer header shows the plugin name
    expect(screen.getByTestId(DRAWER_DATA_TEST_ID.TITLE)).toHaveTextContent(
      expectedPluginName,
    );

    // AND the slot pills are present with the correct slot types
    const slotPillsContainer = screen.getByTestId(DATA_TEST_ID.SLOT_PILLS);
    expect(slotPillsContainer).toBeInTheDocument();
    expect(slotPillsContainer).toHaveTextContent(`In: ${expectedInputSlot}`);
    expect(slotPillsContainer).toHaveTextContent(`Out: ${expectedOutputSlot}`);
  });

  it("shows the errors block when errors prop has items", () => {
    // GIVEN one validation issue for this stage
    const givenErrors: PipelineValidationIssue[] = [
      { code: "MISSING_REQUIRED", message: "model_id is required" },
    ];
    const expectedErrorMessage = givenErrors[0].message;

    // WHEN the component is rendered with errors
    renderWithReadyOptions(
      <StageDetailDrawer
        open={true}
        onClose={vi.fn()}
        stage={givenNerStage}
        manifest={fixtureNerManifest}
        errors={givenErrors}
        onChange={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    // THEN the errors block is visible with the issue message
    const errorsBlock = screen.getByTestId(DATA_TEST_ID.ERRORS_BLOCK);
    expect(errorsBlock).toBeInTheDocument();
    expect(errorsBlock).toHaveTextContent(expectedErrorMessage);
  });

  it("fires onDelete when the delete button is clicked", async () => {
    // GIVEN a delete spy
    const givenOnDelete = vi.fn();

    // WHEN the component is rendered and the delete button is clicked
    renderWithReadyOptions(
      <StageDetailDrawer
        open={true}
        onClose={vi.fn()}
        stage={givenNerStage}
        manifest={fixtureNerManifest}
        onChange={vi.fn()}
        onDelete={givenOnDelete}
      />,
    );
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.DELETE_BUTTON));

    // THEN the onDelete callback is called once
    expect(givenOnDelete).toHaveBeenCalledTimes(1);
  });

  it("fires onClose when the close button is clicked", async () => {
    // GIVEN a close spy
    const givenOnClose = vi.fn();

    // WHEN the component is rendered and the close button is clicked
    renderWithReadyOptions(
      <StageDetailDrawer
        open={true}
        onClose={givenOnClose}
        stage={givenNerStage}
        manifest={fixtureNerManifest}
        onChange={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CLOSE_BUTTON));

    // THEN the onClose callback is called once
    expect(givenOnClose).toHaveBeenCalledTimes(1);
  });
});
