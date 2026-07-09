import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import type { PipelineValidationIssue } from "@/lib/api";
import { DATA_TEST_ID, PipelineSaveBar } from "./PipelineSaveBar";
import type { PipelineSaveBarProps } from "./PipelineSaveBar";

// ── default props ─────────────────────────────────────────────────────────────

const defaultProps: PipelineSaveBarProps = {
  name: "My Pipeline",
  onNameChange: vi.fn(),
  isReadonly: false,
  issues: [],
  isValidating: false,
  isDirty: true,
  isSaving: false,
  onSave: vi.fn(),
  onCancel: vi.fn(),
};

function renderSaveBar(overrides: Partial<PipelineSaveBarProps> = {}) {
  return render(<PipelineSaveBar {...defaultProps} {...overrides} />);
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe("PipelineSaveBar", () => {
  it("shows the name input when isReadonly is false and hides it when true", () => {
    // GIVEN isReadonly is false (name input should appear)
    const givenIsReadonly = false;

    // WHEN we render with isReadonly=false
    const { rerender } = renderSaveBar({ isReadonly: givenIsReadonly });

    // THEN the name input is present
    expect(screen.getByTestId(DATA_TEST_ID.NAME_INPUT)).toBeInTheDocument();

    // AND WHEN we rerender with isReadonly=true
    rerender(<PipelineSaveBar {...defaultProps} isReadonly={true} />);

    // THEN the name input is no longer rendered
    expect(screen.queryByTestId(DATA_TEST_ID.NAME_INPUT)).not.toBeInTheDocument();
  });

  it("validation pill shows the validating text when isValidating is true", () => {
    // GIVEN isValidating=true and the expected text from i18n
    const givenIsValidating = true;
    const expectedPillText = i18n.t("pipelines.editor.saveBar.validating");

    // WHEN we render
    renderSaveBar({ isValidating: givenIsValidating });

    // THEN the pill renders with the validating key text (falls back to key if not in catalogue)
    const pill = screen.getByTestId(DATA_TEST_ID.VALIDATION_PILL);
    expect(pill).toBeInTheDocument();
    expect(pill).toHaveTextContent(expectedPillText);
  });

  it("validation pill shows the issues count when issues are non-empty", () => {
    // GIVEN one validation issue
    const givenIssues: PipelineValidationIssue[] = [
      { code: "slot_mismatch", message: "Output slot does not match input slot" },
    ];
    const expectedPillText = i18n.t("pipelines.editor.saveBar.issuesCount", {
      count: givenIssues.length,
    });

    // WHEN we render with issues
    renderSaveBar({ isValidating: false, issues: givenIssues });

    // THEN the pill contains the issues-count text
    const pill = screen.getByTestId(DATA_TEST_ID.VALIDATION_PILL);
    expect(pill).toHaveTextContent(expectedPillText);
  });

  it("save button is disabled when issues are present", () => {
    // GIVEN one validation issue
    const givenIssues: PipelineValidationIssue[] = [
      { code: "missing_plugin", message: "Plugin not found" },
    ];

    // WHEN we render with that issue
    renderSaveBar({ issues: givenIssues, isDirty: true });

    // THEN the save button is disabled
    expect(screen.getByTestId(DATA_TEST_ID.SAVE_BUTTON)).toBeDisabled();
  });

  it("save button is disabled when isDirty is false", () => {
    // GIVEN the pipeline has no unsaved changes
    const givenIsDirty = false;

    // WHEN we render with isDirty=false
    renderSaveBar({ isDirty: givenIsDirty, issues: [] });

    // THEN the save button is disabled
    expect(screen.getByTestId(DATA_TEST_ID.SAVE_BUTTON)).toBeDisabled();
  });

  it("calls onSave when the save button is clicked and saving is enabled", async () => {
    // GIVEN a save handler and a clean, dirty, non-saving state
    const givenOnSave = vi.fn();
    const user = userEvent.setup();

    // WHEN we render with valid conditions for saving and click Save
    renderSaveBar({
      isDirty: true,
      isReadonly: false,
      issues: [],
      isValidating: false,
      isSaving: false,
      onSave: givenOnSave,
    });
    await user.click(screen.getByTestId(DATA_TEST_ID.SAVE_BUTTON));

    // THEN onSave was called exactly once
    expect(givenOnSave).toHaveBeenCalledTimes(1);
  });

  it("calls onCancel when the cancel button is clicked", async () => {
    // GIVEN a cancel handler
    const givenOnCancel = vi.fn();
    const user = userEvent.setup();

    // WHEN we render and click Cancel
    renderSaveBar({ onCancel: givenOnCancel });
    await user.click(screen.getByTestId(DATA_TEST_ID.CANCEL_BUTTON));

    // THEN onCancel was called exactly once
    expect(givenOnCancel).toHaveBeenCalledTimes(1);
  });
});
