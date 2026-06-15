import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import { SaveBar, DATA_TEST_ID } from "./SaveBar";

describe("SaveBar", () => {
  it("does not render when the draft is clean and save is idle", () => {
    // GIVEN no unsaved changes and an idle save state
    // WHEN we render the bar
    render(
      <SaveBar
        isDirty={false}
        saveStatus="idle"
        onSave={() => {}}
        onDiscard={() => {}}
      />,
    );

    // THEN the bar is absent from the DOM
    expect(screen.queryByTestId(DATA_TEST_ID.CONTAINER)).not.toBeInTheDocument();
  });

  it("renders the dirty-state copy when the draft has unsaved changes", () => {
    // GIVEN unsaved changes
    const expectedDirtyTitle = i18n.t("configuration.saveBar.dirtyTitle");
    const expectedDirtyHelp = i18n.t("configuration.saveBar.dirtyHelp");

    // WHEN we render the bar
    render(
      <SaveBar
        isDirty
        saveStatus="idle"
        onSave={() => {}}
        onDiscard={() => {}}
      />,
    );

    // THEN the bar surfaces the unsaved-changes copy
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(
      expectedDirtyTitle,
    );
    expect(screen.getByTestId(DATA_TEST_ID.HELP)).toHaveTextContent(
      expectedDirtyHelp,
    );
  });

  it("renders the saving copy and disables Discard while saving", () => {
    // GIVEN a save in flight
    const expectedSavingTitle = i18n.t("configuration.saveBar.savingTitle");

    // WHEN we render the bar
    render(
      <SaveBar
        isDirty
        saveStatus="saving"
        onSave={() => {}}
        onDiscard={() => {}}
      />,
    );

    // THEN the saving title appears and the Discard button is disabled
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(
      expectedSavingTitle,
    );
    expect(screen.getByTestId(DATA_TEST_ID.DISCARD_BUTTON)).toBeDisabled();
  });

  it("renders the saved indicator and copy when saveStatus='saved'", () => {
    // GIVEN a recently-completed save
    const expectedSavedTitle = i18n.t("configuration.saveBar.savedTitle");

    // WHEN we render the bar
    render(
      <SaveBar
        isDirty={false}
        saveStatus="saved"
        onSave={() => {}}
        onDiscard={() => {}}
      />,
    );

    // THEN the saved title and the saved indicator are visible
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(
      expectedSavedTitle,
    );
    expect(screen.getByTestId(DATA_TEST_ID.SAVED_INDICATOR)).toBeInTheDocument();
  });

  it("invokes onSave when the Save button is clicked", async () => {
    // GIVEN an onSave spy
    const onSave = vi.fn();
    render(
      <SaveBar
        isDirty
        saveStatus="idle"
        onSave={onSave}
        onDiscard={() => {}}
      />,
    );

    // WHEN the user clicks the Save button
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.SAVE_BUTTON));

    // THEN onSave is invoked exactly once
    expect(onSave).toHaveBeenCalledTimes(1);
  });

  it("invokes onDiscard when the Discard button is clicked", async () => {
    // GIVEN an onDiscard spy
    const onDiscard = vi.fn();
    render(
      <SaveBar
        isDirty
        saveStatus="idle"
        onSave={() => {}}
        onDiscard={onDiscard}
      />,
    );

    // WHEN the user clicks the Discard button
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.DISCARD_BUTTON));

    // THEN onDiscard is invoked exactly once
    expect(onDiscard).toHaveBeenCalledTimes(1);
  });
});
