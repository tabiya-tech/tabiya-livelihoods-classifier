import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import { MODAL_DATA_TEST_ID } from "@/components";
import { DATA_TEST_ID, UnsavedChangesGuard } from "./UnsavedChangesGuard";

describe("UnsavedChangesGuard", () => {
  it("does not render the dialog when open is false", () => {
    // GIVEN the guard is closed
    // WHEN we render it
    render(
      <UnsavedChangesGuard
        open={false}
        onConfirm={() => {}}
        onCancel={() => {}}
      />,
    );

    // THEN the dialog is not present
    expect(
      screen.queryByTestId(MODAL_DATA_TEST_ID.DIALOG),
    ).not.toBeInTheDocument();
  });

  it("renders the localized title, description, and action labels", () => {
    // GIVEN the guard is open
    const expectedTitle = i18n.t("configuration.unsavedGuard.title");
    const expectedDescription = i18n.t(
      "configuration.unsavedGuard.description",
    );
    const expectedConfirmLabel = i18n.t(
      "configuration.unsavedGuard.confirmLabel",
    );
    const expectedCancelLabel = i18n.t("common.buttons.keepEditing");

    // WHEN we render it
    render(
      <UnsavedChangesGuard
        open
        onConfirm={() => {}}
        onCancel={() => {}}
      />,
    );

    // THEN the localized copy and action buttons are visible
    expect(screen.getByTestId(MODAL_DATA_TEST_ID.TITLE)).toHaveTextContent(
      expectedTitle,
    );
    expect(
      screen.getByTestId(MODAL_DATA_TEST_ID.DESCRIPTION),
    ).toHaveTextContent(expectedDescription);
    expect(screen.getByTestId(DATA_TEST_ID.CONFIRM_BUTTON)).toHaveTextContent(
      expectedConfirmLabel,
    );
    expect(screen.getByTestId(DATA_TEST_ID.CANCEL_BUTTON)).toHaveTextContent(
      expectedCancelLabel,
    );
  });

  it("invokes onConfirm when the discard button is clicked", async () => {
    // GIVEN an onConfirm spy on an open guard
    const onConfirm = vi.fn();
    render(
      <UnsavedChangesGuard
        open
        onConfirm={onConfirm}
        onCancel={() => {}}
      />,
    );

    // WHEN the user clicks the discard-changes button
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONFIRM_BUTTON));

    // THEN onConfirm is invoked once and onCancel is not
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("invokes onCancel when the keep-editing button is clicked", async () => {
    // GIVEN an onCancel spy on an open guard
    const onCancel = vi.fn();
    render(
      <UnsavedChangesGuard
        open
        onConfirm={() => {}}
        onCancel={onCancel}
      />,
    );

    // WHEN the user clicks the keep-editing button
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CANCEL_BUTTON));

    // THEN onCancel is invoked once
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("invokes onCancel when the modal close button is clicked", async () => {
    // GIVEN an onCancel spy on an open guard
    const onCancel = vi.fn();
    render(
      <UnsavedChangesGuard
        open
        onConfirm={() => {}}
        onCancel={onCancel}
      />,
    );

    // WHEN the user clicks the modal's close button
    await userEvent.click(
      screen.getByTestId(MODAL_DATA_TEST_ID.CLOSE_BUTTON),
    );

    // THEN onCancel is invoked once (close maps to cancel for guards)
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
