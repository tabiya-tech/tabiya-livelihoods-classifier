import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import { MODAL_DATA_TEST_ID } from "@/components";
import { DATA_TEST_ID, RevokeConfirmModal } from "./RevokeConfirmModal";

describe("RevokeConfirmModal", () => {
  it("does not render when open is false", () => {
    // GIVEN the modal closed
    // WHEN we render it
    render(
      <RevokeConfirmModal
        open={false}
        keyLabel="anything"
        onConfirm={() => {}}
        onCancel={() => {}}
      />,
    );

    // THEN the dialog is absent
    expect(
      screen.queryByTestId(MODAL_DATA_TEST_ID.DIALOG),
    ).not.toBeInTheDocument();
  });

  it("interpolates the key label into the description", () => {
    // GIVEN a label
    const givenKeyLabel = "analyst-laptop";
    const expectedDescription = i18n.t("keys.revokeModal.description", {
      label: givenKeyLabel,
    });

    // WHEN we render open
    render(
      <RevokeConfirmModal
        open
        keyLabel={givenKeyLabel}
        onConfirm={() => {}}
        onCancel={() => {}}
      />,
    );

    // THEN the description shows the interpolated label
    expect(
      screen.getByTestId(MODAL_DATA_TEST_ID.DESCRIPTION),
    ).toHaveTextContent(expectedDescription);
  });

  it("invokes onConfirm when the danger button is clicked", async () => {
    // GIVEN an onConfirm spy
    const onConfirm = vi.fn();
    render(
      <RevokeConfirmModal
        open
        keyLabel="x"
        onConfirm={onConfirm}
        onCancel={() => {}}
      />,
    );

    // WHEN the user clicks confirm
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONFIRM_BUTTON));

    // THEN onConfirm fires once
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("invokes onCancel when the cancel button is clicked", async () => {
    // GIVEN an onCancel spy
    const onCancel = vi.fn();
    render(
      <RevokeConfirmModal
        open
        keyLabel="x"
        onConfirm={() => {}}
        onCancel={onCancel}
      />,
    );

    // WHEN the user clicks cancel
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CANCEL_BUTTON));

    // THEN onCancel fires once
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("disables both buttons while isSubmitting is true", () => {
    // GIVEN a modal in the submitting state
    render(
      <RevokeConfirmModal
        open
        keyLabel="x"
        isSubmitting
        onConfirm={() => {}}
        onCancel={() => {}}
      />,
    );

    // THEN both action buttons are disabled
    expect(screen.getByTestId(DATA_TEST_ID.CANCEL_BUTTON)).toBeDisabled();
    expect(screen.getByTestId(DATA_TEST_ID.CONFIRM_BUTTON)).toBeDisabled();
  });
});
