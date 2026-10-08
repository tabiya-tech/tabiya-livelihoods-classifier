import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import { CreateKeyForm, DATA_TEST_ID } from "./CreateKeyForm";

describe("CreateKeyForm", () => {
  it("disables submit until the label is non-empty", async () => {
    // GIVEN a form with no initial label
    render(<CreateKeyForm onSubmit={() => {}} />);

    // THEN the submit button starts disabled
    expect(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON)).toBeDisabled();

    // WHEN the user types a label
    const givenLabel = "analyst-laptop";
    await userEvent.type(
      screen.getByTestId(DATA_TEST_ID.LABEL_INPUT),
      givenLabel,
    );

    // THEN the submit button is enabled
    expect(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON)).toBeEnabled();
  });

  it("invokes onSubmit with the trimmed label and clears the input", async () => {
    // GIVEN an onSubmit spy
    const onSubmit = vi.fn();
    const givenLabel = "  ci-pipeline  ";
    const expectedLabel = "ci-pipeline";

    // WHEN the user types and submits
    render(<CreateKeyForm onSubmit={onSubmit} />);
    const labelInput = screen.getByTestId(DATA_TEST_ID.LABEL_INPUT);
    await userEvent.type(labelInput, givenLabel);
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON));

    // THEN onSubmit fires with the trimmed value and the input clears
    expect(onSubmit).toHaveBeenCalledWith(expectedLabel);
    expect(labelInput).toHaveValue("");
  });

  it("shows the max-reached helper and disables the form when maxReached is true", () => {
    // GIVEN maxReached + a limit of 5
    const givenMax = 5;
    const expectedHelp = i18n.t("keys.createForm.maxReachedHelp", {
      max: givenMax,
    });

    // WHEN we render in max-reached state
    render(
      <CreateKeyForm onSubmit={() => {}} maxReached maxKeys={givenMax} />,
    );

    // THEN the helper text matches and submit is disabled
    expect(screen.getByTestId(DATA_TEST_ID.MAX_REACHED_HELP)).toHaveTextContent(
      expectedHelp,
    );
    expect(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON)).toBeDisabled();
    expect(screen.getByTestId(DATA_TEST_ID.LABEL_INPUT)).toBeDisabled();
  });

  it("disables the submit button when isSubmitting is true", () => {
    // GIVEN a form in the submitting state
    // WHEN we render it
    render(<CreateKeyForm onSubmit={() => {}} isSubmitting />);

    // THEN the input + submit are disabled
    expect(screen.getByTestId(DATA_TEST_ID.LABEL_INPUT)).toBeDisabled();
    expect(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON)).toBeDisabled();
  });
});
