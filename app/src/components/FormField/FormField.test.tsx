import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { FormField, DATA_TEST_ID } from "./FormField";
import { Input, INPUT_DATA_TEST_ID, LABEL_DATA_TEST_ID } from "@/components";

describe("FormField", () => {
  it("associates label, help text, and control via aria-describedby", () => {
    // GIVEN a label and helper text for the field
    const givenLabelText = "Key";
    const givenHelpText = "Pick something memorable";

    // WHEN we render a FormField wrapping an Input with that label and help
    render(
      <FormField label={givenLabelText} help={givenHelpText}>
        <Input />
      </FormField>,
    );

    // THEN the help node carries the given text and the input is wired to it via aria-describedby
    const inputElement = screen.getByTestId(INPUT_DATA_TEST_ID.CONTAINER);
    const helpElement = screen.getByTestId(DATA_TEST_ID.HELP);
    expect(helpElement).toHaveTextContent(givenHelpText);
    expect(inputElement.getAttribute("aria-describedby")).toBe(helpElement.id);
  });

  it("renders the error message and sets aria-invalid on the child", () => {
    // GIVEN an expected error message
    const givenErrorMessage = "That email isn't valid";

    // WHEN we render a FormField with that error
    render(
      <FormField label="Email" error={givenErrorMessage}>
        <Input />
      </FormField>,
    );

    // THEN the error node carries the given message and the child input is aria-invalid
    expect(screen.getByTestId(DATA_TEST_ID.ERROR)).toHaveTextContent(
      givenErrorMessage,
    );
    expect(
      screen.getByTestId(INPUT_DATA_TEST_ID.CONTAINER).getAttribute("aria-invalid"),
    ).toBe("true");
  });

  it("shows the required asterisk when required", () => {
    // GIVEN a FormField with required=true
    // WHEN we render it
    render(
      <FormField label="Email" required>
        <Input />
      </FormField>,
    );

    // THEN the required mark span is visible next to the label
    expect(
      screen.getByTestId(LABEL_DATA_TEST_ID.REQUIRED_MARK),
    ).toBeInTheDocument();
  });
});
