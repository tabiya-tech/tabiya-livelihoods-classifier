import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Label, DATA_TEST_ID } from "./Label";

describe("Label", () => {
  it("renders children as label text", () => {
    // GIVEN a label text and a target field id
    const givenLabelText = "Email";
    const givenForFieldId = "email-field";

    // WHEN we render a Label associated with that field
    render(<Label htmlFor={givenForFieldId}>{givenLabelText}</Label>);

    // THEN the label uses the <label> tag, carries the given text, and is bound to the field id
    const labelNode = screen.getByTestId(DATA_TEST_ID.CONTAINER);
    expect(labelNode.tagName).toBe("LABEL");
    expect(labelNode).toHaveTextContent(givenLabelText);
    expect(labelNode.getAttribute("for")).toBe(givenForFieldId);
  });

  it("shows a required marker when required", () => {
    // GIVEN a Label with required=true
    // WHEN we render it
    render(<Label required>Email</Label>);

    // THEN the required mark span is in the DOM
    expect(screen.getByTestId(DATA_TEST_ID.REQUIRED_MARK)).toBeInTheDocument();
  });

  it("omits the required marker by default", () => {
    // GIVEN a Label without required
    // WHEN we render it
    render(<Label>Email</Label>);

    // THEN the required marker is absent
    expect(screen.queryByTestId(DATA_TEST_ID.REQUIRED_MARK)).not.toBeInTheDocument();
  });
});
