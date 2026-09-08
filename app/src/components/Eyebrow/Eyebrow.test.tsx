import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Eyebrow, DATA_TEST_ID } from "./Eyebrow";

describe("Eyebrow", () => {
  it("renders children inside an .eyebrow container", () => {
    // GIVEN an expected eyebrow text
    const givenEyebrowText = "Settings · Pipeline";

    // WHEN we render an Eyebrow with that text
    render(<Eyebrow>{givenEyebrowText}</Eyebrow>);

    // THEN the eyebrow container is in the DOM, carries the given text, and the editorial class
    const eyebrowNode = screen.getByTestId(DATA_TEST_ID.CONTAINER);
    expect(eyebrowNode).toBeInTheDocument();
    expect(eyebrowNode).toHaveTextContent(givenEyebrowText);
    expect(eyebrowNode.className).toMatch(/eyebrow/);
  });
});
