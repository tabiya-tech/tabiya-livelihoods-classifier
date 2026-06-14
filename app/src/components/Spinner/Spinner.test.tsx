import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Spinner, DATA_TEST_ID } from "./Spinner";

describe("Spinner", () => {
  it("renders with status role for assistive tech", () => {
    // GIVEN the default Spinner
    // WHEN we render it
    render(<Spinner />);

    // THEN it exposes a status role with the default "Loading" label and the container is in the DOM
    expect(screen.getByRole("status", { name: /loading/i })).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toBeInTheDocument();
  });

  it("accepts a custom aria-label", () => {
    // GIVEN an expected aria-label
    const givenAriaLabel = "Classifying";

    // WHEN we render the spinner with that label
    render(<Spinner aria-label={givenAriaLabel} />);

    // THEN the spinner is reachable by that label in the accessibility tree
    expect(screen.getByRole("status", { name: givenAriaLabel })).toBeInTheDocument();
  });

  it("applies the requested pixel size to the inline style", () => {
    // GIVEN an expected size in pixels
    const givenSizeInPixels = 24;
    const expectedCssSize = `${givenSizeInPixels}px`;

    // WHEN we render the spinner with that size
    render(<Spinner size={givenSizeInPixels} />);

    // THEN width and height in the inline style match
    const spinner = screen.getByTestId(DATA_TEST_ID.CONTAINER);
    expect(spinner.style.width).toBe(expectedCssSize);
    expect(spinner.style.height).toBe(expectedCssSize);
  });
});
