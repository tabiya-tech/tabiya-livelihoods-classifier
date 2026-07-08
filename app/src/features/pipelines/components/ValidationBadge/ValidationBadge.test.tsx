import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ValidationBadge, DATA_TEST_ID } from "./ValidationBadge";

describe("ValidationBadge", () => {
  it("renders the count inside the badge", () => {
    // GIVEN a badge with count 3 and severity error
    const givenCount = 3;
    const expectedCountText = "3";

    // WHEN we render
    render(<ValidationBadge severity="error" count={givenCount} />);

    // THEN the count is visible
    expect(screen.getByTestId(DATA_TEST_ID.COUNT)).toHaveTextContent(
      expectedCountText,
    );
  });

  it("renders a dot alongside the count", () => {
    // GIVEN a badge
    const givenCount = 1;

    // WHEN we render
    render(<ValidationBadge severity="warning" count={givenCount} />);

    // THEN both the dot and count elements are present
    expect(screen.getByTestId(DATA_TEST_ID.DOT)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.COUNT)).toBeInTheDocument();
  });

  it("passes the title prop as a tooltip attribute on the wrapper", () => {
    // GIVEN a badge with a title
    const givenTitle = "Slot type mismatch";
    const givenCount = 2;

    // WHEN we render
    render(
      <ValidationBadge severity="error" count={givenCount} title={givenTitle} />,
    );

    // THEN the wrapper element carries the title
    expect(screen.getByTestId(DATA_TEST_ID.BADGE)).toHaveAttribute(
      "title",
      givenTitle,
    );
  });

  it("renders correctly for info severity", () => {
    // GIVEN an info-severity badge
    const givenCount = 1;

    // WHEN we render
    render(<ValidationBadge severity="info" count={givenCount} />);

    // THEN the badge element is present
    expect(screen.getByTestId(DATA_TEST_ID.BADGE)).toBeInTheDocument();
  });
});
