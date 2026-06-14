import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Divider, DATA_TEST_ID } from "./Divider";

describe("Divider", () => {
  it("renders as a separator with horizontal orientation by default", () => {
    // GIVEN the default expected orientation
    const expectedDefaultOrientation = "horizontal";

    // WHEN we render the default Divider
    render(<Divider />);

    // THEN it exposes the separator role with horizontal orientation
    const dividerNode = screen.getByTestId(DATA_TEST_ID.CONTAINER);
    expect(dividerNode.getAttribute("role")).toBe("separator");
    expect(dividerNode.getAttribute("aria-orientation")).toBe(
      expectedDefaultOrientation,
    );
  });

  it("uses vertical orientation when requested", () => {
    // GIVEN a vertical orientation
    const givenOrientation = "vertical" as const;

    // WHEN we render the Divider with that orientation
    render(<Divider orientation={givenOrientation} />);

    // THEN the separator reports the given orientation
    expect(
      screen.getByTestId(DATA_TEST_ID.CONTAINER).getAttribute("aria-orientation"),
    ).toBe(givenOrientation);
  });
});
