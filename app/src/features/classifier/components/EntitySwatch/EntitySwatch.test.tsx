import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { DATA_TEST_ID, EntitySwatch } from "./EntitySwatch";

describe("EntitySwatch", () => {
  it("tags the dot with the entity type via data-type", () => {
    // GIVEN a swatch for an occupation
    render(<EntitySwatch entityType="occupation" />);

    // THEN the dot carries data-type="occupation"
    expect(screen.getByTestId(DATA_TEST_ID.DOT)).toHaveAttribute(
      "data-type",
      "occupation",
    );
  });

  it("applies the requested size in pixels", () => {
    // GIVEN a custom size
    const givenSize = 16;

    // WHEN we render
    render(<EntitySwatch entityType="skill" size={givenSize} />);

    // THEN the inline style reflects the size
    const renderedDot = screen.getByTestId(DATA_TEST_ID.DOT);
    expect(renderedDot.style.width).toBe(`${givenSize}px`);
    expect(renderedDot.style.height).toBe(`${givenSize}px`);
  });
});
