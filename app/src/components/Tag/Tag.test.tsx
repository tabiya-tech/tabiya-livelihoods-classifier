import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Tag, DATA_TEST_ID } from "./Tag";

describe("Tag", () => {
  it("renders its children as the label", () => {
    // GIVEN an expected tag label
    const givenTagLabel = "recommended";

    // WHEN we render a Tag with that label
    render(<Tag>{givenTagLabel}</Tag>);

    // THEN the tag container carries the given label
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toHaveTextContent(givenTagLabel);
  });

  it("renders a leading dot when dot is true", () => {
    // GIVEN a Tag with dot=true
    // WHEN we render it
    render(<Tag dot>active</Tag>);

    // THEN the dot span is present in the DOM
    expect(screen.getByTestId(DATA_TEST_ID.DOT)).toBeInTheDocument();
  });

  it("omits the leading dot by default", () => {
    // GIVEN a Tag without dot
    // WHEN we render it
    render(<Tag>plain</Tag>);

    // THEN no dot span exists
    expect(screen.queryByTestId(DATA_TEST_ID.DOT)).not.toBeInTheDocument();
  });

  it("applies the requested tone classes", () => {
    // GIVEN a Tag with tone=lime
    // WHEN we render it
    render(<Tag tone="lime">connected</Tag>);

    // THEN the lime background utility is present
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER).className).toMatch(/bg-lime/);
  });
});
