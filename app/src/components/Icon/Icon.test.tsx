import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Icon, DATA_TEST_ID } from "./Icon";

describe("Icon", () => {
  it("renders an svg with the given icon name as a data attribute", () => {
    // GIVEN an icon name to render
    const givenIconName = "check";

    // WHEN we render the Icon with that name
    render(<Icon name={givenIconName} />);

    // THEN an svg is rendered and the icon name is recorded for selector targeting
    const svg = screen.getByTestId(DATA_TEST_ID.SVG);
    expect(svg.tagName).toBe("svg");
    expect(svg.getAttribute("data-icon")).toBe(givenIconName);
  });

  it("applies a custom size to width and height", () => {
    // GIVEN an expected pixel size
    const givenIconSize = 32;
    const expectedSizeAttribute = String(givenIconSize);

    // WHEN we render the Icon with that size
    render(<Icon name="plus" size={givenIconSize} />);

    // THEN the svg has matching width and height attributes
    const svg = screen.getByTestId(DATA_TEST_ID.SVG);
    expect(svg.getAttribute("width")).toBe(expectedSizeAttribute);
    expect(svg.getAttribute("height")).toBe(expectedSizeAttribute);
  });

  it("is marked aria-hidden so it is decorative by default", () => {
    // GIVEN the default Icon
    // WHEN we render it
    render(<Icon name="docs" />);

    // THEN the svg is hidden from assistive tech
    expect(screen.getByTestId(DATA_TEST_ID.SVG).getAttribute("aria-hidden")).toBe(
      "true",
    );
  });
});
