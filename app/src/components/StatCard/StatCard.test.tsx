import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatCard, DATA_TEST_ID } from "./StatCard";

describe("StatCard", () => {
  it("renders label and value", () => {
    // GIVEN a label and value
    const givenLabel = "Calls this week";
    const givenValue = "42";

    // WHEN rendered
    render(<StatCard label={givenLabel} value={givenValue} />);

    // THEN both are visible
    expect(screen.getByTestId(DATA_TEST_ID.LABEL)).toHaveTextContent(givenLabel);
    expect(screen.getByTestId(DATA_TEST_ID.VALUE)).toHaveTextContent(givenValue);
  });

  it("renders delta when provided", () => {
    // GIVEN a delta string
    const givenDelta = "+12 vs last week";

    // WHEN rendered with a delta
    render(<StatCard label="Calls" value="99" delta={givenDelta} />);

    // THEN delta is visible
    expect(screen.getByTestId(DATA_TEST_ID.DELTA)).toHaveTextContent(givenDelta);
  });

  it("omits delta element when not provided", () => {
    // GIVEN no delta prop
    // WHEN rendered without delta
    render(<StatCard label="Calls" value="0" />);

    // THEN no delta element in the DOM
    expect(screen.queryByTestId(DATA_TEST_ID.DELTA)).not.toBeInTheDocument();
  });

  it("applies positive delta class when deltaPositive is true", () => {
    // GIVEN a positive delta
    const givenDelta = "+5 vs last week";

    // WHEN rendered with deltaPositive=true
    render(<StatCard label="Calls" value="20" delta={givenDelta} deltaPositive={true} />);

    // THEN the delta element has the teal colour class
    expect(screen.getByTestId(DATA_TEST_ID.DELTA)).toHaveClass("text-teal");
  });

  it("applies negative delta class when deltaPositive is false", () => {
    // GIVEN a negative delta
    const givenDelta = "-3 vs last week";

    // WHEN rendered with deltaPositive=false
    render(<StatCard label="Calls" value="12" delta={givenDelta} deltaPositive={false} />);

    // THEN the delta element has the error colour class
    expect(screen.getByTestId(DATA_TEST_ID.DELTA)).toHaveClass("text-error");
  });
});
