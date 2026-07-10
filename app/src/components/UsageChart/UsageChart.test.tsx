import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { UsageChart, DATA_TEST_ID } from "./UsageChart";
import type { DailyCount } from "@/lib/api";

describe("UsageChart", () => {
  it("renders the chart when data is provided", () => {
    // GIVEN non-empty daily count data
    const givenData: DailyCount[] = [
      { date: "2026-07-08", count: 5 },
      { date: "2026-07-09", count: 12 },
      { date: "2026-07-10", count: 3 },
    ];

    // WHEN rendered
    render(<UsageChart data={givenData} />);

    // THEN the chart container is in the document
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toBeInTheDocument();
  });

  it("renders the empty state when data is empty", () => {
    // GIVEN empty data
    const givenData: DailyCount[] = [];

    // WHEN rendered
    render(<UsageChart data={givenData} />);

    // THEN the empty state is shown instead of the chart
    expect(screen.getByTestId(DATA_TEST_ID.EMPTY)).toBeInTheDocument();
    expect(screen.queryByTestId(DATA_TEST_ID.CONTAINER)).not.toBeInTheDocument();
  });
});
