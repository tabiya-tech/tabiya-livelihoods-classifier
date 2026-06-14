import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Topbar, DATA_TEST_ID } from "./Topbar";
import { BREADCRUMBS_DATA_TEST_ID } from "@/components";

describe("Topbar", () => {
  it("renders breadcrumbs and right slot", () => {
    // GIVEN breadcrumb labels and a right-slot content
    const givenBreadcrumbLabels = ["Settings", "Keys"];
    const givenRightSlotContent = "v1.0.0";

    // WHEN we render the Topbar with those breadcrumbs and the right slot
    render(
      <Topbar
        breadcrumbs={givenBreadcrumbLabels.map((label) => ({ label }))}
        right={<span>{givenRightSlotContent}</span>}
      />,
    );

    // THEN the container, breadcrumb nav, and right-slot content all render
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toBeInTheDocument();
    expect(screen.getByTestId(BREADCRUMBS_DATA_TEST_ID.NAV)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.RIGHT_SLOT)).toHaveTextContent(
      givenRightSlotContent,
    );
  });

  it("omits the right slot when no right content is provided", () => {
    // GIVEN a Topbar with no right slot
    // WHEN we render it
    render(<Topbar breadcrumbs={[{ label: "x" }]} />);

    // THEN the right slot is absent from the DOM
    expect(screen.queryByTestId(DATA_TEST_ID.RIGHT_SLOT)).not.toBeInTheDocument();
  });
});
