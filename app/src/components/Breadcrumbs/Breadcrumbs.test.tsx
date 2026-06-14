import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Breadcrumbs, DATA_TEST_ID } from "./Breadcrumbs";

describe("Breadcrumbs", () => {
  it("renders one list item per breadcrumb", () => {
    // GIVEN three breadcrumb labels
    const givenBreadcrumbLabels = ["Docs", "Endpoints", "POST /classify"];

    // WHEN we render Breadcrumbs with those labels
    render(
      <Breadcrumbs
        items={givenBreadcrumbLabels.map((label) => ({ label }))}
      />,
    );

    // THEN exactly that many items render, with the right text in each position
    const renderedItems = screen.getAllByTestId(DATA_TEST_ID.ITEM);
    expect(renderedItems).toHaveLength(givenBreadcrumbLabels.length);
    givenBreadcrumbLabels.forEach((label, index) => {
      expect(renderedItems[index]).toHaveTextContent(label);
    });
  });

  it("renders intermediate items as anchors when onClick is provided", () => {
    // GIVEN a clickable intermediate label and a terminal label
    const givenClickableLabel = "Docs";
    const givenTerminalLabel = "Endpoints";

    // WHEN we render Breadcrumbs with the clickable label first
    render(
      <Breadcrumbs
        items={[
          { label: givenClickableLabel, onClick: () => {} },
          { label: givenTerminalLabel },
        ]}
      />,
    );

    // THEN the clickable item renders as a single link with that label
    const renderedLinks = screen.getAllByTestId(DATA_TEST_ID.LINK);
    expect(renderedLinks).toHaveLength(1);
    expect(renderedLinks[0]).toHaveTextContent(givenClickableLabel);
  });

  it("calls onClick when an item is activated", async () => {
    // GIVEN an onClick spy
    const onClick = vi.fn();

    // AND a rendered breadcrumb bound to that handler
    render(
      <Breadcrumbs items={[{ label: "Home", onClick }, { label: "Now" }]} />,
    );

    // WHEN the user clicks the breadcrumb link
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.LINK));

    // THEN onClick is invoked exactly once
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
