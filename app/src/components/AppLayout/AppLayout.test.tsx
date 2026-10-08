import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { AppLayout, DATA_TEST_ID } from "./AppLayout";

const GIVEN_SIDEBAR_TEST_ID = "given-sidebar-slot";
const GIVEN_TOPBAR_TEST_ID = "given-topbar-slot";

describe("AppLayout", () => {
  it("renders sidebar, topbar, and children content", () => {
    // GIVEN main content for the layout
    const givenMainContent = "main content";

    // WHEN we render the AppLayout composed with three slots
    render(
      <AppLayout
        sidebar={<aside data-testid={GIVEN_SIDEBAR_TEST_ID}>sidebar</aside>}
        topbar={<div data-testid={GIVEN_TOPBAR_TEST_ID}>topbar</div>}
      >
        <div>{givenMainContent}</div>
      </AppLayout>,
    );

    // THEN every slot is in the document with the given content
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toBeInTheDocument();
    expect(screen.getByTestId(GIVEN_SIDEBAR_TEST_ID)).toBeInTheDocument();
    expect(screen.getByTestId(GIVEN_TOPBAR_TEST_ID)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.MAIN)).toHaveTextContent(
      givenMainContent,
    );
  });
});
