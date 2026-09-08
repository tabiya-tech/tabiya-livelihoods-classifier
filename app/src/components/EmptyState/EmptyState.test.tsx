import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { EmptyState, DATA_TEST_ID } from "./EmptyState";

describe("EmptyState", () => {
  it("renders title and optional description and action", () => {
    // GIVEN title, description, and action content
    const givenEmptyStateTitle = "No keys yet";
    const givenEmptyStateDescription = "Create one above.";
    const givenEmptyStateActionLabel = "create-cta";

    // WHEN we render an EmptyState with all three slots
    render(
      <EmptyState
        title={givenEmptyStateTitle}
        description={givenEmptyStateDescription}
        action={<span>{givenEmptyStateActionLabel}</span>}
      />,
    );

    // THEN each slot carries the given content
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(
      givenEmptyStateTitle,
    );
    expect(screen.getByTestId(DATA_TEST_ID.DESCRIPTION)).toHaveTextContent(
      givenEmptyStateDescription,
    );
    expect(screen.getByTestId(DATA_TEST_ID.ACTION)).toHaveTextContent(
      givenEmptyStateActionLabel,
    );
  });

  it("renders without a description or action", () => {
    // GIVEN an EmptyState with just a title
    const givenEmptyStateTitle = "Nothing here";

    // WHEN we render it
    render(<EmptyState title={givenEmptyStateTitle} />);

    // THEN only the title is visible and the optional slots are absent
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(
      givenEmptyStateTitle,
    );
    expect(screen.queryByTestId(DATA_TEST_ID.DESCRIPTION)).not.toBeInTheDocument();
    expect(screen.queryByTestId(DATA_TEST_ID.ACTION)).not.toBeInTheDocument();
  });
});
