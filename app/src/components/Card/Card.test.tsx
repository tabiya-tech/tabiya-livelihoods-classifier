import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Card, CardHead, DATA_TEST_ID } from "./Card";

describe("Card", () => {
  it("renders children inside a styled container", () => {
    // GIVEN expected card contents
    const givenCardContents = "contents";

    // WHEN we render a Card with those contents
    render(<Card>{givenCardContents}</Card>);

    // THEN the container carries the given contents
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toHaveTextContent(
      givenCardContents,
    );
  });

  it("omits inner padding when flush", () => {
    // GIVEN a Card with flush=true
    // WHEN we render it
    render(<Card flush>x</Card>);

    // THEN the rendered element does not carry the p-5 utility
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER).className).not.toMatch(/p-5/);
  });

  it("applies the elevated shadow class when elevated", () => {
    // GIVEN a Card with elevated=true
    // WHEN we render it
    render(<Card elevated>x</Card>);

    // THEN the container carries the elevated shadow utility
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER).className).toMatch(
      /shadow-card-2/,
    );
  });
});

describe("CardHead", () => {
  it("renders title and action slots", () => {
    // GIVEN a title text and action label
    const givenCardTitle = "Title";
    const givenCardActionLabel = "act";

    // WHEN we render a CardHead with that title and action
    render(
      <CardHead
        title={givenCardTitle}
        action={<span>{givenCardActionLabel}</span>}
      />,
    );

    // THEN both slots carry their given content
    expect(screen.getByTestId(DATA_TEST_ID.HEAD_TITLE)).toHaveTextContent(
      givenCardTitle,
    );
    expect(screen.getByTestId(DATA_TEST_ID.HEAD_ACTION)).toHaveTextContent(
      givenCardActionLabel,
    );
  });
});
