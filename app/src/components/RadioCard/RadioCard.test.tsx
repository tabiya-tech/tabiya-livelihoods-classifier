import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RadioCard, DATA_TEST_ID } from "./RadioCard";

describe("RadioCard", () => {
  it("exposes radio role with aria-checked reflecting selection", () => {
    // GIVEN a title for the selected card
    const givenRadioCardTitle = "ESCO v1.2.0";

    // WHEN we render a RadioCard with selected=true
    render(<RadioCard selected title={givenRadioCardTitle} />);

    // THEN the radio reports aria-checked=true and the selected indicator dot is visible
    expect(
      screen.getByTestId(DATA_TEST_ID.CONTAINER).getAttribute("aria-checked"),
    ).toBe("true");
    expect(screen.getByTestId(DATA_TEST_ID.INDICATOR_DOT)).toBeInTheDocument();
  });

  it("hides the indicator dot when unselected", () => {
    // GIVEN a RadioCard with selected omitted
    // WHEN we render it
    render(<RadioCard title="Unpicked" />);

    // THEN the dot is absent (indicator ring still rendered)
    expect(screen.queryByTestId(DATA_TEST_ID.INDICATOR_DOT)).not.toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.INDICATOR)).toBeInTheDocument();
  });

  it("fires onClick to allow consumers to update selection", async () => {
    // GIVEN an onClick spy
    const onClick = vi.fn();

    // AND a rendered RadioCard bound to that spy
    render(<RadioCard title="Pick me" onClick={onClick} />);

    // WHEN the user clicks the card
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONTAINER));

    // THEN onClick is invoked exactly once
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("renders title, description, and meta slots", () => {
    // GIVEN title, description, and meta content for the card
    const givenRadioCardTitle = "MPNet base v2";
    const givenRadioCardDescription = "Higher quality embeddings.";
    const givenRadioCardMetaLabel = "768-dim";

    // WHEN we render a RadioCard with all three slots
    render(
      <RadioCard
        title={givenRadioCardTitle}
        description={givenRadioCardDescription}
        meta={<span>{givenRadioCardMetaLabel}</span>}
      />,
    );

    // THEN each slot carries the given content
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(
      givenRadioCardTitle,
    );
    expect(screen.getByTestId(DATA_TEST_ID.DESCRIPTION)).toHaveTextContent(
      givenRadioCardDescription,
    );
    expect(screen.getByTestId(DATA_TEST_ID.META)).toHaveTextContent(
      givenRadioCardMetaLabel,
    );
  });
});
