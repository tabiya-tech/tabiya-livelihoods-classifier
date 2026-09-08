import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NavLink, DATA_TEST_ID } from "./NavLink";

describe("NavLink", () => {
  it("renders children and exposes aria-current when active", () => {
    // GIVEN an expected nav link label
    const givenNavLinkLabel = "Classifier";

    // WHEN we render an active NavLink with that label
    render(<NavLink active>{givenNavLinkLabel}</NavLink>);

    // THEN the active item is marked aria-current=page and the label carries the given text
    expect(
      screen.getByTestId(DATA_TEST_ID.CONTAINER).getAttribute("aria-current"),
    ).toBe("page");
    expect(screen.getByTestId(DATA_TEST_ID.LABEL)).toHaveTextContent(
      givenNavLinkLabel,
    );
  });

  it("does not set aria-current when inactive", () => {
    // GIVEN active is omitted (defaults to false)
    // WHEN we render the NavLink
    render(<NavLink>Dashboard</NavLink>);

    // THEN aria-current is absent
    expect(
      screen.getByTestId(DATA_TEST_ID.CONTAINER).getAttribute("aria-current"),
    ).toBeNull();
  });

  it("renders the icon slot when an icon is provided", () => {
    // GIVEN a NavLink with an icon prop
    // WHEN we render it
    render(<NavLink icon={<span>★</span>}>Starred</NavLink>);

    // THEN the icon slot is in the document
    expect(screen.getByTestId(DATA_TEST_ID.ICON_SLOT)).toBeInTheDocument();
  });

  it("fires onClick when activated", async () => {
    // GIVEN an onClick spy
    const onClick = vi.fn();

    // AND a rendered NavLink bound to that spy
    render(<NavLink onClick={onClick}>Go</NavLink>);

    // WHEN the user clicks the link
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONTAINER));

    // THEN onClick is invoked exactly once
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
