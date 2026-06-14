import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatusPill, DATA_TEST_ID } from "./StatusPill";

describe("StatusPill", () => {
  it("renders its children as the label", () => {
    // GIVEN an expected status pill label
    const givenStatusLabel = "API healthy";

    // WHEN we render a StatusPill with that label
    render(<StatusPill>{givenStatusLabel}</StatusPill>);

    // THEN the container carries the given label
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toHaveTextContent(
      givenStatusLabel,
    );
  });

  it("uses the lime dot color when healthy", () => {
    // GIVEN a StatusPill with status='healthy'
    // WHEN we render it
    render(<StatusPill status="healthy">ok</StatusPill>);

    // THEN the leading dot carries the lime utility
    expect(screen.getByTestId(DATA_TEST_ID.DOT).className).toMatch(/bg-lime/);
  });

  it("uses the error dot color when down", () => {
    // GIVEN a StatusPill with status='down'
    // WHEN we render it
    render(<StatusPill status="down">offline</StatusPill>);

    // THEN the leading dot uses the error utility
    expect(screen.getByTestId(DATA_TEST_ID.DOT).className).toMatch(/bg-error/);
  });
});
