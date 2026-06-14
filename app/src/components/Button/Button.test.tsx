import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button, DATA_TEST_ID } from "./Button";

describe("Button", () => {
  it("renders its children as the label", () => {
    // GIVEN an expected label text
    const givenLabelText = "Save";

    // WHEN we render a Button with that label
    render(<Button>{givenLabelText}</Button>);

    // THEN the label node carries the given text
    expect(screen.getByTestId(DATA_TEST_ID.LABEL)).toHaveTextContent(givenLabelText);
  });

  it("fires onClick when activated", async () => {
    // GIVEN an onClick spy
    const onClick = vi.fn();

    // AND a rendered Button bound to that spy
    render(<Button onClick={onClick}>Run</Button>);

    // WHEN the user clicks the button
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONTAINER));

    // THEN the spy is called exactly once
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("is disabled and skips clicks while loading", async () => {
    // GIVEN an onClick spy
    const onClick = vi.fn();

    // AND a Button rendered with loading=true
    render(
      <Button loading onClick={onClick}>
        Running
      </Button>,
    );

    // WHEN the user clicks the loading button
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONTAINER));

    // THEN the button is marked disabled, the spinner is shown, and onClick is never called
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toBeDisabled();
    expect(screen.getByTestId(DATA_TEST_ID.SPINNER)).toBeInTheDocument();
    expect(onClick).not.toHaveBeenCalled();
  });

  it("does not render the trailing slot while loading", () => {
    // GIVEN a Button with a trailing icon-like marker and loading=true
    // WHEN it renders
    render(
      <Button loading trailing={<span>→</span>}>
        Running
      </Button>,
    );

    // THEN the trailing slot is suppressed in favor of the spinner
    expect(screen.queryByTestId(DATA_TEST_ID.TRAILING)).not.toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.SPINNER)).toBeInTheDocument();
  });

  it("renders leading and trailing slots when provided and not loading", () => {
    // GIVEN a Button with both leading and trailing slots
    // WHEN it renders
    render(
      <Button leading={<span>←</span>} trailing={<span>→</span>}>
        Go
      </Button>,
    );

    // THEN both slot containers are present in the DOM
    expect(screen.getByTestId(DATA_TEST_ID.LEADING)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.TRAILING)).toBeInTheDocument();
  });

  it("applies the primary variant classes", () => {
    // GIVEN a primary Button
    // WHEN it renders
    render(<Button variant="primary">Go</Button>);

    // THEN the button element carries the primary background utility
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER).className).toMatch(/bg-navy/);
  });
});
