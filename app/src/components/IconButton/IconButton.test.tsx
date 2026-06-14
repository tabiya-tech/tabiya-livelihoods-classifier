import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { IconButton, DATA_TEST_ID } from "./IconButton";
import { ICON_DATA_TEST_ID } from "@/components";

describe("IconButton", () => {
  it("exposes its aria-label and renders an Icon inside", () => {
    // GIVEN an accessible label and icon for the button
    const givenAriaLabel = "Copy";
    const givenIconName = "copy";

    // WHEN we render an IconButton with that label and icon
    render(<IconButton icon={givenIconName} aria-label={givenAriaLabel} />);

    // THEN the button carries the given aria-label and an Icon svg is inside it
    const iconButton = screen.getByTestId(DATA_TEST_ID.CONTAINER);
    expect(iconButton.getAttribute("aria-label")).toBe(givenAriaLabel);
    expect(screen.getByTestId(ICON_DATA_TEST_ID.SVG)).toBeInTheDocument();
  });

  it("fires onClick", async () => {
    // GIVEN an onClick spy
    const onClick = vi.fn();

    // AND a rendered IconButton bound to that spy
    render(<IconButton icon="copy" aria-label="Copy" onClick={onClick} />);

    // WHEN the user clicks it
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONTAINER));

    // THEN onClick is called exactly once
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("does not fire onClick when disabled", async () => {
    // GIVEN a disabled IconButton with an onClick spy
    const onClick = vi.fn();
    render(<IconButton icon="trash" aria-label="Delete" disabled onClick={onClick} />);

    // WHEN the user clicks it
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONTAINER));

    // THEN onClick is never called
    expect(onClick).not.toHaveBeenCalled();
  });
});
