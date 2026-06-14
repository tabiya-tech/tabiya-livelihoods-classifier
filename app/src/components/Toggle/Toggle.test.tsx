import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Toggle, DATA_TEST_ID } from "./Toggle";

describe("Toggle", () => {
  it("renders with role=switch and reflects aria-checked", () => {
    // GIVEN an accessible label for the toggle
    const givenToggleLabel = "auto-run";

    // WHEN we render a Toggle defaulting to checked
    render(<Toggle defaultChecked label={givenToggleLabel} />);

    // THEN the switch is exposed with aria-checked=true
    expect(
      screen.getByTestId(DATA_TEST_ID.CONTAINER).getAttribute("aria-checked"),
    ).toBe("true");
  });

  it("invokes onChange with the new value on click", async () => {
    // GIVEN an onChange spy and a starting checked value
    const onChange = vi.fn();
    const givenInitialChecked = false;
    const expectedNextChecked = !givenInitialChecked;

    // AND a Toggle bound to that spy starting from the initial value
    render(<Toggle checked={givenInitialChecked} onChange={onChange} />);

    // WHEN the user clicks the switch
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONTAINER));

    // THEN onChange is called with the inverted value
    expect(onChange).toHaveBeenCalledWith(expectedNextChecked);
  });

  it("toggles its own state when uncontrolled", async () => {
    // GIVEN an uncontrolled Toggle starting unchecked
    render(<Toggle defaultChecked={false} />);
    const toggleContainer = screen.getByTestId(DATA_TEST_ID.CONTAINER);
    expect(toggleContainer.getAttribute("aria-checked")).toBe("false");

    // WHEN the user clicks it once
    await userEvent.click(toggleContainer);

    // THEN aria-checked flips to true
    expect(toggleContainer.getAttribute("aria-checked")).toBe("true");

    // AND the thumb is translated to its right-side position
    expect(screen.getByTestId(DATA_TEST_ID.THUMB).className).toMatch(
      /translate-x-\[18px\]/,
    );
  });

  it("does not invoke onChange when disabled", async () => {
    // GIVEN a disabled Toggle and an onChange spy
    const onChange = vi.fn();
    render(<Toggle disabled defaultChecked onChange={onChange} />);

    // WHEN the user clicks the disabled switch
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONTAINER));

    // THEN onChange remains untouched
    expect(onChange).not.toHaveBeenCalled();
  });
});
