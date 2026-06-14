import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Input, DATA_TEST_ID } from "./Input";

describe("Input", () => {
  it("accepts user keystrokes", async () => {
    // GIVEN text the user is expected to type
    const givenTypedText = "Sara";

    // AND a rendered Input
    render(<Input />);

    // WHEN the user types that text into the input
    await userEvent.type(screen.getByTestId(DATA_TEST_ID.CONTAINER), givenTypedText);

    // THEN the input's value reflects the typed text
    expect(
      (screen.getByTestId(DATA_TEST_ID.CONTAINER) as HTMLInputElement).value,
    ).toBe(givenTypedText);
  });

  it("fires onChange for each keystroke", async () => {
    // GIVEN an onChange spy and a sequence of characters to type
    const onChange = vi.fn();
    const givenTypedText = "abc";

    // AND a rendered Input bound to that spy
    render(<Input onChange={onChange} />);

    // WHEN the user types each character
    await userEvent.type(screen.getByTestId(DATA_TEST_ID.CONTAINER), givenTypedText);

    // THEN onChange is invoked once per keystroke
    expect(onChange).toHaveBeenCalledTimes(givenTypedText.length);
  });

  it("applies aria-invalid when invalid", () => {
    // GIVEN an Input with invalid=true
    // WHEN we render it
    render(<Input invalid />);

    // THEN aria-invalid='true' is on the element
    expect(
      screen.getByTestId(DATA_TEST_ID.CONTAINER).getAttribute("aria-invalid"),
    ).toBe("true");
  });

  it("uses the mono face when requested", () => {
    // GIVEN an Input with mono=true
    // WHEN we render it
    render(<Input mono />);

    // THEN the input carries the mono utility class
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER).className).toMatch(/font-mono/);
  });
});
