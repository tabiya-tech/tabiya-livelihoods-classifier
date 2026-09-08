import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Textarea, DATA_TEST_ID } from "./Textarea";

describe("Textarea", () => {
  it("accepts user keystrokes", async () => {
    // GIVEN multi-line text the user is expected to type
    const givenTypedText = "Senior Engineer\nPython, SQL";

    // AND a rendered Textarea
    render(<Textarea />);

    // WHEN the user types that text
    await userEvent.type(screen.getByTestId(DATA_TEST_ID.CONTAINER), givenTypedText);

    // THEN the textarea value reflects the typed content
    expect(
      (screen.getByTestId(DATA_TEST_ID.CONTAINER) as HTMLTextAreaElement).value,
    ).toBe(givenTypedText);
  });

  it("fires onChange while the user types", async () => {
    // GIVEN an onChange spy and a sequence of characters to type
    const onChange = vi.fn();
    const givenTypedText = "hi";

    // AND a rendered Textarea bound to it
    render(<Textarea onChange={onChange} />);

    // WHEN the user types each character
    await userEvent.type(screen.getByTestId(DATA_TEST_ID.CONTAINER), givenTypedText);

    // THEN onChange is invoked once per keystroke
    expect(onChange).toHaveBeenCalledTimes(givenTypedText.length);
  });

  it("applies aria-invalid when invalid", () => {
    // GIVEN a Textarea with invalid=true
    // WHEN it renders
    render(<Textarea invalid />);

    // THEN aria-invalid is true on the element
    expect(
      screen.getByTestId(DATA_TEST_ID.CONTAINER).getAttribute("aria-invalid"),
    ).toBe("true");
  });
});
