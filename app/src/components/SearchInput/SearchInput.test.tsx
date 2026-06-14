import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SearchInput, DATA_TEST_ID } from "./SearchInput";
import { ICON_DATA_TEST_ID } from "@/components";

describe("SearchInput", () => {
  it("renders a search-type input with the leading icon", () => {
    // GIVEN the default SearchInput
    // WHEN we render it
    render(<SearchInput />);

    // THEN the input has type=search and the search Icon is present
    const searchInput = screen.getByTestId(DATA_TEST_ID.INPUT) as HTMLInputElement;
    expect(searchInput.type).toBe("search");
    expect(screen.getByTestId(ICON_DATA_TEST_ID.SVG)).toBeInTheDocument();
  });

  it("accepts typed text and fires onChange", async () => {
    // GIVEN an onChange spy and a sequence of characters to type
    const onChange = vi.fn();
    const givenTypedText = "data";

    // AND a rendered SearchInput bound to that spy
    render(<SearchInput onChange={onChange} />);

    // WHEN the user types each character
    await userEvent.type(screen.getByTestId(DATA_TEST_ID.INPUT), givenTypedText);

    // THEN onChange fires per keystroke and the input value reflects the typed text
    expect(onChange).toHaveBeenCalledTimes(givenTypedText.length);
    expect(
      (screen.getByTestId(DATA_TEST_ID.INPUT) as HTMLInputElement).value,
    ).toBe(givenTypedText);
  });
});
