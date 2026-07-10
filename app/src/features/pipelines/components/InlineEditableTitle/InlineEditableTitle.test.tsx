import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DATA_TEST_ID, InlineEditableTitle } from "./InlineEditableTitle";

describe("InlineEditableTitle", () => {
  it("shows the value as plain text with no input until clicked", () => {
    // GIVEN a title with a value
    const givenValue = "My Pipeline";

    // WHEN rendered
    render(<InlineEditableTitle value={givenValue} onChange={vi.fn()} />);

    // THEN the text shows and there is no input yet
    expect(screen.getByTestId(DATA_TEST_ID.DISPLAY)).toHaveTextContent(givenValue);
    expect(screen.queryByTestId(DATA_TEST_ID.INPUT)).toBeNull();
  });

  it("shows the placeholder when the value is empty", () => {
    // GIVEN an empty value and a placeholder
    const givenPlaceholder = "Untitled pipeline";

    // WHEN rendered
    render(
      <InlineEditableTitle value="" onChange={vi.fn()} placeholder={givenPlaceholder} />,
    );

    // THEN the placeholder text shows
    expect(screen.getByTestId(DATA_TEST_ID.DISPLAY)).toHaveTextContent(givenPlaceholder);
  });

  it("commits a new value on Enter", async () => {
    // GIVEN an editable title
    const onChange = vi.fn();
    const givenValue = "Old";
    const givenNewValue = "New name";
    render(<InlineEditableTitle value={givenValue} onChange={onChange} />);

    // WHEN the user clicks, clears, types, and presses Enter
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.DISPLAY));
    const input = screen.getByTestId(DATA_TEST_ID.INPUT);
    await userEvent.clear(input);
    await userEvent.type(input, `${givenNewValue}{Enter}`);

    // THEN onChange fires with the trimmed new value
    expect(onChange).toHaveBeenCalledWith(givenNewValue);
  });

  it("reverts on Escape without calling onChange", async () => {
    // GIVEN an editable title
    const onChange = vi.fn();
    const givenValue = "Keep me";
    render(<InlineEditableTitle value={givenValue} onChange={onChange} />);

    // WHEN the user edits then presses Escape
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.DISPLAY));
    await userEvent.type(screen.getByTestId(DATA_TEST_ID.INPUT), " changed{Escape}");

    // THEN onChange is not called and the original text is restored
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByTestId(DATA_TEST_ID.DISPLAY)).toHaveTextContent(givenValue);
  });

  it("is not clickable when readOnly", async () => {
    // GIVEN a read-only title
    const onChange = vi.fn();
    const givenValue = "Default Tabiya";
    render(<InlineEditableTitle value={givenValue} onChange={onChange} readOnly />);

    // WHEN the user clicks it
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.ROOT));

    // THEN it never enters edit mode
    expect(screen.queryByTestId(DATA_TEST_ID.INPUT)).toBeNull();
  });
});
