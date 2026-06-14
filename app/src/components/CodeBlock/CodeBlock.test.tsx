import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CodeBlock, DATA_TEST_ID } from "./CodeBlock";

describe("CodeBlock", () => {
  it("renders the code prop as the body", () => {
    // GIVEN an expected code snippet
    const givenCodeSnippet = "curl https://example.com";

    // WHEN we render a CodeBlock with that snippet
    render(<CodeBlock code={givenCodeSnippet} />);

    // THEN the rendered code carries the given snippet
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toHaveTextContent(
      givenCodeSnippet,
    );
  });

  it("applies the muted variant class when muted", () => {
    // GIVEN a CodeBlock with muted=true
    // WHEN we render it
    render(<CodeBlock muted code="ok" />);

    // THEN the pre element carries the muted utility class
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER).className).toMatch(/muted/);
  });

  it("copies text to clipboard when the copy button is clicked", async () => {
    // GIVEN a clipboard spy and the expected code snippet
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    const givenCodeSnippet = "hello world";

    // AND a copyable CodeBlock with that snippet
    render(<CodeBlock copyable code={givenCodeSnippet} />);

    // WHEN the user clicks the copy button
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.COPY_BUTTON));

    // THEN the clipboard is called with the given snippet
    expect(writeText).toHaveBeenCalledWith(givenCodeSnippet);
  });
});
