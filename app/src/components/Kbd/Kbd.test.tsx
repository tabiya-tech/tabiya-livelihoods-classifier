import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Kbd, DATA_TEST_ID } from "./Kbd";

describe("Kbd", () => {
  it("renders inside a <kbd> element", () => {
    // GIVEN an expected key combo
    const givenKeyCombo = "⌘ K";

    // WHEN we render a Kbd with that combo
    render(<Kbd>{givenKeyCombo}</Kbd>);

    // THEN the rendered node is a <kbd> tag carrying the given combo
    const kbdNode = screen.getByTestId(DATA_TEST_ID.CONTAINER);
    expect(kbdNode.tagName).toBe("KBD");
    expect(kbdNode).toHaveTextContent(givenKeyCombo);
  });
});
