import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { fixtureClassifyResponse } from "@/mocks/fixtures/classify";
import { DATA_TEST_ID, JsonView } from "./JsonView";

describe("JsonView", () => {
  beforeEach(() => {
    // Stub URL.createObjectURL / revokeObjectURL for jsdom.
    Object.assign(URL, {
      createObjectURL: vi.fn(() => "blob:test"),
      revokeObjectURL: vi.fn(),
    });
  });

  it("renders the value as pretty-printed JSON", () => {
    // GIVEN a simple object
    const givenValue = { foo: "bar", n: 1 };
    const expectedJson = JSON.stringify(givenValue, null, 2);

    // WHEN we render
    render(<JsonView value={givenValue} />);

    // THEN the pre block contains the formatted JSON (verbatim, preserving
    // whitespace — toHaveTextContent collapses it, so compare textContent).
    expect(screen.getByTestId(DATA_TEST_ID.PRE).textContent).toBe(expectedJson);
  });

  it("triggers a download with the supplied filename when the button is clicked", async () => {
    // GIVEN a known filename
    const givenFilename = "my-classification";
    const createObjectURLSpy = URL.createObjectURL as ReturnType<typeof vi.fn>;

    // Spy on the anchor click so we can assert the filename + extension.
    const anchorClickSpy = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tagName: string) => {
      const element = originalCreateElement(tagName);
      if (tagName === "a") {
        Object.defineProperty(element, "click", { value: anchorClickSpy });
      }
      return element;
    });

    // WHEN the user clicks Download
    render(<JsonView value={fixtureClassifyResponse} filename={givenFilename} />);
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.DOWNLOAD_BUTTON));

    // THEN the click fires and the URL is created
    expect(anchorClickSpy).toHaveBeenCalledTimes(1);
    expect(createObjectURLSpy).toHaveBeenCalledTimes(1);
  });
});
