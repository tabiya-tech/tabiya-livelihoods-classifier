import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Tabs, DATA_TEST_ID } from "./Tabs";

const givenTabItems = [
  { id: "a", label: "A" },
  { id: "b", label: "B" },
  { id: "c", label: "C", disabled: true },
];

function findRenderedTabByItemId(itemId: string) {
  return screen
    .getAllByTestId(DATA_TEST_ID.TAB)
    .find((node) => node.getAttribute("data-tab-id") === itemId);
}

describe("Tabs", () => {
  it("marks the current tab as selected", () => {
    // GIVEN the active tab id and an inactive tab id
    const givenActiveTabId = "b";
    const givenInactiveTabId = "a";

    // WHEN we render Tabs with the active tab selected
    render(
      <Tabs
        aria-label="x"
        items={givenTabItems}
        value={givenActiveTabId}
        onChange={() => {}}
      />,
    );

    // THEN the active tab reports aria-selected=true and the inactive one reports false
    expect(
      findRenderedTabByItemId(givenActiveTabId)?.getAttribute("aria-selected"),
    ).toBe("true");
    expect(
      findRenderedTabByItemId(givenInactiveTabId)?.getAttribute("aria-selected"),
    ).toBe("false");
  });

  it("calls onChange when a different tab is clicked", async () => {
    // GIVEN an onChange spy, the initial active tab, and the tab the user will click
    const onChange = vi.fn();
    const givenInitialActiveTabId = "a";
    const givenClickedTabId = "b";

    // AND rendered Tabs starting on the initial tab
    render(
      <Tabs
        aria-label="x"
        items={givenTabItems}
        value={givenInitialActiveTabId}
        onChange={onChange}
      />,
    );

    // WHEN the user clicks the next tab
    await userEvent.click(findRenderedTabByItemId(givenClickedTabId)!);

    // THEN onChange is called with the clicked tab's id
    expect(onChange).toHaveBeenCalledWith(givenClickedTabId);
  });

  it("does not call onChange when a disabled tab is clicked", async () => {
    // GIVEN an onChange spy and a disabled tab id
    const onChange = vi.fn();
    const givenDisabledTabId = "c";

    // AND rendered Tabs with the disabled tab present
    render(
      <Tabs
        aria-label="x"
        items={givenTabItems}
        value="a"
        onChange={onChange}
      />,
    );

    // WHEN the user clicks the disabled tab
    await userEvent.click(findRenderedTabByItemId(givenDisabledTabId)!);

    // THEN onChange is never called
    expect(onChange).not.toHaveBeenCalled();
  });
});
