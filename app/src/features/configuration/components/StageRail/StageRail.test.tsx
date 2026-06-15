import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StageRail, DATA_TEST_ID } from "./StageRail";

const givenStages = [
  {
    id: "nel" as const,
    number: "01",
    label: "NEL",
    subLabel: "Entity linking",
    currentValue: "MPNet base v2",
  },
  {
    id: "taxonomy" as const,
    number: "02",
    label: "Taxonomy",
    subLabel: "Reference vocabulary",
    currentValue: "ESCO v1.2.0",
  },
];

function findRailItemByStageId(stageId: string) {
  return screen
    .getAllByTestId(DATA_TEST_ID.ITEM)
    .find((node) => node.getAttribute("data-stage-id") === stageId);
}

describe("StageRail", () => {
  it("renders one item per stage with its number, label, sub-label, and current value", () => {
    // GIVEN a stage rail with two items and the NEL stage active
    // WHEN we render it
    render(
      <StageRail items={givenStages} activeId="nel" onSelect={() => {}} />,
    );

    // THEN exactly two items are rendered and each shows the given strings
    const renderedItems = screen.getAllByTestId(DATA_TEST_ID.ITEM);
    expect(renderedItems).toHaveLength(givenStages.length);

    const renderedLabels = screen.getAllByTestId(DATA_TEST_ID.ITEM_LABEL);
    expect(renderedLabels[0]).toHaveTextContent(givenStages[0].label);
    expect(renderedLabels[1]).toHaveTextContent(givenStages[1].label);

    const renderedNumbers = screen.getAllByTestId(DATA_TEST_ID.ITEM_NUMBER);
    expect(renderedNumbers[0]).toHaveTextContent(givenStages[0].number);
    expect(renderedNumbers[1]).toHaveTextContent(givenStages[1].number);

    const renderedCurrentValues = screen.getAllByTestId(
      DATA_TEST_ID.ITEM_CURRENT_VALUE,
    );
    expect(renderedCurrentValues[0]).toHaveTextContent(
      givenStages[0].currentValue,
    );
    expect(renderedCurrentValues[1]).toHaveTextContent(
      givenStages[1].currentValue,
    );
  });

  it("marks the active stage with aria-current=step and leaves the others unmarked", () => {
    // GIVEN the taxonomy stage active
    // WHEN we render the rail
    render(
      <StageRail items={givenStages} activeId="taxonomy" onSelect={() => {}} />,
    );

    // THEN the taxonomy item carries aria-current=step and NEL does not
    expect(findRailItemByStageId("taxonomy")?.getAttribute("aria-current")).toBe(
      "step",
    );
    expect(findRailItemByStageId("nel")?.getAttribute("aria-current")).toBeNull();
  });

  it("calls onSelect with the stage id when a non-active item is clicked", async () => {
    // GIVEN an onSelect spy and the NEL stage active
    const onSelect = vi.fn();
    render(<StageRail items={givenStages} activeId="nel" onSelect={onSelect} />);

    // WHEN the user clicks the taxonomy item
    await userEvent.click(findRailItemByStageId("taxonomy")!);

    // THEN onSelect is invoked with the taxonomy id
    expect(onSelect).toHaveBeenCalledWith("taxonomy");
  });

  it("still calls onSelect even when the currently-active stage is clicked", async () => {
    // GIVEN the NEL stage active and an onSelect spy
    const onSelect = vi.fn();
    render(<StageRail items={givenStages} activeId="nel" onSelect={onSelect} />);

    // WHEN the user clicks the already-active item
    await userEvent.click(findRailItemByStageId("nel")!);

    // THEN onSelect is still called (the rail doesn't filter)
    expect(onSelect).toHaveBeenCalledWith("nel");
  });
});
