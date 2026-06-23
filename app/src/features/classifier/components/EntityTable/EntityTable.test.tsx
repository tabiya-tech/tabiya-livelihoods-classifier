import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { fixtureClassifyEntities } from "@/mocks/fixtures/classify";
import { DATA_TEST_ID, EntityTable } from "./EntityTable";

describe("EntityTable", () => {
  beforeEach(() => {
    Object.assign(URL, {
      createObjectURL: vi.fn(() => "blob:test"),
      revokeObjectURL: vi.fn(),
    });
  });

  it("renders one row per entity", () => {
    // GIVEN the canonical fixture
    const expectedRowCount = fixtureClassifyEntities.length;

    // WHEN we render
    render(<EntityTable entities={fixtureClassifyEntities} />);

    // THEN one row per entity
    expect(screen.getAllByTestId(DATA_TEST_ID.ROW)).toHaveLength(
      expectedRowCount,
    );
  });

  it("renders an enabled Download button that fires a click on the anchor", async () => {
    // GIVEN the canonical fixture and stubbed anchor click
    const anchorClickSpy = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tagName: string) => {
      const element = originalCreateElement(tagName);
      if (tagName === "a") {
        Object.defineProperty(element, "click", { value: anchorClickSpy });
      }
      return element;
    });

    // WHEN we render and click Download
    render(<EntityTable entities={fixtureClassifyEntities} />);
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.DOWNLOAD_BUTTON));

    // THEN the anchor's click was called once and a blob URL was created
    expect(anchorClickSpy).toHaveBeenCalledTimes(1);
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
  });

  it("disables the Download button when there are no entities", () => {
    // GIVEN no entities
    // WHEN we render
    render(<EntityTable entities={[]} />);

    // THEN download is disabled
    expect(screen.getByTestId(DATA_TEST_ID.DOWNLOAD_BUTTON)).toBeDisabled();
  });

  it("invokes onEntityClick when a row is clicked", async () => {
    // GIVEN an onEntityClick spy
    const onEntityClick = vi.fn();

    // WHEN the user clicks the first row
    render(
      <EntityTable
        entities={fixtureClassifyEntities}
        onEntityClick={onEntityClick}
      />,
    );
    await userEvent.click(screen.getAllByTestId(DATA_TEST_ID.ROW)[0]);

    // THEN the callback fires with the first entity and index 0
    expect(onEntityClick).toHaveBeenCalledWith(fixtureClassifyEntities[0], 0);
  });

  it("invokes onEntityClick via the row Open button without double-firing", async () => {
    // GIVEN an onEntityClick spy
    const onEntityClick = vi.fn();

    // WHEN we click the Open button on the first row
    render(
      <EntityTable
        entities={fixtureClassifyEntities}
        onEntityClick={onEntityClick}
      />,
    );
    await userEvent.click(
      screen.getAllByTestId(DATA_TEST_ID.OPEN_BUTTON)[0],
    );

    // THEN exactly one click is observed (stopPropagation guards the row click)
    expect(onEntityClick).toHaveBeenCalledTimes(1);
    expect(onEntityClick).toHaveBeenCalledWith(fixtureClassifyEntities[0], 0);
  });

  it("marks the selected row via data-selected=true", () => {
    // GIVEN selectedEntityIndex pointing at row 2
    render(
      <EntityTable
        entities={fixtureClassifyEntities}
        selectedEntityIndex={2}
      />,
    );

    // THEN exactly one row carries data-selected="true"
    const renderedRows = screen.getAllByTestId(DATA_TEST_ID.ROW);
    const selectedIndices = renderedRows
      .filter((row) => row.getAttribute("data-selected") === "true")
      .map((row) => Number(row.getAttribute("data-entity-index")));
    expect(selectedIndices).toEqual([2]);
  });
});
