import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { fixtureClassifyResponse } from "@/mocks/fixtures/classify";
import type { ClassifyEntityType } from "@/lib/api";
import { ENTITY_TYPES } from "../EntityTypeFilter/EntityTypeFilter";
import {
  DATA_TEST_ID,
  ResultsTabs,
  type ResultsTabId,
} from "./ResultsTabs";
import {
  DATA_TEST_ID as ENTITY_GROUP_DATA_TEST_ID,
} from "../EntityGroupCard/EntityGroupCard";
import {
  DATA_TEST_ID as ENTITY_TABLE_DATA_TEST_ID,
} from "../EntityTable/EntityTable";

const allSelected = new Set<ClassifyEntityType>(ENTITY_TYPES);

describe("ResultsTabs", () => {
  it("renders one EntityGroupCard per visible entity type with non-zero entries", () => {
    // GIVEN the canonical response and all types selected (1 occ, 3 skill, 1 qual)
    const expectedGroupCount = 3;

    // WHEN we render on the entities tab
    render(
      <ResultsTabs
        response={fixtureClassifyResponse}
        selectedEntityTypes={allSelected}
        activeTabId="entities"
        onActiveTabChange={() => {}}
      />,
    );

    // THEN three group cards appear
    expect(
      screen.getAllByTestId(ENTITY_GROUP_DATA_TEST_ID.CONTAINER),
    ).toHaveLength(expectedGroupCount);
  });

  it("filters out hidden entity types from the entities tab", () => {
    // GIVEN only skills selected (3 entries)
    const givenSelected = new Set<ClassifyEntityType>(["skill"]);

    // WHEN we render on the entities tab
    render(
      <ResultsTabs
        response={fixtureClassifyResponse}
        selectedEntityTypes={givenSelected}
        activeTabId="entities"
        onActiveTabChange={() => {}}
      />,
    );

    // THEN only one group card appears, for skills
    const groups = screen.getAllByTestId(ENTITY_GROUP_DATA_TEST_ID.CONTAINER);
    expect(groups).toHaveLength(1);
    expect(groups[0]).toHaveAttribute("data-entity-type", "skill");
  });

  it("shows the empty state when no visible entities remain", () => {
    // GIVEN no entity types selected
    const givenSelected = new Set<ClassifyEntityType>();

    // WHEN we render on the entities tab
    render(
      <ResultsTabs
        response={fixtureClassifyResponse}
        selectedEntityTypes={givenSelected}
        activeTabId="entities"
        onActiveTabChange={() => {}}
      />,
    );

    // THEN the empty-state block is visible and no group cards exist
    expect(screen.getByTestId(DATA_TEST_ID.EMPTY_STATE)).toBeInTheDocument();
    expect(
      screen.queryAllByTestId(ENTITY_GROUP_DATA_TEST_ID.CONTAINER),
    ).toHaveLength(0);
  });

  it("renders the table panel and respects the filter on activeTabId='table'", () => {
    // GIVEN only skills selected (3 entries)
    const givenSelected = new Set<ClassifyEntityType>(["skill"]);

    // WHEN we render the table tab
    render(
      <ResultsTabs
        response={fixtureClassifyResponse}
        selectedEntityTypes={givenSelected}
        activeTabId="table"
        onActiveTabChange={() => {}}
      />,
    );

    // THEN the table panel is the visible one and shows three skill rows
    expect(screen.getByTestId(DATA_TEST_ID.PANEL_TABLE)).toBeInTheDocument();
    expect(
      screen.getAllByTestId(ENTITY_TABLE_DATA_TEST_ID.ROW),
    ).toHaveLength(3);
  });

  it("invokes onActiveTabChange when a tab is clicked", async () => {
    // GIVEN a spy
    const onActiveTabChange = vi.fn();

    // WHEN we click the JSON tab
    render(
      <ResultsTabs
        response={fixtureClassifyResponse}
        selectedEntityTypes={allSelected}
        activeTabId="entities"
        onActiveTabChange={onActiveTabChange}
      />,
    );
    const jsonTab = screen
      .getAllByRole("tab")
      .find((tab) => tab.getAttribute("data-tab-id") === "json")!;
    await userEvent.click(jsonTab);

    // THEN the spy fires with "json"
    const argument: ResultsTabId = "json";
    expect(onActiveTabChange).toHaveBeenCalledWith(argument);
  });

  it("renders the JSON panel on activeTabId='json'", () => {
    // GIVEN no filtering
    // WHEN we render the JSON tab
    render(
      <ResultsTabs
        response={fixtureClassifyResponse}
        selectedEntityTypes={allSelected}
        activeTabId="json"
        onActiveTabChange={() => {}}
      />,
    );

    // THEN the JSON panel is the visible one
    expect(screen.getByTestId(DATA_TEST_ID.PANEL_JSON)).toBeInTheDocument();
  });
});
