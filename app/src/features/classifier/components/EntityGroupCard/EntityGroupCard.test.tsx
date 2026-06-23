import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import { fixtureClassifyEntities } from "@/mocks/fixtures/classify";
import { DATA_TEST_ID, EntityGroupCard } from "./EntityGroupCard";
import {
  DATA_TEST_ID as ENTITY_ROW_DATA_TEST_ID,
} from "../EntityRow/EntityRow";

describe("EntityGroupCard", () => {
  it("renders the localized type label and the entry count", () => {
    // GIVEN the three skill entities from the fixture
    const skillEntries = fixtureClassifyEntities
      .map((entity, entityIndex) => ({ entity, entityIndex }))
      .filter(({ entity }) => entity.entity_type === "skill");
    const expectedLabel = i18n.t("classifier.entityTypeFilter.types.skill");

    // WHEN we render
    render(
      <EntityGroupCard entityType="skill" entries={skillEntries} />,
    );

    // THEN the label and count are visible
    expect(screen.getByTestId(DATA_TEST_ID.LABEL)).toHaveTextContent(
      expectedLabel,
    );
    expect(screen.getByTestId(DATA_TEST_ID.COUNT)).toHaveTextContent(
      String(skillEntries.length),
    );
  });

  it("renders one EntityRow per entry", () => {
    // GIVEN three entries
    const entries = fixtureClassifyEntities
      .slice(0, 3)
      .map((entity, entityIndex) => ({ entity, entityIndex }));

    // WHEN we render
    render(<EntityGroupCard entityType="skill" entries={entries} />);

    // THEN three rows are rendered
    const renderedRows = screen.getAllByTestId(
      ENTITY_ROW_DATA_TEST_ID.CONTAINER,
    );
    expect(renderedRows).toHaveLength(entries.length);
  });

  it("marks the row matching selectedEntityIndex with aria-pressed", () => {
    // GIVEN two rows and the second selected
    const entries = fixtureClassifyEntities
      .slice(0, 2)
      .map((entity, entityIndex) => ({ entity, entityIndex }));

    // WHEN we render with selectedEntityIndex pointing at entity 1
    render(
      <EntityGroupCard
        entityType="skill"
        entries={entries}
        selectedEntityIndex={1}
      />,
    );

    // THEN exactly one row is aria-pressed=true
    const renderedRows = screen.getAllByTestId(
      ENTITY_ROW_DATA_TEST_ID.CONTAINER,
    );
    const pressedIndices = renderedRows
      .filter((row) => row.getAttribute("aria-pressed") === "true")
      .map((row) => Number(row.getAttribute("data-entity-index")));
    expect(pressedIndices).toEqual([1]);
  });

  it("forwards row clicks through onEntityClick with the original index", async () => {
    // GIVEN an onEntityClick spy
    const onEntityClick = vi.fn();
    const givenIndex = 7;
    const entries = [
      { entity: fixtureClassifyEntities[0], entityIndex: givenIndex },
    ];

    // WHEN the user clicks the row
    render(
      <EntityGroupCard
        entityType="occupation"
        entries={entries}
        onEntityClick={onEntityClick}
      />,
    );
    await userEvent.click(
      screen.getByTestId(ENTITY_ROW_DATA_TEST_ID.CONTAINER),
    );

    // THEN the callback fires with the entity and the supplied entityIndex
    expect(onEntityClick).toHaveBeenCalledWith(
      fixtureClassifyEntities[0],
      givenIndex,
    );
  });
});
