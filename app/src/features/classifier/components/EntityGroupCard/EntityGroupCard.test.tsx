import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import { fixtureClassifyEntities } from "@/mocks/fixtures/classify";
import {
  COLLAPSED_ROW_LIMIT,
  DATA_TEST_ID,
  EntityGroupCard,
} from "./EntityGroupCard";
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

  // The fixture has only a handful of top-level entities, so synthesize a
  // longer list with distinct spans (distinct React keys) to exercise the
  // collapse threshold.
  function makeEntries(total: number) {
    const template = fixtureClassifyEntities[1]; // a "skill" entity
    return Array.from({ length: total }, (_unused, entityIndex) => ({
      entity: {
        ...template,
        surface_form: `skill ${entityIndex}`,
        span: { ...template.span, start: entityIndex * 100 },
      },
      entityIndex,
    }));
  }

  it("collapses to the row limit by default and hides the rest", () => {
    // GIVEN more entries than the collapsed limit (7 > 5)
    const givenEntries = makeEntries(7);
    const expectedVisibleRowCount = COLLAPSED_ROW_LIMIT;

    // WHEN we render without expanding
    render(<EntityGroupCard entityType="skill" entries={givenEntries} />);

    // THEN only the first COLLAPSED_ROW_LIMIT rows are shown
    const renderedRows = screen.getAllByTestId(
      ENTITY_ROW_DATA_TEST_ID.CONTAINER,
    );
    expect(renderedRows).toHaveLength(expectedVisibleRowCount);
    // AND a "show more" toggle offers the hidden ones
    expect(screen.getByTestId(DATA_TEST_ID.TOGGLE)).toHaveAttribute(
      "aria-expanded",
      "false",
    );
  });

  it("reveals all rows after clicking the toggle, then collapses again", async () => {
    // GIVEN a collapsed card with 7 entries
    const givenEntries = makeEntries(7);
    render(<EntityGroupCard entityType="skill" entries={givenEntries} />);

    // WHEN the user expands
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TOGGLE));

    // THEN every row is visible
    expect(
      screen.getAllByTestId(ENTITY_ROW_DATA_TEST_ID.CONTAINER),
    ).toHaveLength(givenEntries.length);
    expect(screen.getByTestId(DATA_TEST_ID.TOGGLE)).toHaveAttribute(
      "aria-expanded",
      "true",
    );

    // AND clicking again collapses back to the limit
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TOGGLE));
    expect(
      screen.getAllByTestId(ENTITY_ROW_DATA_TEST_ID.CONTAINER),
    ).toHaveLength(COLLAPSED_ROW_LIMIT);
  });

  it("shows no toggle when entries fit within the row limit", () => {
    // GIVEN fewer entries than the limit
    const givenEntries = fixtureClassifyEntities
      .slice(0, 3)
      .map((entity, entityIndex) => ({ entity, entityIndex }));

    // WHEN we render
    render(<EntityGroupCard entityType="skill" entries={givenEntries} />);

    // THEN no expand/collapse control appears
    expect(screen.queryByTestId(DATA_TEST_ID.TOGGLE)).toBeNull();
  });
});
