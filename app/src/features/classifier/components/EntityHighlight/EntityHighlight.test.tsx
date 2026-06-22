import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  fixtureClassifyEntities,
  fixtureClassifySourceText,
} from "@/mocks/fixtures/classify";
import { DATA_TEST_ID, EntityHighlight } from "./EntityHighlight";

describe("EntityHighlight", () => {
  it("renders one entity span per kept entity", () => {
    // GIVEN the canonical fixture with 5 non-overlapping entities
    const expectedEntityCount = fixtureClassifyEntities.length;

    // WHEN we render
    render(
      <EntityHighlight
        text={fixtureClassifySourceText}
        entities={fixtureClassifyEntities}
      />,
    );

    // THEN the number of entity spans matches the fixture
    const renderedEntitySpans = screen.getAllByTestId(
      DATA_TEST_ID.ENTITY_SEGMENT,
    );
    expect(renderedEntitySpans).toHaveLength(expectedEntityCount);
  });

  it("tags each entity span with its entity_type via data-type", () => {
    // GIVEN the fixture (1 occupation, 3 skills, 1 qualification)
    // WHEN we render
    render(
      <EntityHighlight
        text={fixtureClassifySourceText}
        entities={fixtureClassifyEntities}
      />,
    );

    // THEN each entity span carries the matching data-type attribute
    const renderedEntitySpans = screen.getAllByTestId(
      DATA_TEST_ID.ENTITY_SEGMENT,
    );
    const renderedTypes = renderedEntitySpans.map((span) =>
      span.getAttribute("data-type"),
    );
    const expectedTypes = fixtureClassifyEntities.map((entity) => entity.entity_type);
    expect(renderedTypes).toEqual(expectedTypes);
  });

  it("invokes onEntityClick with the entity and its original index", async () => {
    // GIVEN an onEntityClick spy
    const onEntityClick = vi.fn();

    // WHEN we render and click the first entity span
    render(
      <EntityHighlight
        text={fixtureClassifySourceText}
        entities={fixtureClassifyEntities}
        onEntityClick={onEntityClick}
      />,
    );
    const firstSpan = screen.getAllByTestId(DATA_TEST_ID.ENTITY_SEGMENT)[0];
    await userEvent.click(firstSpan);

    // THEN the callback fires with the matching entity + index 0
    expect(onEntityClick).toHaveBeenCalledTimes(1);
    expect(onEntityClick).toHaveBeenCalledWith(fixtureClassifyEntities[0], 0);
  });

  it("applies the 'selected' class to the span matching selectedEntityIndex", () => {
    // GIVEN selectedEntityIndex pointing at the second entity (index 1)
    const givenSelectedIndex = 1;

    // WHEN we render
    render(
      <EntityHighlight
        text={fixtureClassifySourceText}
        entities={fixtureClassifyEntities}
        selectedEntityIndex={givenSelectedIndex}
      />,
    );

    // THEN exactly one span has the selected class — the one with that index
    const renderedEntitySpans = screen.getAllByTestId(
      DATA_TEST_ID.ENTITY_SEGMENT,
    );
    const selectedSpans = renderedEntitySpans.filter((span) =>
      span.classList.contains("selected"),
    );
    expect(selectedSpans).toHaveLength(1);
    expect(selectedSpans[0].getAttribute("data-entity-index")).toBe(
      String(givenSelectedIndex),
    );
  });

  it("applies the 'dimmed' class only to spans in dimmedEntityIndices", () => {
    // GIVEN a dimmed set covering entity indices 0 and 2
    const givenDimmed = new Set([0, 2]);

    // WHEN we render
    render(
      <EntityHighlight
        text={fixtureClassifySourceText}
        entities={fixtureClassifyEntities}
        dimmedEntityIndices={givenDimmed}
      />,
    );

    // THEN exactly those spans carry the dimmed class
    const renderedEntitySpans = screen.getAllByTestId(
      DATA_TEST_ID.ENTITY_SEGMENT,
    );
    const dimmedIndices = renderedEntitySpans
      .filter((span) => span.classList.contains("dimmed"))
      .map((span) => Number(span.getAttribute("data-entity-index")));
    expect(new Set(dimmedIndices)).toEqual(givenDimmed);
  });

  it("fires onEntityClick on Enter keypress for keyboard users", async () => {
    // GIVEN an onEntityClick spy
    const onEntityClick = vi.fn();

    // WHEN we focus the first entity span and press Enter
    render(
      <EntityHighlight
        text={fixtureClassifySourceText}
        entities={fixtureClassifyEntities}
        onEntityClick={onEntityClick}
      />,
    );
    const firstSpan = screen.getAllByTestId(DATA_TEST_ID.ENTITY_SEGMENT)[0];
    firstSpan.focus();
    await userEvent.keyboard("{Enter}");

    // THEN the callback fires
    expect(onEntityClick).toHaveBeenCalledTimes(1);
  });
});
