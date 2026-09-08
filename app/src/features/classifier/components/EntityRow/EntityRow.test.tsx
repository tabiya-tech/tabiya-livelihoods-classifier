import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import type { ClassifiedEntity } from "@/lib/api";
import { fixtureClassifyEntities } from "@/mocks/fixtures/classify";
import { DATA_TEST_ID, EntityRow } from "./EntityRow";

const givenEntity: ClassifiedEntity = fixtureClassifyEntities[0];

describe("EntityRow", () => {
  it("renders the surface form and the top match's preferred label", () => {
    // GIVEN an entity from the fixture
    const expectedSurface = givenEntity.surface_form;
    const expectedLabel = givenEntity.matches[0].entity.preferred_label;

    // WHEN we render
    render(<EntityRow entity={givenEntity} entityIndex={0} />);

    // THEN both pieces of copy are visible
    expect(screen.getByTestId(DATA_TEST_ID.SURFACE_FORM)).toHaveTextContent(
      expectedSurface,
    );
    expect(screen.getByTestId(DATA_TEST_ID.TOP_MATCH_LABEL)).toHaveTextContent(
      expectedLabel,
    );
  });

  it("formats the top score as a rounded percent", () => {
    // GIVEN the entity's top similarity_score is 0.94
    const expectedPercent = "94%";

    // WHEN we render
    render(<EntityRow entity={givenEntity} entityIndex={0} />);

    // THEN the rounded percent is shown
    expect(screen.getByTestId(DATA_TEST_ID.TOP_MATCH_SCORE)).toHaveTextContent(
      expectedPercent,
    );
  });

  it("shows the empty-match note when matches is empty", () => {
    // GIVEN an entity with no matches
    const givenEmpty: ClassifiedEntity = { ...givenEntity, matches: [] };
    const expectedNote = i18n.t("classifier.results.noMatches");

    // WHEN we render
    render(<EntityRow entity={givenEmpty} entityIndex={0} />);

    // THEN the empty note appears and the score block is absent
    expect(screen.getByTestId(DATA_TEST_ID.EMPTY_MATCH_NOTE)).toHaveTextContent(
      expectedNote,
    );
    expect(
      screen.queryByTestId(DATA_TEST_ID.TOP_MATCH_SCORE),
    ).not.toBeInTheDocument();
  });

  it("invokes onClick with the entity and its index", async () => {
    // GIVEN an onClick spy
    const onClick = vi.fn();

    // WHEN the user clicks the row
    render(
      <EntityRow entity={givenEntity} entityIndex={3} onClick={onClick} />,
    );
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CONTAINER));

    // THEN onClick fires with the entity and the original index
    expect(onClick).toHaveBeenCalledWith(givenEntity, 3);
  });

  it("reflects isSelected via aria-pressed", () => {
    // GIVEN isSelected=true
    // WHEN we render
    render(<EntityRow entity={givenEntity} entityIndex={0} isSelected />);

    // THEN the button is aria-pressed=true
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });
});
