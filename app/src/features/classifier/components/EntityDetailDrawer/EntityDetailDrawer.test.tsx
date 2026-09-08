import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import { fixtureClassifyEntities } from "@/mocks/fixtures/classify";
import type { ClassifiedEntity } from "@/lib/api";
import { DRAWER_DATA_TEST_ID } from "@/components";
import { DATA_TEST_ID, EntityDetailDrawer } from "./EntityDetailDrawer";

const givenEntity: ClassifiedEntity = fixtureClassifyEntities[0];

describe("EntityDetailDrawer", () => {
  it("does not render the panel when open is false", () => {
    // GIVEN open=false
    // WHEN we render
    render(
      <EntityDetailDrawer
        open={false}
        entity={givenEntity}
        onClose={() => {}}
      />,
    );

    // THEN the drawer panel is absent
    expect(
      screen.queryByTestId(DRAWER_DATA_TEST_ID.PANEL),
    ).not.toBeInTheDocument();
  });

  it("does not render the panel when entity is null even if open=true", () => {
    // GIVEN open=true but no entity
    // WHEN we render
    render(
      <EntityDetailDrawer open entity={null} onClose={() => {}} />,
    );

    // THEN the panel is absent
    expect(
      screen.queryByTestId(DRAWER_DATA_TEST_ID.PANEL),
    ).not.toBeInTheDocument();
  });

  it("renders the surface form as the drawer title and the span as description", () => {
    // GIVEN the canonical first entity
    const expectedSpanCopy = i18n.t("classifier.entityDetail.spanLabel", {
      start: givenEntity.span.start,
      end: givenEntity.span.end,
    });

    // WHEN we render
    render(
      <EntityDetailDrawer open entity={givenEntity} onClose={() => {}} />,
    );

    // THEN title shows the surface form and description shows the span range
    expect(screen.getByTestId(DRAWER_DATA_TEST_ID.TITLE)).toHaveTextContent(
      givenEntity.surface_form,
    );
    expect(screen.getByTestId(DATA_TEST_ID.SPAN)).toHaveTextContent(
      expectedSpanCopy,
    );
  });

  it("renders one MatchCard per match", () => {
    // GIVEN an entity with 2 matches
    const expectedMatchCount = givenEntity.matches.length;

    // WHEN we render
    render(
      <EntityDetailDrawer open entity={givenEntity} onClose={() => {}} />,
    );

    // THEN two match rows render
    expect(screen.getAllByTestId(DATA_TEST_ID.MATCH_ROW)).toHaveLength(
      expectedMatchCount,
    );
  });

  it("links each match to its taxonomy origin_uri", () => {
    // GIVEN the canonical entity (origin_uri set on every match)
    const expectedHref = givenEntity.matches[0].entity.origin_uri;

    // WHEN we render
    render(
      <EntityDetailDrawer open entity={givenEntity} onClose={() => {}} />,
    );

    // THEN the first match's link points at its origin_uri and opens in a new tab
    const firstLink = screen.getAllByTestId(DATA_TEST_ID.MATCH_LINK)[0];
    expect(firstLink).toHaveAttribute("href", expectedHref);
    expect(firstLink).toHaveAttribute("target", "_blank");
  });

  it("shows the empty note when matches is empty", () => {
    // GIVEN an entity with no matches
    const givenEmpty: ClassifiedEntity = { ...givenEntity, matches: [] };
    const expectedEmptyNote = i18n.t("classifier.results.noMatches");

    // WHEN we render
    render(
      <EntityDetailDrawer open entity={givenEmpty} onClose={() => {}} />,
    );

    // THEN the empty note is visible and no match rows render
    expect(screen.getByTestId(DATA_TEST_ID.EMPTY_NOTE)).toHaveTextContent(
      expectedEmptyNote,
    );
    expect(screen.queryAllByTestId(DATA_TEST_ID.MATCH_ROW)).toHaveLength(0);
  });

  it("invokes onClose when the drawer's backdrop is clicked", async () => {
    // GIVEN an onClose spy
    const onClose = vi.fn();

    // WHEN the user clicks the backdrop (Drawer primitive's default close trigger)
    render(
      <EntityDetailDrawer open entity={givenEntity} onClose={onClose} />,
    );
    await userEvent.click(screen.getByTestId(DRAWER_DATA_TEST_ID.BACKDROP));

    // THEN onClose fires
    expect(onClose).toHaveBeenCalled();
  });
});
