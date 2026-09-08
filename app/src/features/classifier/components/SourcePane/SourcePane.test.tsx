import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  fixtureClassifyEntities,
  fixtureClassifySourceText,
} from "@/mocks/fixtures/classify";
import type { ClassifyEntityType } from "@/lib/api";
import { ENTITY_TYPES, DATA_TEST_ID, SourcePane } from "./SourcePane";
import {
  DATA_TEST_ID as HIGHLIGHT_DATA_TEST_ID,
} from "../EntityHighlight/EntityHighlight";

function renderPane(overrides: Partial<React.ComponentProps<typeof SourcePane>> = {}) {
  const baseProps: React.ComponentProps<typeof SourcePane> = {
    text: "",
    onTextChange: () => {},
    entities: null,
    selectedEntityTypes: new Set<ClassifyEntityType>(ENTITY_TYPES),
    onSelectedEntityTypesChange: () => {},
    topK: 5,
    minSimilarity: 0,
    onTopKChange: () => {},
    onMinSimilarityChange: () => {},
    isRunning: false,
    canRun: false,
    onRun: () => {},
    onClear: () => {},
  };
  return render(<SourcePane {...baseProps} {...overrides} />);
}

describe("SourcePane", () => {
  it("shows the textarea when no entities have been classified yet", () => {
    // GIVEN no prior results
    // WHEN we render
    renderPane();

    // THEN the textarea is visible and the highlight is not
    expect(screen.getByTestId(DATA_TEST_ID.TEXTAREA)).toBeInTheDocument();
    expect(screen.queryByTestId(DATA_TEST_ID.HIGHLIGHT)).not.toBeInTheDocument();
  });

  it("shows the highlight view once entities are present", () => {
    // GIVEN entities from the canonical fixture
    // WHEN we render with results
    renderPane({
      text: fixtureClassifySourceText,
      entities: fixtureClassifyEntities,
    });

    // THEN the highlight is visible and the textarea is hidden
    expect(screen.getByTestId(DATA_TEST_ID.HIGHLIGHT)).toBeInTheDocument();
    expect(screen.queryByTestId(DATA_TEST_ID.TEXTAREA)).not.toBeInTheDocument();
  });

  it("forwards entity clicks back to the page through onEntitySelect", async () => {
    // GIVEN a prior run with results and an onEntitySelect spy
    const onEntitySelect = vi.fn();
    renderPane({
      text: fixtureClassifySourceText,
      entities: fixtureClassifyEntities,
      onEntitySelect,
    });

    // AND the highlight is the visible source view
    expect(
      screen.getByTestId(HIGHLIGHT_DATA_TEST_ID.CONTAINER),
    ).toBeInTheDocument();

    // WHEN the user clicks the first entity span
    await userEvent.click(
      screen.getAllByTestId(HIGHLIGHT_DATA_TEST_ID.ENTITY_SEGMENT)[0],
    );

    // THEN onEntitySelect fires with that entity's index
    expect(onEntitySelect).toHaveBeenCalledWith(0);
  });

  it("invokes onRun and onClear via their buttons", async () => {
    // GIVEN onRun + onClear spies and text already entered
    const onRun = vi.fn();
    const onClear = vi.fn();

    // WHEN the user clicks Run, then Clear
    renderPane({
      text: "some job ad",
      canRun: true,
      onRun,
      onClear,
    });
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.RUN_BUTTON));
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CLEAR_BUTTON));

    // THEN both spies fire
    expect(onRun).toHaveBeenCalledTimes(1);
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it("disables the run button until canRun is true", () => {
    // GIVEN canRun=false
    // WHEN we render
    renderPane({ text: "", canRun: false });

    // THEN the Run button is disabled
    expect(screen.getByTestId(DATA_TEST_ID.RUN_BUTTON)).toBeDisabled();
  });

  it("computes type counts and passes them to the filter chips", () => {
    // GIVEN the canonical fixture (1 occupation, 3 skills, 1 qualification)
    const expectedOccupation = 1;
    const expectedSkill = 3;
    const expectedQualification = 1;

    // WHEN we render with results
    renderPane({
      text: fixtureClassifySourceText,
      entities: fixtureClassifyEntities,
    });

    // THEN each chip displays its computed count
    const chips = screen.getAllByRole("switch");
    expect(chips[0]).toHaveTextContent(String(expectedOccupation));
    expect(chips[1]).toHaveTextContent(String(expectedSkill));
    expect(chips[2]).toHaveTextContent(String(expectedQualification));
  });

  it("hides the entity-type filter when there are no results", () => {
    // GIVEN no results
    // WHEN we render
    renderPane();

    // THEN no chips are present
    expect(screen.queryAllByRole("switch")).toHaveLength(0);
  });

  it("disables Run and inputs while a request is in flight", () => {
    // GIVEN isRunning=true
    // WHEN we render
    renderPane({ isRunning: true, canRun: true, text: "x" });

    // THEN Run is disabled, Clear is disabled, textarea is disabled
    expect(screen.getByTestId(DATA_TEST_ID.RUN_BUTTON)).toBeDisabled();
    expect(screen.getByTestId(DATA_TEST_ID.CLEAR_BUTTON)).toBeDisabled();
    expect(screen.getByTestId(DATA_TEST_ID.TEXTAREA)).toBeDisabled();
  });

  it("renders the activeConfigSlot when provided", () => {
    // GIVEN a custom slot
    const givenSlotText = "MiniLM · ESCO v1.2";

    // WHEN we render with the slot
    renderPane({
      text: "x",
      activeConfigSlot: <span>{givenSlotText}</span>,
    });

    // THEN the slot content appears
    expect(screen.getByTestId(DATA_TEST_ID.CONFIG_CHIP)).toHaveTextContent(
      givenSlotText,
    );
  });
});
