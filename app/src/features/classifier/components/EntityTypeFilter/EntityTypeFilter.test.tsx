import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ClassifyEntityType } from "@/lib/api";
import { DATA_TEST_ID, EntityTypeFilter } from "./EntityTypeFilter";

const givenCounts: Record<ClassifyEntityType, number> = {
  occupation: 1,
  skill: 3,
  qualification: 1,
};

describe("EntityTypeFilter", () => {
  it("renders one chip per entity type with its count", () => {
    // GIVEN counts across all three entity types
    const expectedChipCount = 3;

    // WHEN we render
    render(
      <EntityTypeFilter
        selected={new Set(["occupation", "skill", "qualification"])}
        counts={givenCounts}
        onChange={() => {}}
      />,
    );

    // THEN three chips appear and each shows the corresponding count
    const renderedChips = screen.getAllByTestId(DATA_TEST_ID.CHIP);
    expect(renderedChips).toHaveLength(expectedChipCount);
    expect(renderedChips[0]).toHaveTextContent(String(givenCounts.occupation));
    expect(renderedChips[1]).toHaveTextContent(String(givenCounts.skill));
    expect(renderedChips[2]).toHaveTextContent(String(givenCounts.qualification));
  });

  it("marks chips matching the selected set with aria-checked=true", () => {
    // GIVEN only the skill type is selected
    const givenSelected = new Set<ClassifyEntityType>(["skill"]);

    // WHEN we render
    render(
      <EntityTypeFilter
        selected={givenSelected}
        counts={givenCounts}
        onChange={() => {}}
      />,
    );

    // THEN exactly the skill chip is aria-checked=true
    const renderedChips = screen.getAllByTestId(DATA_TEST_ID.CHIP);
    const checkedTypes = renderedChips
      .filter((chip) => chip.getAttribute("aria-checked") === "true")
      .map((chip) => chip.getAttribute("data-entity-type"));
    expect(checkedTypes).toEqual(["skill"]);
  });

  it("adds a type to the selection on click when it was previously deselected", async () => {
    // GIVEN nothing selected and an onChange spy
    const onChange = vi.fn();

    // WHEN the user clicks the occupation chip
    render(
      <EntityTypeFilter
        selected={new Set()}
        counts={givenCounts}
        onChange={onChange}
      />,
    );
    const occupationChip = screen
      .getAllByTestId(DATA_TEST_ID.CHIP)
      .find((chip) => chip.getAttribute("data-entity-type") === "occupation")!;
    await userEvent.click(occupationChip);

    // THEN onChange fires with a set containing only occupation
    expect(onChange).toHaveBeenCalledTimes(1);
    const nextSelection = onChange.mock.calls[0][0] as Set<ClassifyEntityType>;
    expect(Array.from(nextSelection)).toEqual(["occupation"]);
  });

  it("removes a type from the selection on click when it was previously selected", async () => {
    // GIVEN all three types selected
    const onChange = vi.fn();

    // WHEN the user clicks the skill chip (deselect)
    render(
      <EntityTypeFilter
        selected={new Set(["occupation", "skill", "qualification"])}
        counts={givenCounts}
        onChange={onChange}
      />,
    );
    const skillChip = screen
      .getAllByTestId(DATA_TEST_ID.CHIP)
      .find((chip) => chip.getAttribute("data-entity-type") === "skill")!;
    await userEvent.click(skillChip);

    // THEN onChange fires with the same set minus skill
    const nextSelection = onChange.mock.calls[0][0] as Set<ClassifyEntityType>;
    expect(Array.from(nextSelection).sort()).toEqual([
      "occupation",
      "qualification",
    ]);
  });

  it("disables every chip when disabled is true", () => {
    // GIVEN the filter in the disabled state
    // WHEN we render
    render(
      <EntityTypeFilter
        selected={new Set(["skill"])}
        counts={givenCounts}
        onChange={() => {}}
        disabled
      />,
    );

    // THEN every chip is disabled
    const renderedChips = screen.getAllByTestId(DATA_TEST_ID.CHIP);
    renderedChips.forEach((chip) => expect(chip).toBeDisabled());
  });
});
