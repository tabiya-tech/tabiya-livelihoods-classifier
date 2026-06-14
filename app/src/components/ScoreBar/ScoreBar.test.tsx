import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ScoreBar, DATA_TEST_ID } from "./ScoreBar";

describe("ScoreBar", () => {
  it("exposes the score on the progressbar role", () => {
    // GIVEN a normalized score in [0, 1]
    const givenScore = 0.42;
    const expectedAriaValueNow = String(givenScore);

    // WHEN we render a ScoreBar with that score
    render(<ScoreBar score={givenScore} />);

    // THEN the progressbar reports aria-valuenow equal to the given score
    expect(
      screen.getByTestId(DATA_TEST_ID.CONTAINER).getAttribute("aria-valuenow"),
    ).toBe(expectedAriaValueNow);
  });

  it("clamps scores above 1", () => {
    // GIVEN a score above the upper bound and the bound itself
    const givenScoreAboveOne = 1.5;
    const expectedClampedValue = "1";

    // WHEN we render the ScoreBar with that score
    render(<ScoreBar score={givenScoreAboveOne} />);

    // THEN aria-valuenow is clamped to the upper bound
    expect(
      screen.getByTestId(DATA_TEST_ID.CONTAINER).getAttribute("aria-valuenow"),
    ).toBe(expectedClampedValue);
  });

  it("clamps scores below 0", () => {
    // GIVEN a negative score and the lower bound
    const givenNegativeScore = -0.3;
    const expectedClampedValue = "0";

    // WHEN we render the ScoreBar with that score
    render(<ScoreBar score={givenNegativeScore} />);

    // THEN aria-valuenow is clamped to the lower bound
    expect(
      screen.getByTestId(DATA_TEST_ID.CONTAINER).getAttribute("aria-valuenow"),
    ).toBe(expectedClampedValue);
  });

  it("colors the fill with the entity-skill utility when given that type", () => {
    // GIVEN a ScoreBar typed as 'skill'
    // WHEN we render it
    render(<ScoreBar score={0.7} entityType="skill" />);

    // THEN the inner fill carries the skill foreground utility
    expect(screen.getByTestId(DATA_TEST_ID.FILL).className).toMatch(
      /bg-entity-skill-fg/,
    );
  });
});
