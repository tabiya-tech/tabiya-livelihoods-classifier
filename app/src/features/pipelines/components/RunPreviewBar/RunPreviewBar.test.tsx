import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { DATA_TEST_ID, RunPreviewBar } from "./RunPreviewBar";

const givenStepIntervalMs = 100;

describe("RunPreviewBar", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("advances through every stage on Play, firing onActiveStageChange for each index", () => {
    // GIVEN a pipeline with three stages and a spy handler
    const givenStageCount = 3;
    const givenOnActiveStageChange = vi.fn();
    render(
      <RunPreviewBar
        stageCount={givenStageCount}
        onActiveStageChange={givenOnActiveStageChange}
        stepIntervalMs={givenStepIntervalMs}
      />,
    );

    // WHEN we click Play and let two step intervals elapse
    fireEvent.click(screen.getByTestId(DATA_TEST_ID.PLAY_BUTTON));

    // THEN the first invocation was for stage index 0 (fired synchronously)
    const expectedFirstCallIndex = 0;
    expect(givenOnActiveStageChange).toHaveBeenNthCalledWith(
      1,
      expectedFirstCallIndex,
    );

    // AND after the first interval we advance to index 1
    act(() => {
      vi.advanceTimersByTime(givenStepIntervalMs);
    });
    const expectedSecondCallIndex = 1;
    expect(givenOnActiveStageChange).toHaveBeenNthCalledWith(
      2,
      expectedSecondCallIndex,
    );

    // AND after the second interval we advance to index 2 (last stage)
    act(() => {
      vi.advanceTimersByTime(givenStepIntervalMs);
    });
    const expectedThirdCallIndex = 2;
    expect(givenOnActiveStageChange).toHaveBeenNthCalledWith(
      3,
      expectedThirdCallIndex,
    );
  });

  it("clears state and fires onActiveStageChange(null) when Reset is clicked", () => {
    // GIVEN a running preview over two stages
    const givenStageCount = 2;
    const givenOnActiveStageChange = vi.fn();
    render(
      <RunPreviewBar
        stageCount={givenStageCount}
        onActiveStageChange={givenOnActiveStageChange}
        stepIntervalMs={givenStepIntervalMs}
      />,
    );
    fireEvent.click(screen.getByTestId(DATA_TEST_ID.PLAY_BUTTON));
    act(() => {
      vi.advanceTimersByTime(givenStepIntervalMs);
    });

    // WHEN we click Reset
    givenOnActiveStageChange.mockClear();
    fireEvent.click(screen.getByTestId(DATA_TEST_ID.RESET_BUTTON));

    // THEN onActiveStageChange fires with null AND Play re-enables
    const expectedResetPayload = null;
    expect(givenOnActiveStageChange).toHaveBeenCalledWith(expectedResetPayload);
    expect(screen.getByTestId(DATA_TEST_ID.PLAY_BUTTON)).not.toBeDisabled();
  });

  it("disables Play when stageCount is zero", () => {
    // GIVEN a pipeline with no stages
    const givenStageCount = 0;
    const givenOnActiveStageChange = vi.fn();

    // WHEN we render
    render(
      <RunPreviewBar
        stageCount={givenStageCount}
        onActiveStageChange={givenOnActiveStageChange}
      />,
    );

    // THEN the Play button is disabled
    expect(screen.getByTestId(DATA_TEST_ID.PLAY_BUTTON)).toBeDisabled();
  });

  it("never invokes fetch across the entire animation — 'no backend' contract", () => {
    // GIVEN a spy on the global fetch and a pipeline with three stages
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const givenStageCount = 3;
    const givenOnActiveStageChange = vi.fn();

    // WHEN we render, click Play, and let the animation complete
    render(
      <RunPreviewBar
        stageCount={givenStageCount}
        onActiveStageChange={givenOnActiveStageChange}
        stepIntervalMs={givenStepIntervalMs}
      />,
    );
    fireEvent.click(screen.getByTestId(DATA_TEST_ID.PLAY_BUTTON));
    act(() => {
      // Advance well past the end of the animation.
      vi.advanceTimersByTime(givenStepIntervalMs * (givenStageCount + 2));
    });

    // THEN fetch was never called — preview is pure UI.
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("disables Play while an animation is in progress", () => {
    // GIVEN a pipeline mid-animation
    const givenStageCount = 3;
    const givenOnActiveStageChange = vi.fn();
    render(
      <RunPreviewBar
        stageCount={givenStageCount}
        onActiveStageChange={givenOnActiveStageChange}
        stepIntervalMs={givenStepIntervalMs}
      />,
    );

    // WHEN we click Play (animation begins)
    fireEvent.click(screen.getByTestId(DATA_TEST_ID.PLAY_BUTTON));

    // THEN the Play button is disabled while playing
    expect(screen.getByTestId(DATA_TEST_ID.PLAY_BUTTON)).toBeDisabled();
  });
});
