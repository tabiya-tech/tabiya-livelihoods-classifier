import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { SLIDER_DATA_TEST_ID } from "@/components";
import {
  DATA_TEST_ID,
  PipelineParamsCard,
} from "./PipelineParamsCard";

// The Slider primitive spreads {...rest} onto its inner <input>, so the
// data-testid we pass at the call site lands on the <input> itself.
function getSliderInput(sliderTestId: string): HTMLInputElement {
  return screen.getByTestId(sliderTestId) as HTMLInputElement;
}

describe("PipelineParamsCard", () => {
  it("renders both sliders seeded with the supplied values", () => {
    // GIVEN the params seeded with top_k=3 and min_similarity=0.5
    const givenTopK = 3;
    const givenMinSimilarity = 0.5;

    // WHEN we render
    render(
      <PipelineParamsCard
        topK={givenTopK}
        minSimilarity={givenMinSimilarity}
        onTopKChange={() => {}}
        onMinSimilarityChange={() => {}}
      />,
    );

    // THEN both sliders are present and the formatted values are visible
    const renderedValues = screen.getAllByTestId(SLIDER_DATA_TEST_ID.VALUE);
    expect(renderedValues[0]).toHaveTextContent(String(givenTopK));
    expect(renderedValues[1]).toHaveTextContent(givenMinSimilarity.toFixed(2));
  });

  it("invokes onTopKChange with the new number when the top_k slider moves", () => {
    // GIVEN an onTopKChange spy
    const onTopKChange = vi.fn();
    const givenNewValue = 8;

    // WHEN the user drags the top_k slider
    render(
      <PipelineParamsCard
        topK={5}
        minSimilarity={0}
        onTopKChange={onTopKChange}
        onMinSimilarityChange={() => {}}
      />,
    );
    const topKInput = getSliderInput(DATA_TEST_ID.TOP_K_SLIDER);
    fireEvent.change(topKInput, { target: { value: String(givenNewValue) } });

    // THEN the callback fires with the parsed number
    expect(onTopKChange).toHaveBeenCalledWith(givenNewValue);
  });

  it("invokes onMinSimilarityChange with the new number when that slider moves", () => {
    // GIVEN an onMinSimilarityChange spy
    const onMinSimilarityChange = vi.fn();
    const givenNewValue = 0.75;

    // WHEN the user drags the min_similarity slider
    render(
      <PipelineParamsCard
        topK={5}
        minSimilarity={0}
        onTopKChange={() => {}}
        onMinSimilarityChange={onMinSimilarityChange}
      />,
    );
    const minSimInput = getSliderInput(DATA_TEST_ID.MIN_SIMILARITY_SLIDER);
    fireEvent.change(minSimInput, {
      target: { value: String(givenNewValue) },
    });

    // THEN the callback fires with the parsed number
    expect(onMinSimilarityChange).toHaveBeenCalledWith(givenNewValue);
  });

  it("disables both sliders when disabled is true", () => {
    // GIVEN the card in the disabled state
    // WHEN we render
    render(
      <PipelineParamsCard
        topK={5}
        minSimilarity={0}
        onTopKChange={() => {}}
        onMinSimilarityChange={() => {}}
        disabled
      />,
    );

    // THEN both underlying inputs are disabled
    expect(getSliderInput(DATA_TEST_ID.TOP_K_SLIDER)).toBeDisabled();
    expect(getSliderInput(DATA_TEST_ID.MIN_SIMILARITY_SLIDER)).toBeDisabled();
  });
});
