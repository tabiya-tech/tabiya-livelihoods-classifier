import { describe, expect, it, vi } from "vitest";
import { useState } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { Slider, DATA_TEST_ID } from "./Slider";

describe("Slider", () => {
  it("renders the label and formatted value", () => {
    // GIVEN a slider label, its default value, and the formatter
    const givenSliderLabel = "min_similarity";
    const givenDefaultValue = 0.5;
    const formatToTwoDecimalPlaces = (value: number) => value.toFixed(2);
    const expectedDisplayedValue = formatToTwoDecimalPlaces(givenDefaultValue);

    // WHEN we render the Slider with that label and formatter
    render(
      <Slider
        label={givenSliderLabel}
        defaultValue={givenDefaultValue}
        min={0}
        max={1}
        step={0.05}
        format={formatToTwoDecimalPlaces}
      />,
    );

    // THEN the label and the formatted value are displayed
    expect(screen.getByTestId(DATA_TEST_ID.LABEL)).toHaveTextContent(givenSliderLabel);
    expect(screen.getByTestId(DATA_TEST_ID.VALUE)).toHaveTextContent(
      expectedDisplayedValue,
    );
  });

  it("fires onChange when the slider moves", () => {
    // GIVEN an onChange spy
    const onChange = vi.fn();

    // AND a rendered Slider bound to that spy
    render(<Slider min={0} max={10} defaultValue={3} onChange={onChange} />);

    // WHEN we move the slider to a new value
    fireEvent.change(screen.getByTestId(DATA_TEST_ID.INPUT), {
      target: { value: "5" },
    });

    // THEN onChange is called
    expect(onChange).toHaveBeenCalled();
  });

  it("updates the displayed value when the slider moves (controlled)", () => {
    // GIVEN initial and target slider values
    const givenInitialValue = 3;
    const givenTargetValue = 7;

    // AND a controlled Slider harness that reflects state into the displayed value
    function ControlledSliderHarness() {
      const [currentValue, setCurrentValue] = useState(givenInitialValue);
      return (
        <Slider
          label="top_k"
          min={1}
          max={10}
          value={currentValue}
          onChange={(event) => setCurrentValue(Number(event.target.value))}
        />
      );
    }
    render(<ControlledSliderHarness />);

    // AND the initial displayed value matches the initial state
    expect(screen.getByTestId(DATA_TEST_ID.VALUE)).toHaveTextContent(
      String(givenInitialValue),
    );

    // WHEN the user slides the input to the target value
    fireEvent.change(screen.getByTestId(DATA_TEST_ID.INPUT), {
      target: { value: String(givenTargetValue) },
    });

    // THEN the displayed value and the input value both reflect the target
    expect(screen.getByTestId(DATA_TEST_ID.VALUE)).toHaveTextContent(
      String(givenTargetValue),
    );
    expect(
      (screen.getByTestId(DATA_TEST_ID.INPUT) as HTMLInputElement).value,
    ).toBe(String(givenTargetValue));
  });
});
