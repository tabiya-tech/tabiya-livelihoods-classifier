import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Select, DATA_TEST_ID } from "./Select";

describe("Select", () => {
  it("renders option children", () => {
    // GIVEN a set of option labels
    const givenFirstOptionLabel = "A";
    const givenSecondOptionLabel = "B";

    // WHEN we render a Select with those options
    render(
      <Select>
        <option value="a">{givenFirstOptionLabel}</option>
        <option value="b">{givenSecondOptionLabel}</option>
      </Select>,
    );

    // THEN the select container is in the DOM and the first option is reachable
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: givenFirstOptionLabel }),
    ).toBeInTheDocument();
  });

  it("fires onChange when a new option is picked", async () => {
    // GIVEN an onChange spy and an option to switch to
    const onChange = vi.fn();
    const givenInitialOptionValue = "a";
    const givenNextOptionValue = "b";

    // AND a Select bound to that spy starting at the initial option
    render(
      <Select defaultValue={givenInitialOptionValue} onChange={onChange}>
        <option value="a">A</option>
        <option value="b">B</option>
      </Select>,
    );

    // WHEN the user selects the next option
    await userEvent.selectOptions(
      screen.getByTestId(DATA_TEST_ID.CONTAINER),
      givenNextOptionValue,
    );

    // THEN onChange is called and the select's value reflects the choice
    expect(onChange).toHaveBeenCalled();
    expect(
      (screen.getByTestId(DATA_TEST_ID.CONTAINER) as HTMLSelectElement).value,
    ).toBe(givenNextOptionValue);
  });
});
