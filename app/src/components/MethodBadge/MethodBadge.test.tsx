import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MethodBadge, DATA_TEST_ID } from "./MethodBadge";

describe("MethodBadge", () => {
  it("renders the method name uppercased", () => {
    // GIVEN an expected HTTP method
    const givenHttpMethod = "POST";

    // WHEN we render a MethodBadge for that method
    render(<MethodBadge method={givenHttpMethod} />);

    // THEN the badge displays the given method
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toHaveTextContent(
      givenHttpMethod,
    );
  });

  it("applies the right background color per method", () => {
    // GIVEN a DELETE method badge
    // WHEN we render it
    render(<MethodBadge method="DELETE" />);

    // THEN the error background utility is present
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER).className).toMatch(/bg-error/);
  });
});
