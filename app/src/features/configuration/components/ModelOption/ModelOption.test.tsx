import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RADIO_CARD_DATA_TEST_ID } from "@/components";
import { ModelOption, DATA_TEST_ID } from "./ModelOption";

describe("ModelOption", () => {
  it("renders the title and description on the underlying radio card", () => {
    // GIVEN a model option with title and description
    const givenModelTitle = "MPNet base v2";
    const givenModelDescription = "Higher quality embeddings; ~2× slower.";

    // WHEN we render it
    render(
      <ModelOption
        title={givenModelTitle}
        description={givenModelDescription}
      />,
    );

    // THEN the radio card surfaces both strings
    expect(
      screen.getByTestId(RADIO_CARD_DATA_TEST_ID.TITLE),
    ).toHaveTextContent(givenModelTitle);
    expect(
      screen.getByTestId(RADIO_CARD_DATA_TEST_ID.DESCRIPTION),
    ).toHaveTextContent(givenModelDescription);
  });

  it("renders the suffix tag when a suffix string is provided", () => {
    // GIVEN a suffix string
    const givenSuffix = "768-dim";

    // WHEN we render the option
    render(<ModelOption title="x" suffix={givenSuffix} />);

    // THEN the suffix tag is in the document with that label
    expect(screen.getByTestId(DATA_TEST_ID.SUFFIX_TAG)).toHaveTextContent(
      givenSuffix,
    );
  });

  it("renders the recommended badge when recommended=true", () => {
    // GIVEN recommended=true
    // WHEN we render the option
    render(<ModelOption title="x" recommended />);

    // THEN the recommended badge is in the document
    expect(
      screen.getByTestId(DATA_TEST_ID.RECOMMENDED_BADGE),
    ).toBeInTheDocument();
  });

  it("omits both badges when neither suffix nor recommended is set", () => {
    // GIVEN a bare option with no meta affordances
    // WHEN we render it
    render(<ModelOption title="x" />);

    // THEN neither badge is rendered
    expect(screen.queryByTestId(DATA_TEST_ID.SUFFIX_TAG)).not.toBeInTheDocument();
    expect(
      screen.queryByTestId(DATA_TEST_ID.RECOMMENDED_BADGE),
    ).not.toBeInTheDocument();
  });

  it("forwards selected state to the radio card's aria-checked", () => {
    // GIVEN selected=true
    // WHEN we render the option
    render(<ModelOption title="x" selected />);

    // THEN the underlying radio card reports aria-checked=true
    expect(
      screen.getByTestId(RADIO_CARD_DATA_TEST_ID.CONTAINER).getAttribute(
        "aria-checked",
      ),
    ).toBe("true");
  });

  it("fires onClick when the option is activated", async () => {
    // GIVEN an onClick spy
    const onClick = vi.fn();
    render(<ModelOption title="x" onClick={onClick} />);

    // WHEN the user clicks the option
    await userEvent.click(screen.getByTestId(RADIO_CARD_DATA_TEST_ID.CONTAINER));

    // THEN onClick is invoked once
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("exposes the modelId as a data attribute for stable targeting", () => {
    // GIVEN a known model id
    const givenModelId = "mpnet-base-v2";

    // WHEN we render the option
    render(<ModelOption title="x" modelId={givenModelId} />);

    // THEN data-model-id surfaces on the radio card button
    expect(
      screen
        .getByTestId(RADIO_CARD_DATA_TEST_ID.CONTAINER)
        .getAttribute("data-model-id"),
    ).toBe(givenModelId);
  });
});
