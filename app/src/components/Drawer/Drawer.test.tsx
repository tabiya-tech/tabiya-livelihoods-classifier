import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Drawer, DATA_TEST_ID } from "./Drawer";

describe("Drawer", () => {
  it("does not render when closed", () => {
    // GIVEN a Drawer with open=false
    // WHEN we render it
    render(
      <Drawer open={false} onClose={() => {}} title="Hidden">
        body
      </Drawer>,
    );

    // THEN no panel is in the DOM
    expect(screen.queryByTestId(DATA_TEST_ID.PANEL)).not.toBeInTheDocument();
  });

  it("renders eyebrow, title, body and footer when open", () => {
    // GIVEN content for each slot
    const givenEyebrowText = "OCCUPATION";
    const givenTitleText = "Senior Engineer";
    const givenBodyText = "details body";
    const givenFooterText = "footer-actions";

    // WHEN we render the open Drawer with all four slots populated
    render(
      <Drawer
        open
        onClose={() => {}}
        eyebrow={givenEyebrowText}
        title={givenTitleText}
        footer={<span>{givenFooterText}</span>}
      >
        {givenBodyText}
      </Drawer>,
    );

    // THEN each slot's node carries the given text
    expect(screen.getByTestId(DATA_TEST_ID.EYEBROW)).toHaveTextContent(givenEyebrowText);
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(givenTitleText);
    expect(screen.getByTestId(DATA_TEST_ID.BODY)).toHaveTextContent(givenBodyText);
    expect(screen.getByTestId(DATA_TEST_ID.FOOTER)).toHaveTextContent(givenFooterText);
  });

  it("invokes onClose on Escape", async () => {
    // GIVEN an onClose spy
    const onClose = vi.fn();

    // AND an open Drawer wired to that spy
    render(
      <Drawer open onClose={onClose} title="x">
        body
      </Drawer>,
    );

    // WHEN the user presses Escape
    await userEvent.keyboard("{Escape}");

    // THEN onClose is invoked exactly once
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("invokes onClose when the backdrop is clicked", async () => {
    // GIVEN an onClose spy
    const onClose = vi.fn();

    // AND an open Drawer wired to that spy
    render(
      <Drawer open onClose={onClose} title="x">
        body
      </Drawer>,
    );

    // WHEN the user mousedowns on the backdrop directly
    const backdrop = screen.getByTestId(DATA_TEST_ID.BACKDROP);
    await userEvent.pointer({ keys: "[MouseLeft>]", target: backdrop });

    // THEN onClose is invoked exactly once
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not invoke onClose when closeOnBackdropClick is false", async () => {
    // GIVEN an onClose spy
    const onClose = vi.fn();

    // AND a Drawer that ignores backdrop clicks
    render(
      <Drawer open onClose={onClose} title="x" closeOnBackdropClick={false}>
        body
      </Drawer>,
    );

    // WHEN the user mousedowns on the backdrop
    const backdrop = screen.getByTestId(DATA_TEST_ID.BACKDROP);
    await userEvent.pointer({ keys: "[MouseLeft>]", target: backdrop });

    // THEN onClose stays untouched
    expect(onClose).not.toHaveBeenCalled();
  });
});
