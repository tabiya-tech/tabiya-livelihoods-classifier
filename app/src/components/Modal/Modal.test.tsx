import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Modal, DATA_TEST_ID } from "./Modal";

describe("Modal", () => {
  it("does not render content when closed", () => {
    // GIVEN a Modal with open=false
    // WHEN we render it
    render(
      <Modal open={false} onClose={() => {}} title="Hidden">
        contents
      </Modal>,
    );

    // THEN no dialog is in the DOM
    expect(screen.queryByTestId(DATA_TEST_ID.DIALOG)).not.toBeInTheDocument();
  });

  it("renders title, description, children and footer when open", () => {
    // GIVEN content for each Modal slot
    const givenModalTitle = "Confirm";
    const givenModalDescription = "Are you sure?";
    const givenModalBodyText = "body";
    const givenModalFooterText = "footer content";

    // WHEN we render an open Modal with all slots populated
    render(
      <Modal
        open
        onClose={() => {}}
        title={givenModalTitle}
        description={givenModalDescription}
        footer={<span>{givenModalFooterText}</span>}
      >
        {givenModalBodyText}
      </Modal>,
    );

    // THEN each slot carries the given content inside the dialog
    expect(screen.getByTestId(DATA_TEST_ID.DIALOG)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(givenModalTitle);
    expect(screen.getByTestId(DATA_TEST_ID.DESCRIPTION)).toHaveTextContent(
      givenModalDescription,
    );
    expect(screen.getByTestId(DATA_TEST_ID.BODY)).toHaveTextContent(givenModalBodyText);
    expect(screen.getByTestId(DATA_TEST_ID.FOOTER)).toHaveTextContent(
      givenModalFooterText,
    );
  });

  it("invokes onClose when the close button is clicked", async () => {
    // GIVEN an onClose spy
    const onClose = vi.fn();

    // AND an open Modal bound to that spy
    render(
      <Modal open onClose={onClose} title="x">
        body
      </Modal>,
    );

    // WHEN the user clicks the explicit close icon button
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.CLOSE_BUTTON));

    // THEN onClose is invoked exactly once
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("invokes onClose when the user presses Escape", async () => {
    // GIVEN an onClose spy
    const onClose = vi.fn();

    // AND an open Modal bound to that spy
    render(
      <Modal open onClose={onClose} title="x">
        body
      </Modal>,
    );

    // WHEN the user presses Escape
    await userEvent.keyboard("{Escape}");

    // THEN onClose is invoked exactly once
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("invokes onClose when the user clicks the backdrop", async () => {
    // GIVEN an onClose spy
    const onClose = vi.fn();

    // AND an open Modal bound to that spy
    render(
      <Modal open onClose={onClose} title="x">
        body
      </Modal>,
    );

    // WHEN the user mousedowns directly on the backdrop (not the dialog)
    const backdrop = screen.getByTestId(DATA_TEST_ID.BACKDROP);
    await userEvent.pointer({ keys: "[MouseLeft>]", target: backdrop });

    // THEN onClose is invoked exactly once
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
