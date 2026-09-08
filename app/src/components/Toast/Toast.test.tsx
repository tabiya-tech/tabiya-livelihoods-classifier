import { describe, expect, it } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ToastProvider, useToast, DATA_TEST_ID } from "./ToastProvider";
import type { ToastPlacement } from "./Toast.types";

const SHORT_TOAST_DURATION_MS = 50;
const DEFAULT_TOAST_PLACEMENT: ToastPlacement = "bottom-right";

function ToastTriggerHarness({
  toastMessage,
  toastDurationMs = SHORT_TOAST_DURATION_MS,
}: {
  toastMessage: string;
  toastDurationMs?: number;
}) {
  const toast = useToast();
  return (
    <button
      onClick={() =>
        toast.show({
          message: toastMessage,
          tone: "success",
          durationMs: toastDurationMs,
        })
      }
    >
      show
    </button>
  );
}

describe("ToastProvider", () => {
  it("shows a toast when show() is invoked and auto-dismisses after the duration", async () => {
    // GIVEN an expected toast message
    const givenToastMessage = "Saved";

    // AND a tree wrapped in ToastProvider with a button that fires a short-lived toast
    render(
      <ToastProvider>
        <ToastTriggerHarness toastMessage={givenToastMessage} />
      </ToastProvider>,
    );

    // WHEN the user clicks the button
    await userEvent.click(screen.getByRole("button"));

    // THEN the toast message appears in the toast viewport
    expect(screen.getByTestId(DATA_TEST_ID.ITEM)).toHaveTextContent(givenToastMessage);

    // AND after the duration elapses, the toast is removed from the DOM
    await waitFor(
      () => expect(screen.queryByTestId(DATA_TEST_ID.ITEM)).not.toBeInTheDocument(),
      { timeout: 1000 },
    );
  });

  it("does not auto-dismiss when durationMs is 0", async () => {
    // GIVEN an expected sticky-toast message
    const givenToastMessage = "Stays";
    const waitPastDefaultMs = 150;

    // AND a tree that shows a sticky toast (durationMs=0)
    render(
      <ToastProvider>
        <ToastTriggerHarness
          toastMessage={givenToastMessage}
          toastDurationMs={0}
        />
      </ToastProvider>,
    );

    // WHEN the user shows the toast
    await userEvent.click(screen.getByRole("button"));

    // AND waits past a normal default duration
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, waitPastDefaultMs));
    });

    // THEN the toast is still visible
    expect(screen.getByTestId(DATA_TEST_ID.ITEM)).toHaveTextContent(givenToastMessage);
  });

  it.each<ToastPlacement>([
    "top-left",
    "top-center",
    "top-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
  ])("positions the viewport at %s when configured", (givenPlacement) => {
    // GIVEN a ToastProvider configured with a specific placement
    // WHEN we render it
    render(
      <ToastProvider placement={givenPlacement}>
        <div>app</div>
      </ToastProvider>,
    );

    // THEN the viewport exposes the given placement via data-placement
    expect(
      screen.getByTestId(DATA_TEST_ID.VIEWPORT).getAttribute("data-placement"),
    ).toBe(givenPlacement);
  });

  it("defaults to bottom-right when no placement is given", () => {
    // GIVEN a ToastProvider with no placement
    // WHEN we render it
    render(
      <ToastProvider>
        <div>app</div>
      </ToastProvider>,
    );

    // THEN the viewport reports the default placement
    expect(
      screen.getByTestId(DATA_TEST_ID.VIEWPORT).getAttribute("data-placement"),
    ).toBe(DEFAULT_TOAST_PLACEMENT);
  });

  it("applies the success tone color to the leading dot", async () => {
    // GIVEN a Toast harness that shows success-tone toasts
    render(
      <ToastProvider>
        <ToastTriggerHarness toastMessage="Saved" />
      </ToastProvider>,
    );

    // WHEN the user shows the toast
    await userEvent.click(screen.getByRole("button"));

    // THEN the leading dot carries the success utility
    expect(screen.getByTestId(DATA_TEST_ID.ITEM_DOT).className).toMatch(/bg-lime/);
  });
});
