import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import { DATA_TEST_ID, RevealOnceBanner } from "./RevealOnceBanner";

describe("RevealOnceBanner", () => {
  beforeEach(() => {
    // jsdom doesn't ship a clipboard implementation — stub it per test.
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn(async () => undefined) },
    });
  });

  it("renders the plaintext key inline", () => {
    // GIVEN a plaintext key
    const givenKey = "AIzaSyTEST-KEY-VALUE";

    // WHEN we render the banner
    render(<RevealOnceBanner apiKey={givenKey} onDismiss={() => {}} />);

    // THEN the key is visible in the code element
    expect(screen.getByTestId(DATA_TEST_ID.KEY_VALUE)).toHaveTextContent(
      givenKey,
    );
  });

  it("renders the localized title and description", () => {
    // GIVEN the expected copy
    const expectedTitle = i18n.t("keys.revealBanner.title");
    const expectedDescription = i18n.t("keys.revealBanner.description");

    // WHEN we render
    render(<RevealOnceBanner apiKey="x" onDismiss={() => {}} />);

    // THEN the copy is present
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(
      expectedTitle,
    );
    expect(screen.getByTestId(DATA_TEST_ID.DESCRIPTION)).toHaveTextContent(
      expectedDescription,
    );
  });

  it("writes the key to the clipboard when Copy is clicked", async () => {
    // GIVEN a plaintext key and a clipboard spy
    const givenKey = "AIzaSyCOPY-TEST";
    const clipboardSpy = navigator.clipboard.writeText as ReturnType<
      typeof vi.fn
    >;

    // WHEN the user clicks Copy
    render(<RevealOnceBanner apiKey={givenKey} onDismiss={() => {}} />);
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.COPY_BUTTON));

    // THEN navigator.clipboard.writeText was called with the plaintext value
    expect(clipboardSpy).toHaveBeenCalledWith(givenKey);
  });

  it("invokes onDismiss when the dismiss button is clicked", async () => {
    // GIVEN an onDismiss spy
    const onDismiss = vi.fn();

    // WHEN the user clicks dismiss
    render(<RevealOnceBanner apiKey="x" onDismiss={onDismiss} />);
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.DISMISS_BUTTON));

    // THEN onDismiss fires once
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });
});
