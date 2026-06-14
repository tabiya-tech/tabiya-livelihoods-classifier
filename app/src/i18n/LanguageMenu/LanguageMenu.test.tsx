import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// Stub the SupportedLocales constant so the test can drive the multi-locale
// branch even though production currently ships only one. The Locale enum
// and LocalesLabels stay as their real values so option rendering still works.
const mockSupportedLocales = vi.hoisted(() => ({
  current: [] as string[],
}));
vi.mock("@/i18n/constants", () => {
  return {
    Locale: { EN_US: "en-US", FR_FR: "fr-FR" },
    LocalesLabels: {
      "en-US": "English (US)",
      "fr-FR": "Français (France)",
    },
    FALL_BACK_LOCALE: "en-US",
    get SupportedLocales() {
      return mockSupportedLocales.current;
    },
  };
});

// Spy on i18next via react-i18next's useTranslation.
const mockChangeLanguage = vi.fn();
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: {
      language: "en-US",
      changeLanguage: mockChangeLanguage,
    },
  }),
}));

import { LanguageMenu, DATA_TEST_ID } from "./LanguageMenu";

beforeEach(() => {
  mockChangeLanguage.mockReset().mockResolvedValue(undefined);
});

describe("LanguageMenu", () => {
  it("renders nothing when only one locale is supported", () => {
    // GIVEN a single-locale environment
    mockSupportedLocales.current = ["en-US"];

    // WHEN we render the menu
    render(<LanguageMenu />);

    // THEN no part of the menu is in the DOM
    expect(screen.queryByTestId(DATA_TEST_ID.CONTAINER)).not.toBeInTheDocument();
    expect(screen.queryByTestId(DATA_TEST_ID.TRIGGER)).not.toBeInTheDocument();
  });

  it("renders only the trigger when multiple locales are supported", () => {
    // GIVEN two supported locales
    mockSupportedLocales.current = ["en-US", "fr-FR"];

    // WHEN we render the menu
    render(<LanguageMenu />);

    // THEN the trigger is present and the panel is closed (not rendered)
    expect(screen.getByTestId(DATA_TEST_ID.TRIGGER)).toBeInTheDocument();
    expect(screen.queryByTestId(DATA_TEST_ID.PANEL)).not.toBeInTheDocument();
  });

  it("opens the panel with one option per supported locale on trigger click", async () => {
    // GIVEN two supported locales
    mockSupportedLocales.current = ["en-US", "fr-FR"];
    render(<LanguageMenu />);

    // WHEN the user clicks the trigger
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TRIGGER));

    // THEN the panel renders with one option per locale
    expect(screen.getByTestId(DATA_TEST_ID.PANEL)).toBeInTheDocument();
    const renderedOptions = screen.getAllByTestId(DATA_TEST_ID.OPTION);
    expect(renderedOptions).toHaveLength(2);
  });

  it("marks the active locale's option as aria-selected", async () => {
    // GIVEN two supported locales and the user opens the panel
    mockSupportedLocales.current = ["en-US", "fr-FR"];
    const expectedActiveLocale = "en-US";
    render(<LanguageMenu />);
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TRIGGER));

    // WHEN we inspect the option whose data-locale matches the active one
    const renderedOptions = screen.getAllByTestId(DATA_TEST_ID.OPTION);
    const activeOption = renderedOptions.find(
      (node) => node.getAttribute("data-locale") === expectedActiveLocale,
    );

    // THEN that option reports aria-selected=true and the other does not
    expect(activeOption?.getAttribute("aria-selected")).toBe("true");
    const inactiveOption = renderedOptions.find(
      (node) => node.getAttribute("data-locale") !== expectedActiveLocale,
    );
    expect(inactiveOption?.getAttribute("aria-selected")).toBe("false");
  });

  it("calls i18n.changeLanguage with the picked locale and closes the panel", async () => {
    // GIVEN two supported locales, the menu open, and an expected target locale
    mockSupportedLocales.current = ["en-US", "fr-FR"];
    const expectedTargetLocale = "fr-FR";
    render(<LanguageMenu />);
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TRIGGER));

    // WHEN the user picks the target locale option
    const targetOption = screen
      .getAllByTestId(DATA_TEST_ID.OPTION)
      .find((node) => node.getAttribute("data-locale") === expectedTargetLocale);
    await userEvent.click(targetOption!);

    // THEN i18n.changeLanguage is called with that locale and the panel closes
    expect(mockChangeLanguage).toHaveBeenCalledWith(expectedTargetLocale);
    expect(screen.queryByTestId(DATA_TEST_ID.PANEL)).not.toBeInTheDocument();
  });

  it("does not call changeLanguage when the user picks the active locale", async () => {
    // GIVEN the user opens the panel
    mockSupportedLocales.current = ["en-US", "fr-FR"];
    render(<LanguageMenu />);
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TRIGGER));

    // WHEN the user clicks the currently-active locale option
    const activeOption = screen
      .getAllByTestId(DATA_TEST_ID.OPTION)
      .find((node) => node.getAttribute("data-locale") === "en-US");
    await userEvent.click(activeOption!);

    // THEN changeLanguage stays untouched but the panel still closes
    expect(mockChangeLanguage).not.toHaveBeenCalled();
    expect(screen.queryByTestId(DATA_TEST_ID.PANEL)).not.toBeInTheDocument();
  });

  it("closes the panel when the user presses Escape", async () => {
    // GIVEN the user has the panel open
    mockSupportedLocales.current = ["en-US", "fr-FR"];
    render(<LanguageMenu />);
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TRIGGER));
    expect(screen.getByTestId(DATA_TEST_ID.PANEL)).toBeInTheDocument();

    // WHEN the user presses Escape
    await userEvent.keyboard("{Escape}");

    // THEN the panel is removed from the DOM
    expect(screen.queryByTestId(DATA_TEST_ID.PANEL)).not.toBeInTheDocument();
  });

  it("closes the panel when the user clicks outside", async () => {
    // GIVEN the menu rendered alongside a sibling element, with the panel open
    mockSupportedLocales.current = ["en-US", "fr-FR"];
    render(
      <div>
        <LanguageMenu />
        <button data-testid="given-outside-element">outside</button>
      </div>,
    );
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TRIGGER));
    expect(screen.getByTestId(DATA_TEST_ID.PANEL)).toBeInTheDocument();

    // WHEN the user clicks outside the menu container
    await userEvent.click(screen.getByTestId("given-outside-element"));

    // THEN the panel closes
    expect(screen.queryByTestId(DATA_TEST_ID.PANEL)).not.toBeInTheDocument();
  });
});
