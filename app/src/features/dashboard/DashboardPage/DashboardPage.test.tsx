import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import i18n from "@/i18n/i18n";

const mockUseFirebaseAuth = vi.fn();
vi.mock("@/lib/auth/useFirebaseAuth", () => ({
  useFirebaseAuth: () => mockUseFirebaseAuth(),
}));

import { DashboardPage, DATA_TEST_ID } from "./DashboardPage";

describe("DashboardPage", () => {
  it("greets the signed-in user by email", () => {
    // GIVEN a signed-in user and the expected greeting derived from the i18n template
    const givenUserEmail = "sara.m@tabiya.org";
    const expectedWelcomeMessage = i18n.t("dashboard.welcomeBack", {
      name: givenUserEmail,
    });
    mockUseFirebaseAuth.mockReturnValue({
      user: { id: "uid-1", email: givenUserEmail, initials: "SA" },
      loading: false,
      signInWithEmail: vi.fn(),
      signUpWithEmail: vi.fn(),
      signOut: vi.fn(),
    });

    // WHEN we render the DashboardPage
    render(<DashboardPage />);

    // THEN the welcome message includes the user's email
    expect(screen.getByTestId(DATA_TEST_ID.WELCOME_MESSAGE)).toHaveTextContent(
      expectedWelcomeMessage,
    );
  });

  it("falls back to a neutral greeting when no user is present", () => {
    // GIVEN no signed-in user and the expected fallback greeting from i18n
    const expectedFallbackName = i18n.t("dashboard.fallbackName");
    const expectedWelcomeMessage = i18n.t("dashboard.welcomeBack", {
      name: expectedFallbackName,
    });
    mockUseFirebaseAuth.mockReturnValue({
      user: null,
      loading: false,
      signInWithEmail: vi.fn(),
      signUpWithEmail: vi.fn(),
      signOut: vi.fn(),
    });

    // WHEN we render the DashboardPage
    render(<DashboardPage />);

    // THEN the welcome message uses the fallback greeting
    expect(screen.getByTestId(DATA_TEST_ID.WELCOME_MESSAGE)).toHaveTextContent(
      expectedWelcomeMessage,
    );
  });
});
