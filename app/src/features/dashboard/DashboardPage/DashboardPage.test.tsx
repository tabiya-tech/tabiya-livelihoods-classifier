import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

const mockUseFirebaseAuth = vi.fn();
vi.mock("@/lib/auth/useFirebaseAuth", () => ({
  useFirebaseAuth: () => mockUseFirebaseAuth(),
}));

import { DashboardPage, DATA_TEST_ID } from "./DashboardPage";

describe("DashboardPage", () => {
  it("greets the signed-in user by email", () => {
    // GIVEN a signed-in user
    const givenUserEmail = "sara.m@tabiya.org";
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
      givenUserEmail,
    );
  });

  it("falls back to a neutral greeting when no user is present", () => {
    // GIVEN no signed-in user (e.g. mid-sign-out transition)
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
      /welcome back, there/i,
    );
  });
});
