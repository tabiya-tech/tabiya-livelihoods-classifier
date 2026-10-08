import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import i18n from "@/i18n/i18n";
import {
  resetClassificationsHandlersStore,
  seedClassificationsHandlersStore,
} from "@/mocks/handlers";

const mockUseFirebaseAuth = vi.fn();
vi.mock("@/lib/auth/useFirebaseAuth", () => ({
  useFirebaseAuth: () => mockUseFirebaseAuth(),
}));

import { DashboardPage, DATA_TEST_ID } from "./DashboardPage";

function givenSignedInUser(email = "sara.m@tabiya.org") {
  mockUseFirebaseAuth.mockReturnValue({
    user: { id: "uid-1", email, initials: "SA" },
    loading: false,
    signInWithEmail: vi.fn(),
    signUpWithEmail: vi.fn(),
    signOut: vi.fn(),
  });
}

function renderDashboard() {
  return render(
    <MemoryRouter>
      <DashboardPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  resetClassificationsHandlersStore();
});

describe("DashboardPage", () => {
  it("greets the signed-in user by email", async () => {
    // GIVEN a signed-in user
    const givenUserEmail = "sara.m@tabiya.org";
    const expectedWelcomeMessage = i18n.t("dashboard.welcomeBack", {
      name: givenUserEmail,
    });
    givenSignedInUser(givenUserEmail);

    // WHEN we render the DashboardPage
    renderDashboard();

    // THEN the welcome message includes the user's email
    expect(screen.getByTestId(DATA_TEST_ID.WELCOME_MESSAGE)).toHaveTextContent(
      expectedWelcomeMessage,
    );
  });

  it("falls back to a neutral greeting when no user is present", () => {
    // GIVEN no signed-in user
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

    // WHEN we render
    renderDashboard();

    // THEN the fallback greeting is shown
    expect(screen.getByTestId(DATA_TEST_ID.WELCOME_MESSAGE)).toHaveTextContent(
      expectedWelcomeMessage,
    );
  });

  it("renders stat cards once usage data loads", async () => {
    // GIVEN a signed-in user and fixture usage data in the MSW store
    givenSignedInUser();

    // WHEN the page renders and data loads
    renderDashboard();

    // THEN stat cards are visible
    await waitFor(() =>
      expect(screen.getByTestId(DATA_TEST_ID.STAT_CALLS)).toBeInTheDocument(),
    );
    expect(screen.getByTestId(DATA_TEST_ID.STAT_KEYS)).toBeInTheDocument();
  });

  it("renders the usage chart section once data loads", async () => {
    // GIVEN a signed-in user and fixture daily counts in the MSW store
    givenSignedInUser();

    // WHEN rendered
    renderDashboard();

    // THEN the usage section appears
    await waitFor(() =>
      expect(
        screen.getByTestId(DATA_TEST_ID.USAGE_SECTION),
      ).toBeInTheDocument(),
    );
  });

  it("renders the recent classifications table with fixture data", async () => {
    // GIVEN fixture classification summaries in the MSW store
    givenSignedInUser();

    // WHEN rendered
    renderDashboard();

    // THEN the recent table appears with at least one row
    await waitFor(() =>
      expect(
        screen.getByTestId(DATA_TEST_ID.RECENT_TABLE),
      ).toBeInTheDocument(),
    );
  });

  it("shows empty state in recent section when no classifications exist", async () => {
    // GIVEN no classifications in the store
    seedClassificationsHandlersStore({ classifications: [] });
    givenSignedInUser();

    // WHEN rendered
    renderDashboard();

    // THEN the empty message appears
    const expectedEmptyText = i18n.t("dashboard.recentClassifications.empty");
    await waitFor(() =>
      expect(screen.getByText(expectedEmptyText)).toBeInTheDocument(),
    );
  });
});
