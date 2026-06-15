import { describe, expect, it, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";

const mockSignOut = vi.fn();
const mockUseFirebaseAuth = vi.fn();
vi.mock("@/lib/auth/useFirebaseAuth", () => ({
  useFirebaseAuth: () => mockUseFirebaseAuth(),
}));

const mockUseApiHealth = vi.fn();
vi.mock("../useApiHealth", () => ({
  useApiHealth: () => mockUseApiHealth(),
}));

import { AppShell, DATA_TEST_ID } from "./AppShell";
import { routerPaths } from "@/routes/routerPaths";
import {
  SIDEBAR_DATA_TEST_ID,
  TOPBAR_DATA_TEST_ID,
  NAV_LINK_DATA_TEST_ID,
} from "@/components";
import { NavigationGuardProvider } from "@/lib/navigationGuard";
import { renderWithRouterOutlet } from "@/_test_utilities";

const GIVEN_DASHBOARD_MARKER_TEST_ID = "given-dashboard-marker";
const GIVEN_LOGIN_MARKER_TEST_ID = "given-login-marker";

function renderShellAtPath(currentPath: string) {
  return renderWithRouterOutlet(
    <NavigationGuardProvider>
      <AppShell />
    </NavigationGuardProvider>,
    {
      currentPath,
      outletPath: routerPaths.DASHBOARD,
      outletElement: (
        <div data-testid={GIVEN_DASHBOARD_MARKER_TEST_ID}>dashboard</div>
      ),
      additionalRoutes: [
        {
          path: routerPaths.LOGIN,
          element: <div data-testid={GIVEN_LOGIN_MARKER_TEST_ID}>login</div>,
        },
      ],
    },
  );
}

beforeEach(() => {
  mockSignOut.mockReset();
  mockUseFirebaseAuth.mockReturnValue({
    user: { id: "uid-1", email: "sara.m@tabiya.org", initials: "SA" },
    loading: false,
    signInWithEmail: vi.fn(),
    signUpWithEmail: vi.fn(),
    signOut: mockSignOut,
  });
  mockUseApiHealth.mockReturnValue({
    status: "healthy" as const,
    version: "1.0.0",
    lastCheckedAt: new Date(),
  });
});

describe("AppShell", () => {
  it("renders sidebar, topbar, and the routed child", () => {
    // GIVEN the shell mounted at /dashboard
    // WHEN we render it
    renderShellAtPath(routerPaths.DASHBOARD);

    // THEN every chrome piece + the routed child appear
    expect(screen.getByTestId(SIDEBAR_DATA_TEST_ID.CONTAINER)).toBeInTheDocument();
    expect(screen.getByTestId(TOPBAR_DATA_TEST_ID.CONTAINER)).toBeInTheDocument();
    expect(screen.getByTestId(GIVEN_DASHBOARD_MARKER_TEST_ID)).toBeInTheDocument();
  });

  it("renders the Workspace and Settings nav links in the sidebar", () => {
    // GIVEN the expected nav link labels from i18n
    const expectedDashboardLabel = i18n.t("shell.nav.items.dashboard");
    const expectedConfigurationLabel = i18n.t(
      "shell.nav.items.configuration",
    );

    // WHEN we render the shell
    renderShellAtPath(routerPaths.DASHBOARD);

    // THEN both nav links appear in order: Dashboard, then Configuration
    const renderedNavLinks = screen.getAllByTestId(
      NAV_LINK_DATA_TEST_ID.CONTAINER,
    );
    expect(renderedNavLinks).toHaveLength(2);
    expect(renderedNavLinks[0]).toHaveTextContent(expectedDashboardLabel);
    expect(renderedNavLinks[1]).toHaveTextContent(expectedConfigurationLabel);
  });

  it("displays the API healthy status pill with the version", () => {
    // GIVEN the health hook reports healthy with a version, and the expected label from i18n
    const givenApiVersion = "1.0.0";
    const expectedHealthyLabel = `${i18n.t(
      "shell.topbar.apiHealthy",
    )} · v${givenApiVersion}`;
    mockUseApiHealth.mockReturnValue({
      status: "healthy",
      version: givenApiVersion,
      lastCheckedAt: new Date(),
    });

    // WHEN we render the shell
    renderShellAtPath(routerPaths.DASHBOARD);

    // THEN the health pill shows the healthy label with the given version
    expect(screen.getByTestId(DATA_TEST_ID.HEALTH_PILL)).toHaveTextContent(
      expectedHealthyLabel,
    );
  });

  it("displays a degraded label when the API reports degraded", () => {
    // GIVEN the health hook reports degraded and the expected degraded label
    const expectedDegradedLabel = i18n.t("shell.topbar.apiDegraded");
    mockUseApiHealth.mockReturnValue({
      status: "degraded",
      lastCheckedAt: null,
    });

    // WHEN we render the shell
    renderShellAtPath(routerPaths.DASHBOARD);

    // THEN the health pill reports degraded
    expect(screen.getByTestId(DATA_TEST_ID.HEALTH_PILL)).toHaveTextContent(
      expectedDegradedLabel,
    );
  });

  it("signs the user out and redirects to /login on sign-out", async () => {
    // GIVEN a signed-in user and a sign-out spy that resolves
    mockSignOut.mockResolvedValue(undefined);

    // AND the shell mounted at /dashboard
    renderShellAtPath(routerPaths.DASHBOARD);

    // WHEN the user clicks the sign-out button
    await userEvent.click(screen.getByTestId(SIDEBAR_DATA_TEST_ID.SIGN_OUT));

    // THEN Firebase signOut is invoked and the login route renders
    expect(mockSignOut).toHaveBeenCalledTimes(1);
    await waitFor(() =>
      expect(screen.getByTestId(GIVEN_LOGIN_MARKER_TEST_ID)).toBeInTheDocument(),
    );
  });
});
