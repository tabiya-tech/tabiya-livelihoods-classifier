import { describe, expect, it, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

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
import { renderWithRouterOutlet } from "@/_test_utilities";

const GIVEN_DASHBOARD_MARKER_TEST_ID = "given-dashboard-marker";
const GIVEN_LOGIN_MARKER_TEST_ID = "given-login-marker";

function renderShellAtPath(currentPath: string) {
  return renderWithRouterOutlet(<AppShell />, {
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
  });
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

  it("renders a Dashboard nav link in the sidebar", () => {
    // GIVEN the expected nav link label
    const expectedNavLinkLabel = "Dashboard";

    // WHEN we render the shell
    renderShellAtPath(routerPaths.DASHBOARD);

    // THEN a single nav link is rendered with that label
    const renderedNavLinks = screen.getAllByTestId(NAV_LINK_DATA_TEST_ID.CONTAINER);
    expect(renderedNavLinks).toHaveLength(1);
    expect(renderedNavLinks[0]).toHaveTextContent(expectedNavLinkLabel);
  });

  it("displays the API healthy status pill with the version", () => {
    // GIVEN the health hook reports healthy with a version
    const givenApiVersion = "1.0.0";
    mockUseApiHealth.mockReturnValue({
      status: "healthy",
      version: givenApiVersion,
      lastCheckedAt: new Date(),
    });

    // WHEN we render the shell
    renderShellAtPath(routerPaths.DASHBOARD);

    // THEN the health pill shows the healthy label with the given version
    expect(screen.getByTestId(DATA_TEST_ID.HEALTH_PILL)).toHaveTextContent(
      `API healthy · v${givenApiVersion}`,
    );
  });

  it("displays a degraded label when the API reports degraded", () => {
    // GIVEN the health hook reports degraded
    mockUseApiHealth.mockReturnValue({
      status: "degraded",
      lastCheckedAt: null,
    });

    // WHEN we render the shell
    renderShellAtPath(routerPaths.DASHBOARD);

    // THEN the health pill reports degraded
    expect(screen.getByTestId(DATA_TEST_ID.HEALTH_PILL)).toHaveTextContent(
      /degraded/i,
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
