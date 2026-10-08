import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { PublicOnlyRoute, DATA_TEST_ID } from "./PublicOnlyRoute";
import { routerPaths } from "./routerPaths";
import { renderWithRouter } from "@/_test_utilities";

const mockUseFirebaseAuth = vi.fn();
vi.mock("@/lib/auth/useFirebaseAuth", () => ({
  useFirebaseAuth: () => mockUseFirebaseAuth(),
}));

const GIVEN_PUBLIC_CONTENT_TEST_ID = "given-public-content";
const GIVEN_DASHBOARD_MARKER_TEST_ID = "given-dashboard-marker";
const GIVEN_DEEP_LINK_TARGET_TEST_ID = "given-deep-link-target";
const GIVEN_DEEP_LINK_PATH = "/somewhere-protected";

function renderPublicOnlyLogin(routeState?: { from?: string }) {
  return renderWithRouter(
    <PublicOnlyRoute>
      <div data-testid={GIVEN_PUBLIC_CONTENT_TEST_ID}>login</div>
    </PublicOnlyRoute>,
    {
      currentPath: routerPaths.LOGIN,
      elementPath: routerPaths.LOGIN,
      routeState,
      additionalRoutes: [
        {
          path: routerPaths.DASHBOARD,
          element: (
            <div data-testid={GIVEN_DASHBOARD_MARKER_TEST_ID}>dashboard</div>
          ),
        },
        {
          path: GIVEN_DEEP_LINK_PATH,
          element: (
            <div data-testid={GIVEN_DEEP_LINK_TARGET_TEST_ID}>deep link</div>
          ),
        },
      ],
    },
  );
}

describe("PublicOnlyRoute", () => {
  it("renders the loading spinner while auth is resolving", () => {
    // GIVEN the auth hook is still loading
    mockUseFirebaseAuth.mockReturnValue({ user: null, loading: true });

    // WHEN we render a public-only route
    renderPublicOnlyLogin();

    // THEN the loading container is shown and the public child is not
    expect(screen.getByTestId(DATA_TEST_ID.LOADING_CONTAINER)).toBeInTheDocument();
    expect(
      screen.queryByTestId(GIVEN_PUBLIC_CONTENT_TEST_ID),
    ).not.toBeInTheDocument();
  });

  it("renders the public child when no user is signed in", () => {
    // GIVEN the auth hook resolved with no user
    mockUseFirebaseAuth.mockReturnValue({ user: null, loading: false });

    // WHEN we render a public-only route
    renderPublicOnlyLogin();

    // THEN the public child is shown
    expect(screen.getByTestId(GIVEN_PUBLIC_CONTENT_TEST_ID)).toBeInTheDocument();
  });

  it("redirects to /dashboard when the user is already signed in (no deep-link state)", () => {
    // GIVEN the auth hook resolved with a signed-in user
    mockUseFirebaseAuth.mockReturnValue({
      user: { id: "uid-1", email: "alex@tabiya.org", initials: "AL" },
      loading: false,
    });

    // WHEN we render a public-only route at /login without deep-link state
    renderPublicOnlyLogin();

    // THEN we land on /dashboard instead of the public child
    expect(screen.getByTestId(GIVEN_DASHBOARD_MARKER_TEST_ID)).toBeInTheDocument();
    expect(
      screen.queryByTestId(GIVEN_PUBLIC_CONTENT_TEST_ID),
    ).not.toBeInTheDocument();
  });

  it("redirects to the deep-link destination preserved in location state", () => {
    // GIVEN a signed-in user and a deep-link path stored in the route state
    mockUseFirebaseAuth.mockReturnValue({
      user: { id: "uid-1", email: "alex@tabiya.org", initials: "AL" },
      loading: false,
    });

    // WHEN we render the public-only route with the deep-link in state.from
    renderPublicOnlyLogin({ from: GIVEN_DEEP_LINK_PATH });

    // THEN we land on the original target route, not on /dashboard
    expect(screen.getByTestId(GIVEN_DEEP_LINK_TARGET_TEST_ID)).toBeInTheDocument();
    expect(
      screen.queryByTestId(GIVEN_DASHBOARD_MARKER_TEST_ID),
    ).not.toBeInTheDocument();
  });
});
