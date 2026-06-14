import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { ProtectedRoute, DATA_TEST_ID } from "./ProtectedRoute";
import { routerPaths } from "./routerPaths";
import { renderWithRouter } from "@/_test_utilities";

const mockUseFirebaseAuth = vi.fn();
vi.mock("@/lib/auth/useFirebaseAuth", () => ({
  useFirebaseAuth: () => mockUseFirebaseAuth(),
}));

const GIVEN_PROTECTED_CONTENT_TEST_ID = "given-protected-content";
const GIVEN_LOGIN_PAGE_MARKER_TEST_ID = "given-login-page-marker";

function renderProtectedDashboard() {
  return renderWithRouter(
    <ProtectedRoute>
      <div data-testid={GIVEN_PROTECTED_CONTENT_TEST_ID}>protected</div>
    </ProtectedRoute>,
    {
      currentPath: routerPaths.DASHBOARD,
      elementPath: routerPaths.DASHBOARD,
      additionalRoutes: [
        {
          path: routerPaths.LOGIN,
          element: (
            <div data-testid={GIVEN_LOGIN_PAGE_MARKER_TEST_ID}>login</div>
          ),
        },
      ],
    },
  );
}

describe("ProtectedRoute", () => {
  it("renders the loading spinner while auth is resolving", () => {
    // GIVEN the auth hook is still loading
    mockUseFirebaseAuth.mockReturnValue({ user: null, loading: true });

    // WHEN we render a protected route
    renderProtectedDashboard();

    // THEN the loading container is shown and the protected child is not
    expect(screen.getByTestId(DATA_TEST_ID.LOADING_CONTAINER)).toBeInTheDocument();
    expect(
      screen.queryByTestId(GIVEN_PROTECTED_CONTENT_TEST_ID),
    ).not.toBeInTheDocument();
  });

  it("redirects to /login when the auth state resolves with no user", () => {
    // GIVEN the auth hook resolved to a signed-out state
    mockUseFirebaseAuth.mockReturnValue({ user: null, loading: false });

    // WHEN we render a protected route
    renderProtectedDashboard();

    // THEN the login page is shown and the protected child is not
    expect(
      screen.getByTestId(GIVEN_LOGIN_PAGE_MARKER_TEST_ID),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId(GIVEN_PROTECTED_CONTENT_TEST_ID),
    ).not.toBeInTheDocument();
  });

  it("renders the protected child when a user is signed in", () => {
    // GIVEN the auth hook resolved to a signed-in user
    mockUseFirebaseAuth.mockReturnValue({
      user: { id: "uid-1", email: "alex@tabiya.org", initials: "AL" },
      loading: false,
    });

    // WHEN we render a protected route
    renderProtectedDashboard();

    // THEN the protected child is rendered
    expect(screen.getByTestId(GIVEN_PROTECTED_CONTENT_TEST_ID)).toBeInTheDocument();
  });
});
