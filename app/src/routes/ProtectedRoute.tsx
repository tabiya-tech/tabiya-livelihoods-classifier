/**
 * Wraps protected sections of the router tree.
 *
 * - While the auth state is still resolving (initial Firebase round-trip),
 *   renders a quiet centered spinner so we don't flash the login page for
 *   already-signed-in users.
 * - Once resolved: signed-in users see the children, signed-out users get
 *   redirected to /login.
 */

import { Navigate, useLocation } from "react-router-dom";
import { useFirebaseAuth } from "@/lib/auth/useFirebaseAuth";
import { Spinner } from "@/components";
import { routerPaths } from "./routerPaths";

const uniqueId = "7cb920e8-e7dc-4e23-9082-20b5f6d7e6d0";

export const DATA_TEST_ID = {
  LOADING_CONTAINER: `protected-route-loading-${uniqueId}`,
};

export interface ProtectedRouteProps {
  children: React.ReactNode;
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { user, loading } = useFirebaseAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div
        data-testid={DATA_TEST_ID.LOADING_CONTAINER}
        className="flex min-h-screen items-center justify-center"
      >
        <Spinner size={20} aria-label="Loading session" />
      </div>
    );
  }

  if (!user) {
    // Preserve where the user was trying to go so we can deep-link them
    // back after a successful sign-in.
    return (
      <Navigate
        to={routerPaths.LOGIN}
        replace
        state={{ from: location.pathname }}
      />
    );
  }

  return <>{children}</>;
}
