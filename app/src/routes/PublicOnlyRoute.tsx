/**
 * Mirror of ProtectedRoute for routes that should only be visible while
 * signed out (e.g. /login). When the user *is* signed in we send them to
 * either the destination they were originally trying to reach (preserved
 * by ProtectedRoute as `location.state.from`) or to the dashboard.
 *
 * Keeping this redirect logic in the route wrapper instead of inside
 * LoginPage means every public-only page gets the same bounce behavior
 * automatically — feature components stay focused on rendering.
 */

import { Navigate, useLocation } from "react-router-dom";
import { useFirebaseAuth } from "@/lib/auth/useFirebaseAuth";
import { Spinner } from "@/components";
import { routerPaths } from "./routerPaths";

const uniqueId = "d1f8b3c0-2e9d-4f8a-b6ba-1c4f0a5e9d3a";

export const DATA_TEST_ID = {
  LOADING_CONTAINER: `public-only-route-loading-${uniqueId}`,
};

interface LocationStateWithFrom {
  from?: string;
}

export interface PublicOnlyRouteProps {
  children: React.ReactNode;
}

export function PublicOnlyRoute({ children }: PublicOnlyRouteProps) {
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

  if (user) {
    const destination =
      (location.state as LocationStateWithFrom | null)?.from ??
      routerPaths.DASHBOARD;
    return <Navigate to={destination} replace />;
  }

  return <>{children}</>;
}
