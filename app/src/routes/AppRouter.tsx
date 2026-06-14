/**
 * Top-level router. Public /login route + everything else inside the protected
 * AppShell.
 */

import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { LoginPage } from "@/features/auth/LoginPage/LoginPage";
import { AppShell } from "@/features/shell/AppShell/AppShell";
import { DashboardPage } from "@/features/dashboard/DashboardPage/DashboardPage";
import { ProtectedRoute } from "./ProtectedRoute";
import { PublicOnlyRoute } from "./PublicOnlyRoute";
import { routerPaths } from "./routerPaths";

export function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path={routerPaths.LOGIN}
          element={
            <PublicOnlyRoute>
              <LoginPage />
            </PublicOnlyRoute>
          }
        />
        <Route
          element={
            <ProtectedRoute>
              <AppShell />
            </ProtectedRoute>
          }
        >
          <Route path={routerPaths.DASHBOARD} element={<DashboardPage />} />
        </Route>
        <Route
          path="*"
          element={<Navigate to={routerPaths.DASHBOARD} replace />}
        />
      </Routes>
    </BrowserRouter>
  );
}
