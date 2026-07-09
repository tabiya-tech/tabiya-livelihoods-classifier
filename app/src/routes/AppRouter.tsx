/**
 * Top-level router. Public /login route + everything else inside the protected
 * AppShell.
 */

import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { LoginPage } from "@/features/auth/LoginPage/LoginPage";
import { AppShell } from "@/features/shell/AppShell/AppShell";
import { ClassifierPage } from "@/features/classifier/ClassifierPage/ClassifierPage";
import { ConfigurationPage } from "@/features/configuration/ConfigurationPage/ConfigurationPage";
import { DashboardPage } from "@/features/dashboard/DashboardPage/DashboardPage";
import { ApiKeysPage } from "@/features/keys/ApiKeysPage/ApiKeysPage";
import { PipelinesPage } from "@/features/pipelines/PipelinesPage/PipelinesPage";
import { PipelineEditorPage } from "@/features/pipelines/PipelineEditorPage/PipelineEditorPage";
import { PipelineLibraryPage } from "@/features/pipelines/PipelineLibraryPage/PipelineLibraryPage";
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
          <Route
            path={routerPaths.CLASSIFIER}
            element={<ClassifierPage />}
          />
          <Route
            path={routerPaths.PIPELINES}
            element={<PipelinesPage />}
          />
          <Route
            path={routerPaths.PIPELINE_NEW}
            element={<PipelineEditorPage />}
          />
          {/* PIPELINE_LIBRARY must appear before PIPELINE_EDIT — otherwise
              the :pipelineId param would match "library". */}
          <Route
            path={routerPaths.PIPELINE_LIBRARY}
            element={<PipelineLibraryPage />}
          />
          <Route
            path={routerPaths.PIPELINE_EDIT}
            element={<PipelineEditorPage />}
          />
          <Route
            path={routerPaths.CONFIGURATION}
            element={<ConfigurationPage />}
          />
          <Route path={routerPaths.KEYS} element={<ApiKeysPage />} />
        </Route>
        <Route
          path="*"
          element={<Navigate to={routerPaths.DASHBOARD} replace />}
        />
      </Routes>
    </BrowserRouter>
  );
}
