import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import ProtectedRoute from "./auth/ProtectedRoute";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
// import Configuration from "./pages/Configuration"; // hidden: model config not user-configurable yet
import ApiKeys from "./pages/ApiKeys";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />

        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        {/* <Route
          path="/configuration"
          element={
            <ProtectedRoute>
              <Configuration />
            </ProtectedRoute>
          }
        /> */}
        <Route
          path="/api-keys"
          element={
            <ProtectedRoute>
              <ApiKeys />
            </ProtectedRoute>
          }
        />

        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
