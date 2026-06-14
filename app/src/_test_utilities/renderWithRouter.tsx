/**
 * Renders a React element inside a MemoryRouter so router-aware components
 * (NavLink, useNavigate, useLocation, etc.) can be exercised in unit tests
 * without a full app shell.
 *
 * - `currentPath` is the initial pathname.
 * - `routes` lets you register sibling routes (e.g. a fake `/login` page) so
 *   you can assert post-navigation state.
 * - `routeState` is React Router's `location.state` for the initial entry —
 *   useful for testing redirect logic that reads `state.from`.
 */

import type { ReactNode } from "react";
import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

export interface AdditionalRouteEntry {
  /** Path pattern for the sibling route (e.g. "/login"). */
  path: string;
  /** Element to render when the path matches. */
  element: ReactNode;
}

export interface RenderWithRouterOptions {
  /** Initial pathname rendered by the MemoryRouter. */
  currentPath: string;
  /** Path pattern the element under test is mounted at. */
  elementPath: string;
  /** Optional sibling routes for navigation assertions. */
  additionalRoutes?: AdditionalRouteEntry[];
  /** Optional `location.state` for the initial entry. */
  routeState?: Record<string, unknown>;
}

/**
 * Render the given element behind a MemoryRouter at the configured path,
 * alongside any extra routes the test needs (e.g. redirect targets).
 */
export function renderWithRouter(
  element: ReactNode,
  {
    currentPath,
    elementPath,
    additionalRoutes = [],
    routeState,
  }: RenderWithRouterOptions,
) {
  const initialEntry = routeState
    ? { pathname: currentPath, state: routeState }
    : currentPath;

  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path={elementPath} element={element} />
        {additionalRoutes.map((extraRoute) => (
          <Route
            key={extraRoute.path}
            path={extraRoute.path}
            element={extraRoute.element}
          />
        ))}
      </Routes>
    </MemoryRouter>,
  );
}
