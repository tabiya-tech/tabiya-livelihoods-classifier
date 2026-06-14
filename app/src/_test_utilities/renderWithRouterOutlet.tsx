/**
 * Variant of renderWithRouter for components that render an <Outlet> and
 * therefore need a *nested* child route to fill it.
 *
 * Use this when the element under test (e.g. AppShell) is a layout route —
 * its child renders via <Outlet/>, not as a direct child of <Route element=…>.
 */

import type { ReactNode } from "react";
import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import type { AdditionalRouteEntry } from "./renderWithRouter";

export interface RenderWithRouterOutletOptions {
  /** Initial pathname rendered by the MemoryRouter. */
  currentPath: string;
  /** Path of the nested route (filled by the layout's <Outlet/>). */
  outletPath: string;
  /** Element rendered inside the layout's outlet. */
  outletElement: ReactNode;
  /** Optional sibling top-level routes (e.g. a fake /login marker). */
  additionalRoutes?: AdditionalRouteEntry[];
}

/**
 * Render a layout component that uses <Outlet/> with a nested child route.
 */
export function renderWithRouterOutlet(
  layoutElement: ReactNode,
  {
    currentPath,
    outletPath,
    outletElement,
    additionalRoutes = [],
  }: RenderWithRouterOutletOptions,
) {
  return render(
    <MemoryRouter initialEntries={[currentPath]}>
      <Routes>
        {additionalRoutes.map((extraRoute) => (
          <Route
            key={extraRoute.path}
            path={extraRoute.path}
            element={extraRoute.element}
          />
        ))}
        <Route element={layoutElement}>
          <Route path={outletPath} element={outletElement} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}
