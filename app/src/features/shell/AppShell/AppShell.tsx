/**
 * Persistent app chrome behind every protected route.
 *
 * Composes the Sidebar, the
 * Topbar (breadcrumbs + live API health pill + ⌘K hint), and the routed
 * child via React Router's <Outlet>.
 */

import { Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  AppLayout,
  Kbd,
  Sidebar,
  StatusPill,
  Topbar,
  type SidebarNavGroup,
  type BreadcrumbItem,
} from "@/components";
import { useFirebaseAuth } from "@/lib/auth/useFirebaseAuth";
import { routerPaths } from "@/routes/routerPaths";
import { useApiHealth, type ApiHealthStatus } from "../useApiHealth";

const uniqueId = "9c1b6c2f-3c8e-4d5e-9e1d-5b8c4d2e7a4f";

export const DATA_TEST_ID = {
  CONTAINER: `app-shell-container-${uniqueId}`,
  HEALTH_PILL: `app-shell-health-pill-${uniqueId}`,
};

const NAV_GROUPS: SidebarNavGroup[] = [
  {
    label: "Workspace",
    items: [
      {
        id: "dashboard",
        label: "Dashboard",
        icon: "dashboard",
      },
    ],
  },
];

/** Map a route path to its sidebar nav item id, or null when the path isn't in the sidebar. */
function deriveActiveNavId(pathname: string): string {
  if (pathname.startsWith(routerPaths.DASHBOARD)) return "dashboard";
  return "";
}

function buildBreadcrumbsForPath(
  pathname: string,
  navigateTo: (path: string) => void,
): BreadcrumbItem[] {
  if (pathname.startsWith(routerPaths.DASHBOARD)) {
    return [
      { label: "Workspace", onClick: () => navigateTo(routerPaths.DASHBOARD) },
      { label: "Dashboard" },
    ];
  }
  return [{ label: "Workspace" }];
}

function getStatusPillLabel(
  status: ApiHealthStatus,
  version: string | undefined,
): string {
  if (status === "healthy") return `API healthy${version ? ` · v${version}` : ""}`;
  if (status === "degraded") return "API degraded";
  if (status === "down") return "API offline";
  return "API checking…";
}

function navItemIdToRoutePath(navItemId: string): string {
  if (navItemId === "dashboard") return routerPaths.DASHBOARD;
  return routerPaths.DASHBOARD;
}

export function AppShell() {
  const { user, signOut } = useFirebaseAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const health = useApiHealth();

  const activeNavId = deriveActiveNavId(location.pathname);
  const breadcrumbs = buildBreadcrumbsForPath(location.pathname, (path) =>
    navigate(path),
  );

  async function handleSignOut() {
    await signOut();
    navigate(routerPaths.LOGIN, { replace: true });
  }

  return (
    <div data-testid={DATA_TEST_ID.CONTAINER}>
      <AppLayout
        sidebar={
          <Sidebar
            activeId={activeNavId}
            onNavigate={(navItemId) => navigate(navItemIdToRoutePath(navItemId))}
            onBrandClick={() => navigate(routerPaths.DASHBOARD)}
            groups={NAV_GROUPS}
            user={
              user ? { initials: user.initials, label: user.email } : undefined
            }
            onSignOut={user ? handleSignOut : undefined}
          />
        }
        topbar={
          <Topbar
            breadcrumbs={breadcrumbs}
            right={
              <>
                <StatusPill
                  data-testid={DATA_TEST_ID.HEALTH_PILL}
                  status={health.status === "unknown" ? "unknown" : health.status}
                >
                  {getStatusPillLabel(health.status, health.version)}
                </StatusPill>
                <Kbd>⌘ K</Kbd>
              </>
            }
          />
        }
      >
        <Outlet />
      </AppLayout>
    </div>
  );
}
