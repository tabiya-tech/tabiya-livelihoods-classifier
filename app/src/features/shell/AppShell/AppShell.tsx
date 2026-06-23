/**
 * Persistent app chrome behind every protected route.
 *
 * Composes the Sidebar, the Topbar (breadcrumbs + live API health pill +
 * language picker + ⌘K hint), and the routed child via React Router's
 * <Outlet>.
 */

import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import type { TranslationKey } from "@/react-i18next";
import {
  AppLayout,
  Sidebar,
  StatusPill,
  Topbar,
  type SidebarNavGroup,
  type BreadcrumbItem,
} from "@/components";
import { useFirebaseAuth } from "@/lib/auth/useFirebaseAuth";
import { useNavigationGuard } from "@/lib/navigationGuard";
import { routerPaths } from "@/routes/routerPaths";
import { LanguageMenu } from "@/i18n/LanguageMenu/LanguageMenu";
import { useApiHealth, type ApiHealthStatus } from "../useApiHealth";

const uniqueId = "9c1b6c2f-3c8e-4d5e-9e1d-5b8c4d2e7a4f";

export const DATA_TEST_ID = {
  CONTAINER: `app-shell-container-${uniqueId}`,
  HEALTH_PILL: `app-shell-health-pill-${uniqueId}`,
};

function buildNavGroups(
  t: (key: TranslationKey, opts?: Record<string, unknown>) => string,
): SidebarNavGroup[] {
  return [
    {
      label: t("shell.nav.groups.workspace"),
      items: [
        {
          id: "dashboard",
          label: t("shell.nav.items.dashboard"),
          icon: "dashboard",
        },
        {
          id: "classifier",
          label: t("shell.nav.items.classifier"),
          icon: "classify",
        },
      ],
    },
    {
      label: t("shell.nav.groups.settings"),
      items: [
        {
          id: "configuration",
          label: t("shell.nav.items.configuration"),
          icon: "config",
        },
        {
          id: "keys",
          label: t("shell.nav.items.keys"),
          icon: "key",
        },
      ],
    },
  ];
}

/** Map a route path to its sidebar nav item id. Empty string when no item matches. */
function deriveActiveNavId(pathname: string): string {
  if (pathname.startsWith(routerPaths.DASHBOARD)) return "dashboard";
  if (pathname.startsWith(routerPaths.CLASSIFIER)) return "classifier";
  if (pathname.startsWith(routerPaths.CONFIGURATION)) return "configuration";
  if (pathname.startsWith(routerPaths.KEYS)) return "keys";
  return "";
}

function buildBreadcrumbsForPath(
  pathname: string,
  navigateTo: (path: string) => void,
  t: (key: TranslationKey, opts?: Record<string, unknown>) => string,
): BreadcrumbItem[] {
  if (pathname.startsWith(routerPaths.DASHBOARD)) {
    return [
      {
        label: t("shell.nav.groups.workspace"),
        onClick: () => navigateTo(routerPaths.DASHBOARD),
      },
      { label: t("shell.nav.items.dashboard") },
    ];
  }
  if (pathname.startsWith(routerPaths.CLASSIFIER)) {
    return [
      {
        label: t("shell.nav.groups.workspace"),
        onClick: () => navigateTo(routerPaths.CLASSIFIER),
      },
      { label: t("shell.nav.items.classifier") },
    ];
  }
  if (pathname.startsWith(routerPaths.CONFIGURATION)) {
    return [
      {
        label: t("shell.nav.groups.settings"),
        onClick: () => navigateTo(routerPaths.CONFIGURATION),
      },
      { label: t("shell.nav.items.configuration") },
    ];
  }
  if (pathname.startsWith(routerPaths.KEYS)) {
    return [
      {
        label: t("shell.nav.groups.settings"),
        onClick: () => navigateTo(routerPaths.KEYS),
      },
      { label: t("shell.nav.items.keys") },
    ];
  }
  return [{ label: t("shell.nav.groups.workspace") }];
}

function getStatusPillLabel(
  status: ApiHealthStatus,
  version: string | undefined,
  t: (key: TranslationKey, opts?: Record<string, unknown>) => string,
): string {
  if (status === "healthy") {
    return version
      ? `${t("shell.topbar.apiHealthy")} · v${version}`
      : t("shell.topbar.apiHealthy");
  }
  if (status === "degraded") return t("shell.topbar.apiDegraded");
  if (status === "down") return t("shell.topbar.apiOffline");
  return t("shell.topbar.apiChecking");
}

function navItemIdToRoutePath(navItemId: string): string {
  if (navItemId === "dashboard") return routerPaths.DASHBOARD;
  if (navItemId === "classifier") return routerPaths.CLASSIFIER;
  if (navItemId === "configuration") return routerPaths.CONFIGURATION;
  if (navItemId === "keys") return routerPaths.KEYS;
  return routerPaths.DASHBOARD;
}

export function AppShell() {
  const { t } = useTranslation();
  const { user, signOut } = useFirebaseAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const health = useApiHealth();
  const { requestNavigate } = useNavigationGuard();

  const navGroups = buildNavGroups(t);
  const activeNavId = deriveActiveNavId(location.pathname);
  const breadcrumbs = buildBreadcrumbsForPath(
    location.pathname,
    (path) => {
      void requestNavigate(() => navigate(path));
    },
    t,
  );

  async function handleSignOut() {
    await requestNavigate(async () => {
      await signOut();
      navigate(routerPaths.LOGIN, { replace: true });
    });
  }

  return (
    <div data-testid={DATA_TEST_ID.CONTAINER}>
      <AppLayout
        sidebar={
          <Sidebar
            activeId={activeNavId}
            onNavigate={(navItemId) => {
              void requestNavigate(() =>
                navigate(navItemIdToRoutePath(navItemId)),
              );
            }}
            onBrandClick={() => {
              void requestNavigate(() => navigate(routerPaths.DASHBOARD));
            }}
            groups={navGroups}
            user={
              user ? { initials: user.initials, label: user.email } : undefined
            }
            onSignOut={user ? handleSignOut : undefined}
            signOutLabel={t("common.buttons.signOut")}
            brandName={t("shell.brand.name")}
            brandProduct={t("shell.brand.product")}
          />
        }
        topbar={
          <Topbar
            breadcrumbs={breadcrumbs}
            right={
              <>
                <StatusPill
                  data-testid={DATA_TEST_ID.HEALTH_PILL}
                  status={
                    health.status === "unknown" ? "unknown" : health.status
                  }
                >
                  {getStatusPillLabel(health.status, health.version, t)}
                </StatusPill>
                <LanguageMenu />
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
