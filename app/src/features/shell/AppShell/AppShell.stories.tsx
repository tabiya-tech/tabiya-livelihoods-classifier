import type { Meta, StoryObj } from "@storybook/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { AppShell } from "./AppShell";
import { routerPaths } from "@/routes/routerPaths";
import {
  AuthOverrideProvider,
  type AuthenticatedUser,
} from "@/lib/auth/useFirebaseAuth";
import {
  ApiHealthOverrideProvider,
  type ApiHealthSnapshot,
} from "../useApiHealth";
import { VisualMock } from "@/_test_utilities";

const meta: Meta<typeof AppShell> = {
  title: "Pages/AppShell",
  component: AppShell,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof AppShell>;

const DEFAULT_SIGNED_IN_USER: AuthenticatedUser = {
  id: "uid-sara",
  email: "sara.m@tabiya.org",
  initials: "SM",
};

const HEALTHY_SNAPSHOT: ApiHealthSnapshot = {
  status: "healthy",
  version: "1.0.0",
  lastCheckedAt: new Date("2026-06-15T12:00:00Z"),
};

const DEGRADED_SNAPSHOT: ApiHealthSnapshot = {
  status: "degraded",
  lastCheckedAt: null,
};

const DOWN_SNAPSHOT: ApiHealthSnapshot = {
  status: "down",
  lastCheckedAt: null,
};

interface AppShellStoryHarnessProps {
  user: AuthenticatedUser | null;
  healthSnapshot: ApiHealthSnapshot;
}

function AppShellStoryHarness({
  user,
  healthSnapshot,
}: AppShellStoryHarnessProps) {
  return (
    <MemoryRouter initialEntries={[routerPaths.DASHBOARD]}>
      <AuthOverrideProvider
        value={{
          user,
          loading: false,
          signInWithEmail: async () => undefined,
          signUpWithEmail: async () => undefined,
          signOut: async () => undefined,
        }}
      >
        <ApiHealthOverrideProvider value={healthSnapshot}>
          <Routes>
            <Route element={<AppShell />}>
              <Route
                path={routerPaths.DASHBOARD}
                element={<VisualMock text="Routed page content" />}
              />
            </Route>
          </Routes>
        </ApiHealthOverrideProvider>
      </AuthOverrideProvider>
    </MemoryRouter>
  );
}

export const SignedInHealthy: Story = {
  render: () => (
    <AppShellStoryHarness
      user={DEFAULT_SIGNED_IN_USER}
      healthSnapshot={HEALTHY_SNAPSHOT}
    />
  ),
};

export const ApiDegraded: Story = {
  render: () => (
    <AppShellStoryHarness
      user={DEFAULT_SIGNED_IN_USER}
      healthSnapshot={DEGRADED_SNAPSHOT}
    />
  ),
};

export const ApiDown: Story = {
  render: () => (
    <AppShellStoryHarness
      user={DEFAULT_SIGNED_IN_USER}
      healthSnapshot={DOWN_SNAPSHOT}
    />
  ),
};
