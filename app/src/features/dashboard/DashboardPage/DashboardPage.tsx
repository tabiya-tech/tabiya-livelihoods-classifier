import { Eyebrow } from "@/components";
import { useFirebaseAuth } from "@/lib/auth/useFirebaseAuth";

const uniqueId = "f0e3a6a2-2b18-4f0b-9d8b-7d2c9f1a8b3e";

export const DATA_TEST_ID = {
  CONTAINER: `dashboard-page-container-${uniqueId}`,
  WELCOME_MESSAGE: `dashboard-page-welcome-${uniqueId}`,
};

export function DashboardPage() {
  const { user } = useFirebaseAuth();
  const displayName = user?.email ?? "there";

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className="mx-auto w-full max-w-5xl px-10 py-10"
    >
      <Eyebrow>Workspace · Dashboard</Eyebrow>
      <h1
        data-testid={DATA_TEST_ID.WELCOME_MESSAGE}
        className="h-page mt-2"
      >
        Welcome back, {displayName}
      </h1>
    </div>
  );
}
