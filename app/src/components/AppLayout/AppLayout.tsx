import type { ReactNode } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "b8cabb2d-d140-48e5-b5bd-62d5dfa46628";

export const DATA_TEST_ID = {
  CONTAINER: `app-layout-container-${uniqueId}`,
  MAIN: `app-layout-main-${uniqueId}`,
};

export interface AppLayoutProps {
  sidebar: ReactNode;
  topbar?: ReactNode;
  children: ReactNode;
  className?: string;
}

/**
 * The page chrome: persistent sidebar on the left, sticky topbar, scrolling
 * content area. Page components decide their own inner padding.
 */
export function AppLayout({
  sidebar,
  topbar,
  children,
  className,
}: AppLayoutProps) {
  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "grid min-h-screen grid-cols-[232px_1fr] bg-cream",
        className,
      )}
    >
      {sidebar}
      <div className="flex min-h-screen flex-col">
        {topbar}
        <main data-testid={DATA_TEST_ID.MAIN} className="flex-1">
          {children}
        </main>
      </div>
    </div>
  );
}
