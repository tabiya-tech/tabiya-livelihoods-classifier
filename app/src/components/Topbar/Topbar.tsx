import type { ReactNode } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";
import { Breadcrumbs, type BreadcrumbItem } from "@/components";

const uniqueId = "2114e3e0-40db-40a0-a142-f16816cfbd7b";

export const DATA_TEST_ID = {
  CONTAINER: `topbar-container-${uniqueId}`,
  RIGHT_SLOT: `topbar-right-slot-${uniqueId}`,
};

export interface TopbarProps {
  /** Breadcrumb trail rendered on the left. */
  breadcrumbs: BreadcrumbItem[];
  /** Right-side slot — typically a status pill and shortcut hint. */
  right?: ReactNode;
  className?: string;
}

export function Topbar({ breadcrumbs, right, className }: TopbarProps) {
  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "sticky top-0 z-10 flex items-center justify-between border-b border-line bg-cream px-4 py-3 sm:px-8 sm:py-4",
        className,
      )}
    >
      <Breadcrumbs items={breadcrumbs} />
      {right && (
        <div
          data-testid={DATA_TEST_ID.RIGHT_SLOT}
          className="flex items-center gap-2.5"
        >
          {right}
        </div>
      )}
    </div>
  );
}
