import type { ReactNode } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";
import { Icon, type IconName } from "@/components";

const uniqueId = "a9bb5fab-439b-43c0-be79-70a2e2ad06ce";

export const DATA_TEST_ID = {
  CONTAINER: `empty-state-container-${uniqueId}`,
  ICON_WRAP: `empty-state-icon-wrap-${uniqueId}`,
  TITLE: `empty-state-title-${uniqueId}`,
  DESCRIPTION: `empty-state-description-${uniqueId}`,
  ACTION: `empty-state-action-${uniqueId}`,
};

export interface EmptyStateProps {
  icon?: IconName;
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "flex flex-col items-center justify-center gap-3 rounded-md border border-line bg-paper px-8 py-12 text-center",
        className,
      )}
    >
      {icon && (
        <div
          data-testid={DATA_TEST_ID.ICON_WRAP}
          className="grid h-9 w-9 place-items-center rounded-full bg-cream-200 text-muted"
        >
          <Icon name={icon} size={16} />
        </div>
      )}
      <div
        data-testid={DATA_TEST_ID.TITLE}
        className="font-mono text-sm font-medium text-navy"
      >
        {title}
      </div>
      {description && (
        <p
          data-testid={DATA_TEST_ID.DESCRIPTION}
          className="max-w-prose text-xs leading-relaxed text-muted"
        >
          {description}
        </p>
      )}
      {action && (
        <div data-testid={DATA_TEST_ID.ACTION} className="mt-1">
          {action}
        </div>
      )}
    </div>
  );
}
