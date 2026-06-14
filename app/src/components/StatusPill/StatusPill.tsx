import type { HTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "5c1a8b67-3d49-4f02-91e5-7a4b8d6c2e98";

export const DATA_TEST_ID = {
  CONTAINER: `status-pill-container-${uniqueId}`,
  DOT: `status-pill-dot-${uniqueId}`,
};

export interface StatusPillProps extends HTMLAttributes<HTMLSpanElement> {
  status?: "healthy" | "degraded" | "down" | "unknown";
}

const dotColor: Record<NonNullable<StatusPillProps["status"]>, string> = {
  healthy: "bg-lime-600",
  degraded: "bg-yellow",
  down: "bg-error",
  unknown: "bg-line-strong",
};

export function StatusPill({
  status = "healthy",
  className,
  children,
  ...rest
}: StatusPillProps) {
  return (
    <span
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full",
        "border border-line bg-paper px-2.5 py-1 font-mono text-[11px] text-muted",
        className,
      )}
      {...rest}
    >
      <span
        aria-hidden
        data-testid={DATA_TEST_ID.DOT}
        className={mergeClassNames("h-1.5 w-1.5 rounded-full", dotColor[status])}
      />
      {children}
    </span>
  );
}
