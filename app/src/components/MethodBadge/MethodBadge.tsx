import type { HTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "be62a071-9f3c-4b58-8d14-2706f5e4d8a9";

export const DATA_TEST_ID = {
  CONTAINER: `method-badge-container-${uniqueId}`,
};

export type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";

export interface MethodBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  method: HttpMethod;
}

const methodClasses: Record<HttpMethod, string> = {
  GET: "bg-teal text-white",
  POST: "bg-navy text-white",
  PUT: "bg-[#b8860b] text-white",
  DELETE: "bg-error text-white",
};

export function MethodBadge({
  method,
  className,
  ...rest
}: MethodBadgeProps) {
  return (
    <span
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "inline-block min-w-[44px] rounded-sm px-1.5 py-0.5 text-center",
        "font-mono text-[10px] font-semibold uppercase tracking-wider",
        methodClasses[method],
        className,
      )}
      {...rest}
    >
      {method}
    </span>
  );
}
