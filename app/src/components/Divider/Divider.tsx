import type { HTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "a7e3b610-5d29-4f48-91c8-6b04e8d2a1f5";

export const DATA_TEST_ID = {
  CONTAINER: `divider-container-${uniqueId}`,
};

export interface DividerProps extends HTMLAttributes<HTMLDivElement> {
  /** Use a dashed line instead of solid (often signals a soft section break). */
  dashed?: boolean;
  /** Vertical orientation for inline use. Defaults to horizontal. */
  orientation?: "horizontal" | "vertical";
}

export function Divider({
  dashed,
  orientation = "horizontal",
  className,
  ...rest
}: DividerProps) {
  const vertical = orientation === "vertical";
  return (
    <div
      role="separator"
      aria-orientation={orientation}
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        vertical
          ? "inline-block h-full w-px self-stretch"
          : "h-px w-full my-6",
        dashed
          ? "bg-[length:6px_1px] bg-[linear-gradient(to_right,theme(colors.line.strong)_50%,transparent_50%)] bg-repeat-x"
          : "bg-line",
        className,
      )}
      {...rest}
    />
  );
}
