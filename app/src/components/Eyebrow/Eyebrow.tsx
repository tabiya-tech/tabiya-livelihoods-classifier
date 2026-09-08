import type { HTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "3c4d9a1b-7e62-4a85-9f0c-2d8b6e7a4310";

export const DATA_TEST_ID = {
  CONTAINER: `eyebrow-container-${uniqueId}`,
};

export type EyebrowProps = HTMLAttributes<HTMLDivElement>;

/** Mono uppercase micro-label that sits above a heading. */
export function Eyebrow({ className, children, ...rest }: EyebrowProps) {
  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames("eyebrow", className)}
      {...rest}
    >
      {children}
    </div>
  );
}
