import type { HTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "f2a0c5d4-9b18-4d63-8a72-5e9c10b3a4d6";

export const DATA_TEST_ID = {
  CONTAINER: `kbd-container-${uniqueId}`,
};

export type KbdProps = HTMLAttributes<HTMLElement>;

/** Inline keyboard hint chip, e.g. ⌘K. */
export function Kbd({ className, children, ...rest }: KbdProps) {
  return (
    <kbd
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "inline-block whitespace-nowrap rounded-sm border border-line-strong border-b-2",
        "bg-cream-200 px-1.5 py-px font-mono text-[10px] leading-tight text-muted",
        className,
      )}
      {...rest}
    >
      {children}
    </kbd>
  );
}
