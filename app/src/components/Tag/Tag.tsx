import type { HTMLAttributes } from "react";
import type { VariantProps } from "class-variance-authority";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";
import { tagVariants } from "@/components";

const uniqueId = "d8f4b923-6a17-4e5c-bd2f-9c87a4e1f306";

export const DATA_TEST_ID = {
  CONTAINER: `tag-container-${uniqueId}`,
  DOT: `tag-dot-${uniqueId}`,
};

export interface TagProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof tagVariants> {
  /** A small dot rendered before the label, colored via currentColor. */
  dot?: boolean;
}

export function Tag({ tone, size, dot, className, children, ...rest }: TagProps) {
  return (
    <span
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(tagVariants({ tone, size }), className)}
      {...rest}
    >
      {dot && (
        <span
          aria-hidden
          data-testid={DATA_TEST_ID.DOT}
          className="h-1.5 w-1.5 rounded-full bg-current"
        />
      )}
      {children}
    </span>
  );
}
