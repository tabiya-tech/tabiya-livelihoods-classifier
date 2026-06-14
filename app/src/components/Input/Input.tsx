import { forwardRef, type InputHTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "c55c4d0a-880d-41d9-854f-ed7c268a27fd";

export const DATA_TEST_ID = {
  CONTAINER: `input-container-${uniqueId}`,
};

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  /** Use the monospace face — for API keys, model IDs, etc. */
  mono?: boolean;
  /** Visual error state — pairs with FormField's error message. */
  invalid?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { mono, invalid, className, ...rest },
  ref,
) {
  return (
    <input
      ref={ref}
      data-testid={DATA_TEST_ID.CONTAINER}
      aria-invalid={invalid || undefined}
      className={mergeClassNames(
        "w-full rounded border bg-white px-3 py-2 text-[13px] text-ink outline-none transition-shadow",
        "border-line-strong",
        "focus:border-navy focus:ring-[3px] focus:ring-navy/10",
        mono && "font-mono text-[12.5px]",
        invalid && "border-error focus:ring-error/15",
        className,
      )}
      {...rest}
    />
  );
});
