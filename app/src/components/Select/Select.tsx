import { forwardRef, type SelectHTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "e438f4cc-7837-45b7-90c1-15cb2f145822";

export const DATA_TEST_ID = {
  CONTAINER: `select-container-${uniqueId}`,
};

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  mono?: boolean;
  invalid?: boolean;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { mono, invalid, className, children, ...rest },
  ref,
) {
  return (
    <select
      ref={ref}
      data-testid={DATA_TEST_ID.CONTAINER}
      aria-invalid={invalid || undefined}
      className={mergeClassNames(
        "w-full appearance-none rounded border bg-white bg-no-repeat pl-3 pr-9 py-2 text-[13px] text-ink outline-none transition-shadow",
        "border-line-strong",
        "focus:border-navy focus:ring-[3px] focus:ring-navy/10",
        // chevron via svg background
        "bg-[length:11px_7px] bg-[position:right_12px_center]",
        "bg-[url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='11' height='7' viewBox='0 0 11 7' fill='none' stroke='%236b6b6b' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'><path d='M1 1l4.5 5L10 1'/></svg>\")]",
        mono && "font-mono text-[12.5px]",
        invalid && "border-error focus:ring-error/15",
        className,
      )}
      {...rest}
    >
      {children}
    </select>
  );
});
