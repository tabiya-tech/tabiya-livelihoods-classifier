import type { LabelHTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "e7e4ee06-734e-42c2-b047-b3fb7e4bfc29";

export const DATA_TEST_ID = {
  CONTAINER: `label-container-${uniqueId}`,
  REQUIRED_MARK: `label-required-mark-${uniqueId}`,
};

export interface LabelProps extends LabelHTMLAttributes<HTMLLabelElement> {
  /** Adds a small red asterisk to indicate a required field. */
  required?: boolean;
}

export function Label({
  required,
  className,
  children,
  ...rest
}: LabelProps) {
  return (
    <label
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "mb-1.5 block font-mono text-[11px] font-medium tracking-[0.01em] text-navy",
        className,
      )}
      {...rest}
    >
      {children}
      {required && (
        <span aria-hidden data-testid={DATA_TEST_ID.REQUIRED_MARK} className="ml-0.5 text-error">
          *
        </span>
      )}
    </label>
  );
}
