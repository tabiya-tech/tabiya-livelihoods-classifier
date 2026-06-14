import { forwardRef, type TextareaHTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "a861995a-881a-4a39-a9c2-527b04a27e71";

export const DATA_TEST_ID = {
  CONTAINER: `textarea-container-${uniqueId}`,
};

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  mono?: boolean;
  invalid?: boolean;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  function Textarea({ mono, invalid, className, ...rest }, ref) {
    return (
      <textarea
        ref={ref}
        data-testid={DATA_TEST_ID.CONTAINER}
        aria-invalid={invalid || undefined}
        className={mergeClassNames(
          "w-full resize-y rounded border bg-white px-3 py-2 text-[13px] leading-relaxed text-ink outline-none transition-shadow",
          "border-line-strong",
          "focus:border-navy focus:ring-[3px] focus:ring-navy/10",
          mono && "font-mono text-[12.5px]",
          invalid && "border-error focus:ring-error/15",
          className,
        )}
        {...rest}
      />
    );
  },
);
