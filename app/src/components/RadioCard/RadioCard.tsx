import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "37f502c1-59ec-485b-a0e0-d4385abe9a65";

export const DATA_TEST_ID = {
  CONTAINER: `radio-card-container-${uniqueId}`,
  INDICATOR: `radio-card-indicator-${uniqueId}`,
  INDICATOR_DOT: `radio-card-indicator-dot-${uniqueId}`,
  TITLE: `radio-card-title-${uniqueId}`,
  DESCRIPTION: `radio-card-description-${uniqueId}`,
  META: `radio-card-meta-${uniqueId}`,
};

export interface RadioCardProps
  extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "title"> {
  selected?: boolean;
  /** Primary title shown next to the radio indicator. */
  title: ReactNode;
  /** Optional supporting text under the title. */
  description?: ReactNode;
  /** Slot rendered on the right side (tags, scores, etc.). */
  meta?: ReactNode;
}

export const RadioCard = forwardRef<HTMLButtonElement, RadioCardProps>(
  function RadioCard(
    { selected, title, description, meta, className, type = "button", ...rest },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={type}
        role="radio"
        data-testid={DATA_TEST_ID.CONTAINER}
        aria-checked={!!selected}
        className={mergeClassNames(
          "flex w-full items-start gap-3 rounded-md border bg-paper p-4 text-left transition-colors outline-none",
          "border-line hover:border-line-strong",
          selected && "border-navy bg-white shadow-[0_0_0_3px_rgba(0,33,71,0.08)]",
          "focus-visible:ring-2 focus-visible:ring-navy/30 focus-visible:ring-offset-1",
          className,
        )}
        {...rest}
      >
        <span
          aria-hidden
          data-testid={DATA_TEST_ID.INDICATOR}
          className={mergeClassNames(
            "mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full border bg-white",
            selected ? "border-navy" : "border-line-strong",
          )}
        >
          {selected && (
            <span
              data-testid={DATA_TEST_ID.INDICATOR_DOT}
              className="h-2 w-2 rounded-full bg-navy"
            />
          )}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-baseline justify-between gap-2">
            <span
              data-testid={DATA_TEST_ID.TITLE}
              className="font-mono text-[13px] font-medium text-navy"
            >
              {title}
            </span>
            {meta && (
              <span
                data-testid={DATA_TEST_ID.META}
                className="flex items-center gap-1.5"
              >
                {meta}
              </span>
            )}
          </span>
          {description && (
            <span
              data-testid={DATA_TEST_ID.DESCRIPTION}
              className="mt-1 block text-xs leading-relaxed text-muted"
            >
              {description}
            </span>
          )}
        </span>
      </button>
    );
  },
);
