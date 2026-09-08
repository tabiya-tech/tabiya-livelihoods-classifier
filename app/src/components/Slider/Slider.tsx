import { forwardRef, type InputHTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "d23e7215-f2ce-45fb-bafc-ecd03b4d51b0";

export const DATA_TEST_ID = {
  CONTAINER: `slider-container-${uniqueId}`,
  INPUT: `slider-input-${uniqueId}`,
  LABEL: `slider-label-${uniqueId}`,
  VALUE: `slider-value-${uniqueId}`,
  HINT: `slider-hint-${uniqueId}`,
};

export interface SliderProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, "type"> {
  /** Optional formatter for the displayed value. */
  format?: (value: number) => string;
  /** Optional label shown above the track. */
  label?: string;
  /** Optional hint below the track. */
  hint?: string;
}

export const Slider = forwardRef<HTMLInputElement, SliderProps>(function Slider(
  { label, hint, format, value, defaultValue, className, ...rest },
  ref,
) {
  const displayed = format
    ? format(Number(value ?? defaultValue ?? 0))
    : String(value ?? defaultValue ?? "");

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames("w-full", className)}
    >
      {label && (
        <div className="mb-1 flex items-baseline justify-between font-mono">
          <span
            data-testid={DATA_TEST_ID.LABEL}
            className="text-xs font-medium text-navy"
          >
            {label}
          </span>
          <span
            data-testid={DATA_TEST_ID.VALUE}
            className="text-xs text-muted"
          >
            {displayed}
          </span>
        </div>
      )}
      <input
        ref={ref}
        type="range"
        data-testid={DATA_TEST_ID.INPUT}
        value={value}
        defaultValue={defaultValue}
        className={mergeClassNames(
          "h-1 w-full appearance-none rounded-sm bg-line outline-none",
          "[&::-webkit-slider-thumb]:appearance-none",
          "[&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5",
          "[&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-navy",
          "[&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-paper",
          "[&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5",
          "[&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-navy",
          "[&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-paper",
        )}
        {...rest}
      />
      {hint && (
        <p
          data-testid={DATA_TEST_ID.HINT}
          className="mt-1 text-[11px] leading-snug text-muted"
        >
          {hint}
        </p>
      )}
    </div>
  );
});
