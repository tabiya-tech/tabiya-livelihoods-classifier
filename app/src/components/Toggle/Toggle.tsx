import { forwardRef, useState, type ButtonHTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "3851e84d-3afe-46c0-8f03-f92e2e15fb29";

export const DATA_TEST_ID = {
  CONTAINER: `toggle-container-${uniqueId}`,
  THUMB: `toggle-thumb-${uniqueId}`,
};

export interface ToggleProps
  extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "onChange" | "value"> {
  checked?: boolean;
  defaultChecked?: boolean;
  onChange?: (checked: boolean) => void;
  /** Accessible label describing what the toggle controls. */
  label?: string;
}

export const Toggle = forwardRef<HTMLButtonElement, ToggleProps>(function Toggle(
  { checked, defaultChecked, onChange, label, className, disabled, ...rest },
  ref,
) {
  const isControlled = checked !== undefined;
  const [internalChecked, setInternalChecked] = useState(!!defaultChecked);
  const value = isControlled ? checked : internalChecked;

  function handleClick() {
    const next = !value;
    if (!isControlled) setInternalChecked(next);
    onChange?.(next);
  }

  return (
    <button
      ref={ref}
      type="button"
      role="switch"
      data-testid={DATA_TEST_ID.CONTAINER}
      aria-checked={!!value}
      aria-label={label}
      disabled={disabled}
      onClick={handleClick}
      className={mergeClassNames(
        "relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full",
        "border transition-colors outline-none",
        "focus-visible:ring-2 focus-visible:ring-navy/30 focus-visible:ring-offset-1 focus-visible:ring-offset-cream",
        value ? "bg-navy border-navy" : "bg-line border-line-strong",
        disabled && "opacity-45 cursor-not-allowed",
        className,
      )}
      {...rest}
    >
      <span
        aria-hidden
        data-testid={DATA_TEST_ID.THUMB}
        className={mergeClassNames(
          "inline-block h-3.5 w-3.5 transform rounded-full bg-paper shadow-card-1 transition-transform",
          value ? "translate-x-[18px]" : "translate-x-0.5",
        )}
      />
    </button>
  );
});
