import { forwardRef } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";
import type { ButtonProps } from "@/components";
import { buttonVariants } from "@/components";

const uniqueId = "01f9e13d-269a-41e4-ab18-f6a108050c29";

export const DATA_TEST_ID = {
  CONTAINER: `button-container-${uniqueId}`,
  SPINNER: `button-spinner-${uniqueId}`,
  LEADING: `button-leading-${uniqueId}`,
  TRAILING: `button-trailing-${uniqueId}`,
  LABEL: `button-label-${uniqueId}`,
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant,
    size,
    className,
    leading,
    trailing,
    loading,
    disabled,
    children,
    type = "button",
    ...rest
  },
  ref,
) {
  const isDisabled = disabled || loading;
  return (
    <button
      ref={ref}
      type={type}
      disabled={isDisabled}
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(buttonVariants({ variant, size }), className)}
      {...rest}
    >
      {loading ? (
        <span
          aria-hidden
          data-testid={DATA_TEST_ID.SPINNER}
          className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
        />
      ) : (
        leading && <span data-testid={DATA_TEST_ID.LEADING}>{leading}</span>
      )}
      <span data-testid={DATA_TEST_ID.LABEL}>{children}</span>
      {!loading && trailing && <span data-testid={DATA_TEST_ID.TRAILING}>{trailing}</span>}
    </button>
  );
});
