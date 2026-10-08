import { forwardRef, type ButtonHTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";
import { Icon } from "@/components";
import type { IconName } from "@/components";

const uniqueId = "193eaebd-acff-4efe-9dd5-0bfd0eacd22f";

export const DATA_TEST_ID = {
  CONTAINER: `icon-button-container-${uniqueId}`,
};

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  icon: IconName;
  /** Required accessible label — IconButton has no visible text. */
  "aria-label": string;
  variant?: "default" | "ghost" | "primary" | "danger";
  size?: "sm" | "md";
}

const variantClasses: Record<NonNullable<IconButtonProps["variant"]>, string> = {
  default:
    "bg-paper text-navy border-line-strong hover:bg-cream-200 hover:border-navy",
  ghost: "bg-transparent text-navy border-line hover:bg-paper",
  primary: "bg-navy text-lime border-navy hover:bg-navy-700",
  danger: "bg-paper text-error border-line hover:bg-[#fcecea] hover:border-error",
};

const sizeClasses: Record<NonNullable<IconButtonProps["size"]>, string> = {
  sm: "h-6 w-6 rounded-sm",
  md: "h-8 w-8 rounded",
};

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton(
    { icon, variant = "ghost", size = "md", className, type = "button", ...rest },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={type}
        data-testid={DATA_TEST_ID.CONTAINER}
        className={mergeClassNames(
          "inline-flex items-center justify-center border transition-colors outline-none",
          "focus-visible:ring-2 focus-visible:ring-navy/30 focus-visible:ring-offset-1 focus-visible:ring-offset-cream",
          "disabled:opacity-45 disabled:cursor-not-allowed",
          variantClasses[variant],
          sizeClasses[size],
          className,
        )}
        {...rest}
      >
        <Icon name={icon} size={size === "sm" ? 12 : 14} />
      </button>
    );
  },
);
