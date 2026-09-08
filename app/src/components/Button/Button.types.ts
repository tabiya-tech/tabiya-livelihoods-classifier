import type { ButtonHTMLAttributes } from "react";
import type { VariantProps } from "class-variance-authority";
import type { buttonVariants } from "./Button.variants";

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  /** Renders a leading 14px slot — typically an Icon. */
  leading?: React.ReactNode;
  /** Renders a trailing 14px slot — typically an Icon. */
  trailing?: React.ReactNode;
  /** Shows a spinner and disables interaction while running. */
  loading?: boolean;
}
