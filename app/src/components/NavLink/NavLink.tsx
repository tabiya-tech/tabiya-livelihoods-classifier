import type { ButtonHTMLAttributes, ReactNode } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "afcdd3cc-8b42-4ac3-821c-9dfedb45716b";

export const DATA_TEST_ID = {
  CONTAINER: `nav-link-container-${uniqueId}`,
  ICON_SLOT: `nav-link-icon-slot-${uniqueId}`,
  LABEL: `nav-link-label-${uniqueId}`,
};

export interface NavLinkProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Visual selected state. */
  active?: boolean;
  /** Leading 14px slot — typically an Icon. */
  icon?: ReactNode;
}

/**
 * Sidebar nav row. Renders as a button so navigation is driven by the parent
 * (we don't tie this primitive to a router).
 */
export function NavLink({
  active,
  icon,
  className,
  children,
  type = "button",
  ...rest
}: NavLinkProps) {
  return (
    <button
      type={type}
      data-testid={DATA_TEST_ID.CONTAINER}
      aria-current={active ? "page" : undefined}
      className={mergeClassNames(
        "flex w-full items-center gap-2.5 rounded px-2.5 py-1.5 text-left text-[13px]",
        "transition-colors outline-none",
        "focus-visible:ring-2 focus-visible:ring-lime/40",
        active
          ? "bg-lime text-navy font-medium"
          : "text-cream/80 hover:bg-white/[0.06] hover:text-cream",
        className,
      )}
      {...rest}
    >
      {icon && (
        <span
          data-testid={DATA_TEST_ID.ICON_SLOT}
          className="grid h-3.5 w-3.5 place-items-center"
        >
          {icon}
        </span>
      )}
      <span data-testid={DATA_TEST_ID.LABEL}>{children}</span>
    </button>
  );
}
