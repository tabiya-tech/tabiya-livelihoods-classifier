import type { ReactNode } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "92dfd837-3b3c-463b-9aa0-fed5dfefe5ff";

export const DATA_TEST_ID = {
  NAV: `breadcrumbs-nav-${uniqueId}`,
  ITEM: `breadcrumbs-item-${uniqueId}`,
  LINK: `breadcrumbs-link-${uniqueId}`,
};

export interface BreadcrumbItem {
  label: ReactNode;
  href?: string;
  onClick?: () => void;
}

export interface BreadcrumbsProps {
  items: BreadcrumbItem[];
  className?: string;
  /** Separator character between items. Defaults to /. */
  separator?: string;
}

export function Breadcrumbs({
  items,
  className,
  separator = "/",
}: BreadcrumbsProps) {
  return (
    <nav
      data-testid={DATA_TEST_ID.NAV}
      aria-label="Breadcrumb"
      className={mergeClassNames("font-mono text-xs text-muted", className)}
    >
      <ol className="flex items-center gap-2">
        {items.map((item, index) => {
          const isLast = index === items.length - 1;
          const content =
            item.href || item.onClick ? (
              <a
                href={item.href ?? "#"}
                data-testid={DATA_TEST_ID.LINK}
                onClick={(event) => {
                  if (item.onClick) {
                    event.preventDefault();
                    item.onClick();
                  }
                }}
                className="hover:text-navy"
              >
                {item.label}
              </a>
            ) : (
              <span className={isLast ? "font-medium text-navy" : undefined}>
                {item.label}
              </span>
            );
          return (
            <li
              key={index}
              data-testid={DATA_TEST_ID.ITEM}
              className="flex items-center gap-2"
            >
              {content}
              {!isLast && (
                <span aria-hidden className="opacity-40">
                  {separator}
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
