import { type ReactNode } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "89330fc8-c4a1-4cfa-bf70-d48210063d9e";

export const DATA_TEST_ID = {
  LIST: `tabs-list-${uniqueId}`,
  TAB: `tabs-tab-${uniqueId}`,
};

export interface TabItem {
  id: string;
  label: ReactNode;
  /** Optional count or supporting label rendered next to the tab. */
  meta?: ReactNode;
  disabled?: boolean;
}

export interface TabsProps {
  items: TabItem[];
  value: string;
  onChange: (id: string) => void;
  className?: string;
  /** ARIA label for the tablist (required when no visible heading describes it). */
  "aria-label"?: string;
}

export function Tabs({
  items,
  value,
  onChange,
  className,
  "aria-label": ariaLabel,
}: TabsProps) {
  return (
    <div
      role="tablist"
      data-testid={DATA_TEST_ID.LIST}
      aria-label={ariaLabel}
      className={mergeClassNames("flex border-b border-line", className)}
    >
      {items.map((item) => {
        const active = item.id === value;
        return (
          <button
            key={item.id}
            role="tab"
            type="button"
            data-testid={DATA_TEST_ID.TAB}
            data-tab-id={item.id}
            aria-selected={active}
            disabled={item.disabled}
            onClick={() => onChange(item.id)}
            className={mergeClassNames(
              "-mb-px flex items-baseline gap-2 border-b-2 px-3.5 py-2.5 font-mono text-xs transition-colors",
              active
                ? "border-navy text-navy font-medium"
                : "border-transparent text-muted hover:text-navy",
              item.disabled && "opacity-45 cursor-not-allowed hover:text-muted",
            )}
          >
            {item.label}
            {item.meta && (
              <span className="text-[11px] text-muted-2">{item.meta}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
