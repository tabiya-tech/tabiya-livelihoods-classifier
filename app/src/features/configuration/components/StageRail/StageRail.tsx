/**
 * Left rail for the Configuration page. Renders a vertical list of numbered
 * stage cards (01 NEL, 02 Taxonomy …) — clicking one switches the active
 * stage in the page's right panel.
 *
 * Stages are passed in as data so the page owns the set (and a new stage
 * can land later by adding an entry; the rail itself stays unchanged).
 */

import type { ReactNode } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "a2c39f1a-6d5e-4a8b-9c4f-2e7b1d8a5c3e";

export const DATA_TEST_ID = {
  CONTAINER: `stage-rail-container-${uniqueId}`,
  ITEM: `stage-rail-item-${uniqueId}`,
  ITEM_NUMBER: `stage-rail-item-number-${uniqueId}`,
  ITEM_LABEL: `stage-rail-item-label-${uniqueId}`,
  ITEM_SUB_LABEL: `stage-rail-item-sub-label-${uniqueId}`,
  ITEM_CURRENT_VALUE: `stage-rail-item-current-value-${uniqueId}`,
};

export interface StageRailItem<TStageId extends string = string> {
  id: TStageId;
  /** Sequence indicator shown in the circular badge (e.g. "01"). */
  number: string;
  /** Primary stage label (e.g. "NEL"). */
  label: ReactNode;
  /** One-line description under the label (e.g. "Entity linking"). */
  subLabel?: ReactNode;
  /** Compact display of the current persisted selection (e.g. the model name). */
  currentValue?: ReactNode;
}

export interface StageRailProps<TStageId extends string = string> {
  items: StageRailItem<TStageId>[];
  activeId: TStageId;
  onSelect: (stageId: TStageId) => void;
  /** Accessible label for the navigation landmark. */
  "aria-label"?: string;
  className?: string;
}

export function StageRail<TStageId extends string = string>({
  items,
  activeId,
  onSelect,
  "aria-label": ariaLabel,
  className,
}: StageRailProps<TStageId>) {
  return (
    <nav
      data-testid={DATA_TEST_ID.CONTAINER}
      aria-label={ariaLabel}
      className={mergeClassNames(
        "sticky top-6 flex flex-col gap-1 self-start",
        className,
      )}
    >
      {items.map((stage) => {
        const isActive = stage.id === activeId;
        return (
          <button
            key={stage.id}
            type="button"
            data-testid={DATA_TEST_ID.ITEM}
            data-stage-id={stage.id}
            aria-current={isActive ? "step" : undefined}
            onClick={() => onSelect(stage.id)}
            className={mergeClassNames(
              "grid w-full grid-cols-[auto_1fr] items-start gap-3 rounded-md border p-3.5 text-left",
              "transition-colors outline-none",
              "focus-visible:ring-2 focus-visible:ring-navy/30",
              isActive
                ? "border-line bg-paper"
                : "border-transparent hover:bg-paper",
            )}
          >
            <span
              data-testid={DATA_TEST_ID.ITEM_NUMBER}
              className={mergeClassNames(
                "inline-flex h-[26px] w-[26px] flex-shrink-0 items-center justify-center rounded-full",
                "border font-mono text-[11px] tracking-[0.04em]",
                isActive
                  ? "border-navy bg-navy text-cream"
                  : "border-line bg-cream text-muted",
              )}
            >
              {stage.number}
            </span>
            <span className="flex min-w-0 flex-col gap-0.5">
              <span
                data-testid={DATA_TEST_ID.ITEM_LABEL}
                className="font-mono text-[13px] font-medium text-navy"
              >
                {stage.label}
              </span>
              {stage.subLabel && (
                <span
                  data-testid={DATA_TEST_ID.ITEM_SUB_LABEL}
                  className="text-[11.5px] tracking-[0.01em] text-muted"
                >
                  {stage.subLabel}
                </span>
              )}
              {stage.currentValue && (
                <span
                  data-testid={DATA_TEST_ID.ITEM_CURRENT_VALUE}
                  className={mergeClassNames(
                    "mt-1 overflow-hidden text-ellipsis whitespace-nowrap font-mono text-[11px]",
                    isActive ? "text-navy" : "text-muted-2",
                  )}
                >
                  {stage.currentValue}
                </span>
              )}
            </span>
          </button>
        );
      })}
    </nav>
  );
}
