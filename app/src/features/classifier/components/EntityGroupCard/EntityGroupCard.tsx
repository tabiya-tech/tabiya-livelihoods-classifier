/**
 * Grouped entity card: header (type + count) followed by a stack of rows.
 *
 * The page builds one of these per visible entity type. Pure presentation
 * — selection / click handling is forwarded.
 */

import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { ClassifiedEntity, ClassifyEntityType } from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";
import { EntityRow } from "../EntityRow/EntityRow";
import { EntitySwatch } from "../EntitySwatch/EntitySwatch";

const uniqueId = "6c2e8f4d-1b7a-4d9c-8e5f-3a2d6c9b1e4f";

/** Rows shown before the card must be expanded to reveal the rest. */
export const COLLAPSED_ROW_LIMIT = 5;

export const DATA_TEST_ID = {
  CONTAINER: `entity-group-card-container-${uniqueId}`,
  HEADER: `entity-group-card-header-${uniqueId}`,
  LABEL: `entity-group-card-label-${uniqueId}`,
  COUNT: `entity-group-card-count-${uniqueId}`,
  ROWS: `entity-group-card-rows-${uniqueId}`,
  TOGGLE: `entity-group-card-toggle-${uniqueId}`,
};

export interface EntityGroupCardEntry {
  entity: ClassifiedEntity;
  /** Stable index in the un-filtered entity array. */
  entityIndex: number;
}

export interface EntityGroupCardProps {
  entityType: ClassifyEntityType;
  entries: EntityGroupCardEntry[];
  selectedEntityIndex?: number | null;
  onEntityClick?: (entity: ClassifiedEntity, entityIndex: number) => void;
  showMatches?: boolean;
  className?: string;
}

export function EntityGroupCard({
  entityType,
  entries,
  selectedEntityIndex = null,
  onEntityClick,
  showMatches = true,
  className,
}: EntityGroupCardProps) {
  const { t } = useTranslation();
  const labelKey = `classifier.entityTypeFilter.types.${entityType}` as const;

  // Collapsed by default: show at most COLLAPSED_ROW_LIMIT rows until the
  // user expands. Keeps long occupation/skill lists from dominating the pane.
  const [isExpanded, setIsExpanded] = useState(false);
  const isOverflowing = entries.length > COLLAPSED_ROW_LIMIT;
  const visibleEntries =
    isExpanded || !isOverflowing
      ? entries
      : entries.slice(0, COLLAPSED_ROW_LIMIT);
  const hiddenCount = entries.length - COLLAPSED_ROW_LIMIT;

  return (
    <section
      data-testid={DATA_TEST_ID.CONTAINER}
      data-entity-type={entityType}
      className={mergeClassNames(
        "flex flex-col gap-3 rounded-md border border-line bg-cream px-4 py-3.5",
        className,
      )}
    >
      <header
        data-testid={DATA_TEST_ID.HEADER}
        className="flex items-baseline justify-between"
      >
        <span
          data-testid={DATA_TEST_ID.LABEL}
          className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.08em] text-muted"
        >
          <EntitySwatch entityType={entityType} />
          {t(labelKey)}
        </span>
        <span
          data-testid={DATA_TEST_ID.COUNT}
          className="font-mono text-[11px] text-muted-2"
        >
          {entries.length}
        </span>
      </header>
      <div
        data-testid={DATA_TEST_ID.ROWS}
        className="flex flex-col gap-2"
      >
        {visibleEntries.map(({ entity, entityIndex }) => (
          <EntityRow
            key={`${entityIndex}-${entity.span.start}`}
            entity={entity}
            entityIndex={entityIndex}
            isSelected={entityIndex === selectedEntityIndex}
            onClick={showMatches ? onEntityClick : undefined}
            showMatches={showMatches}
          />
        ))}
      </div>
      {isOverflowing && (
        <button
          type="button"
          data-testid={DATA_TEST_ID.TOGGLE}
          aria-expanded={isExpanded}
          onClick={() => setIsExpanded((prev) => !prev)}
          className={mergeClassNames(
            "self-start rounded font-mono text-[11px] uppercase tracking-[0.08em] text-navy",
            "transition-colors hover:text-ink focus-visible:outline-none",
            "focus-visible:ring-[3px] focus-visible:ring-navy/10",
          )}
        >
          {isExpanded
            ? t("classifier.entityGroup.showLess")
            : t("classifier.entityGroup.showMore", { count: hiddenCount })}
        </button>
      )}
    </section>
  );
}
