/**
 * Filter chips for entity types: occupation / skill / qualification.
 *
 * Pure presentational toggle group. The hosting page decides what filtering
 * means — typically: dim non-selected entities in the source pane and hide
 * them from result tabs. Counts are computed by the caller against the
 * un-filtered entity set.
 */

import { useTranslation } from "react-i18next";
import type { ClassifyEntityType } from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "5e3d8a2c-7f1b-4a9d-8c5e-2b6f3d9a4e8c";

export const DATA_TEST_ID = {
  CONTAINER: `entity-type-filter-container-${uniqueId}`,
  CHIP: `entity-type-filter-chip-${uniqueId}`,
};

export const ENTITY_TYPES: readonly ClassifyEntityType[] = [
  "occupation",
  "skill",
  "qualification",
] as const;

export interface EntityTypeFilterProps {
  /** The set of entity types currently selected (visible). */
  selected: ReadonlySet<ClassifyEntityType>;
  /** Number of entities of each type, regardless of selection. */
  counts: Readonly<Record<ClassifyEntityType, number>>;
  /** Fires with the next selection set when the user toggles a chip. */
  onChange: (next: Set<ClassifyEntityType>) => void;
  /** Disabled while a run is in flight. */
  disabled?: boolean;
  className?: string;
}

export function EntityTypeFilter({
  selected,
  counts,
  onChange,
  disabled = false,
  className,
}: EntityTypeFilterProps) {
  const { t } = useTranslation();

  function toggle(type: ClassifyEntityType) {
    const next = new Set(selected);
    if (next.has(type)) {
      next.delete(type);
    } else {
      next.add(type);
    }
    onChange(next);
  }

  return (
    <div
      role="group"
      aria-label={t("classifier.entityTypeFilter.ariaLabel")}
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames("flex flex-wrap items-center gap-2", className)}
    >
      {ENTITY_TYPES.map((entityType) => {
        const isSelected = selected.has(entityType);
        const labelKey = `classifier.entityTypeFilter.types.${entityType}` as const;
        return (
          <button
            key={entityType}
            type="button"
            role="switch"
            aria-checked={isSelected}
            data-testid={DATA_TEST_ID.CHIP}
            data-entity-type={entityType}
            data-selected={isSelected ? "true" : "false"}
            disabled={disabled}
            onClick={() => toggle(entityType)}
            className={mergeClassNames(
              "inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-[11px] transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy/30",
              isSelected
                ? "border-navy bg-paper text-navy"
                : "border-line bg-transparent text-muted hover:border-line-strong hover:text-navy",
              disabled && "cursor-not-allowed opacity-45",
            )}
          >
            <span
              aria-hidden
              data-type={entityType}
              className="ent inline-block h-2 w-2 rounded-full"
              style={{ padding: 0, margin: 0 }}
            />
            <span>{t(labelKey)}</span>
            <span className="text-muted-2">{counts[entityType]}</span>
          </button>
        );
      })}
    </div>
  );
}
