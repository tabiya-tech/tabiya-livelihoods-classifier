/**
 * Single entity preview row. Shows surface form (mono), best-match label,
 * top similarity score (bar + percent). Clickable — the page opens the
 * detail drawer for the clicked entity.
 *
 * Pure presentation; selection is owned by the page.
 */

import { useTranslation } from "react-i18next";
import { ScoreBar } from "@/components";
import type { ClassifiedEntity } from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";
import { EntitySwatch } from "../EntitySwatch/EntitySwatch";

const uniqueId = "8f1c4d7a-3e9b-4c5d-9a2f-1b6e3d8c5a4f";

export const DATA_TEST_ID = {
  CONTAINER: `entity-row-container-${uniqueId}`,
  SURFACE_FORM: `entity-row-surface-form-${uniqueId}`,
  TOP_MATCH_LABEL: `entity-row-top-match-label-${uniqueId}`,
  TOP_MATCH_SCORE: `entity-row-top-match-score-${uniqueId}`,
  EMPTY_MATCH_NOTE: `entity-row-empty-match-note-${uniqueId}`,
};

export interface EntityRowProps {
  entity: ClassifiedEntity;
  /** Stable index in the un-filtered entity array — passed back via onClick. */
  entityIndex: number;
  isSelected?: boolean;
  onClick?: (entity: ClassifiedEntity, entityIndex: number) => void;
  className?: string;
}

export function EntityRow({
  entity,
  entityIndex,
  isSelected = false,
  onClick,
  className,
}: EntityRowProps) {
  const { t } = useTranslation();
  const topMatch = entity.matches[0];

  return (
    <button
      type="button"
      data-testid={DATA_TEST_ID.CONTAINER}
      data-entity-index={entityIndex}
      aria-pressed={isSelected}
      onClick={() => onClick?.(entity, entityIndex)}
      className={mergeClassNames(
        "flex w-full items-center gap-3 rounded-md border px-3 py-2.5 text-left transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy/30",
        isSelected
          ? "border-navy bg-paper"
          : "border-line bg-paper hover:border-line-strong",
        className,
      )}
    >
      <EntitySwatch entityType={entity.entity_type} />
      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span
          data-testid={DATA_TEST_ID.SURFACE_FORM}
          className="font-mono text-[13px] text-navy"
        >
          {entity.surface_form}
        </span>
        {topMatch ? (
          <span
            data-testid={DATA_TEST_ID.TOP_MATCH_LABEL}
            className="overflow-hidden text-ellipsis whitespace-nowrap text-xs text-muted"
          >
            {topMatch.entity.preferred_label}
          </span>
        ) : (
          <span
            data-testid={DATA_TEST_ID.EMPTY_MATCH_NOTE}
            className="text-xs italic text-muted-2"
          >
            {t("classifier.results.noMatches")}
          </span>
        )}
      </span>
      {topMatch && (
        <span
          data-testid={DATA_TEST_ID.TOP_MATCH_SCORE}
          className="flex w-[110px] shrink-0 items-center gap-2 font-mono text-[11px] text-muted"
        >
          <ScoreBar score={topMatch.similarity_score} className="flex-1" />
          {(topMatch.similarity_score * 100).toFixed(0)}%
        </span>
      )}
    </button>
  );
}
