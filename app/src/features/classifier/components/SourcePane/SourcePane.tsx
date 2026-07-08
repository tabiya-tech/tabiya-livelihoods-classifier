/**
 * Left pane of the Classifier workspace. Composes:
 *   - Source text input (Textarea before a run; EntityHighlight after)
 *   - Drag-drop file upload
 *   - PipelineParamsCard (top_k + min_similarity sliders)
 *   - EntityTypeFilter chips with live counts
 *   - Run / Clear buttons
 *
 * Pure composition — every piece of state lives on the page; this component
 * only re-emits user intents back up.
 */

import { useMemo, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Button, Textarea } from "@/components";
import type { ClassifiedEntity, ClassifyEntityType } from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";
import { EntityHighlight } from "../EntityHighlight/EntityHighlight";
import {
  ENTITY_TYPES,
  EntityTypeFilter,
} from "../EntityTypeFilter/EntityTypeFilter";
import { PipelineParamsCard } from "../PipelineParamsCard/PipelineParamsCard";
import { UploadDropzone } from "../UploadDropzone/UploadDropzone";

const uniqueId = "4c8d1f5e-7b3a-4c9d-8e2f-6a1b3d8c4e7f";

export const DATA_TEST_ID = {
  CONTAINER: `source-pane-container-${uniqueId}`,
  TEXTAREA: `source-pane-textarea-${uniqueId}`,
  HIGHLIGHT: `source-pane-highlight-${uniqueId}`,
  RUN_BUTTON: `source-pane-run-button-${uniqueId}`,
  CLEAR_BUTTON: `source-pane-clear-button-${uniqueId}`,
  CONFIG_CHIP: `source-pane-config-chip-${uniqueId}`,
};

export interface SourcePaneProps {
  text: string;
  onTextChange: (next: string) => void;
  /** Entities to render inline after a successful run. */
  entities: ClassifiedEntity[] | null;
  /** Currently visible entity types — non-selected types are dimmed inline. */
  selectedEntityTypes: ReadonlySet<ClassifyEntityType>;
  onSelectedEntityTypesChange: (next: Set<ClassifyEntityType>) => void;
  topK: number;
  minSimilarity: number;
  onTopKChange: (value: number) => void;
  onMinSimilarityChange: (value: number) => void;
  isRunning: boolean;
  canRun: boolean;
  onRun: () => void;
  onClear: () => void;
  /** Active configuration chip — shown above the run button. Optional. */
  activeConfigSlot?: ReactNode;
  /** Entity index currently selected (for the focus ring inline). */
  selectedEntityIndex?: number | null;
  onEntitySelect?: (index: number | null) => void;
  className?: string;
}

function countByType(
  entities: ClassifiedEntity[] | null,
): Record<ClassifyEntityType, number> {
  // Seed every known type at zero so the chip row always renders the same
  // set in the same order, even when the backend returns no entities of a
  // given type.
  const counts: Record<ClassifyEntityType, number> = {
    occupation: 0,
    skill: 0,
    qualification: 0,
    experience: 0,
    domain: 0,
  };
  if (!entities) return counts;
  for (const entity of entities) {
    // Defensive: a future entity type the model adds shouldn't crash the page.
    if (entity.entity_type in counts) counts[entity.entity_type] += 1;
  }
  return counts;
}

function dimmedIndicesForFilter(
  entities: ClassifiedEntity[] | null,
  selectedTypes: ReadonlySet<ClassifyEntityType>,
): Set<number> {
  const dimmed = new Set<number>();
  if (!entities) return dimmed;
  entities.forEach((entity, index) => {
    if (!selectedTypes.has(entity.entity_type)) dimmed.add(index);
  });
  return dimmed;
}

export function SourcePane({
  text,
  onTextChange,
  entities,
  selectedEntityTypes,
  onSelectedEntityTypesChange,
  topK,
  minSimilarity,
  onTopKChange,
  onMinSimilarityChange,
  isRunning,
  canRun,
  onRun,
  onClear,
  activeConfigSlot,
  selectedEntityIndex = null,
  onEntitySelect,
  className,
}: SourcePaneProps) {
  const { t } = useTranslation();
  const [isDirtySinceRun, setIsDirtySinceRun] = useState(false);

  const counts = useMemo(() => countByType(entities), [entities]);
  const dimmedIndices = useMemo(
    () => dimmedIndicesForFilter(entities, selectedEntityTypes),
    [entities, selectedEntityTypes],
  );

  const showHighlight = entities !== null && !isDirtySinceRun;

  return (
    <section
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames("flex min-h-0 flex-col gap-4", className)}
    >
      {activeConfigSlot && (
        <div data-testid={DATA_TEST_ID.CONFIG_CHIP}>{activeConfigSlot}</div>
      )}

      {showHighlight ? (
        <div
          data-testid={DATA_TEST_ID.HIGHLIGHT}
          className="min-h-0 flex-1 overflow-y-auto rounded-md border border-line bg-paper px-4 py-3"
        >
          <EntityHighlight
            text={text}
            entities={entities ?? []}
            selectedEntityIndex={selectedEntityIndex}
            dimmedEntityIndices={dimmedIndices}
            onEntityClick={(_entity, entityIndex) =>
              onEntitySelect?.(
                selectedEntityIndex === entityIndex ? null : entityIndex,
              )
            }
          />
        </div>
      ) : (
        <Textarea
          data-testid={DATA_TEST_ID.TEXTAREA}
          mono
          className="min-h-0 flex-1 resize-none"
          placeholder={t("classifier.source.placeholder")}
          value={text}
          disabled={isRunning}
          onChange={(event) => {
            onTextChange(event.target.value);
            if (entities !== null) setIsDirtySinceRun(true);
          }}
        />
      )}

      <UploadDropzone
        disabled={isRunning}
        onText={(fileText) => {
          onTextChange(fileText);
          if (entities !== null) setIsDirtySinceRun(true);
        }}
      />

      <PipelineParamsCard
        topK={topK}
        minSimilarity={minSimilarity}
        onTopKChange={onTopKChange}
        onMinSimilarityChange={onMinSimilarityChange}
        disabled={isRunning}
      />

      {entities && (
        <EntityTypeFilter
          selected={selectedEntityTypes}
          counts={counts}
          onChange={onSelectedEntityTypesChange}
          disabled={isRunning}
        />
      )}

      <div className="flex items-center gap-2">
        <Button
          variant="primary"
          loading={isRunning}
          disabled={!canRun || isRunning}
          onClick={() => {
            setIsDirtySinceRun(false);
            onRun();
          }}
          data-testid={DATA_TEST_ID.RUN_BUTTON}
        >
          {t("classifier.source.runButton")}
        </Button>
        <Button
          variant="ghost"
          disabled={isRunning || (text.length === 0 && entities === null)}
          onClick={() => {
            setIsDirtySinceRun(false);
            onClear();
          }}
          data-testid={DATA_TEST_ID.CLEAR_BUTTON}
        >
          {t("classifier.source.clearButton")}
        </Button>
      </div>
    </section>
  );
}

export { ENTITY_TYPES };
