/**
 * Right pane container. Three tabs (Entities, Table, JSON) over the same
 * classify response. Selected tab is controlled by the parent so it
 * survives re-renders from filter or selection changes.
 */

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Tabs, type TabItem } from "@/components";
import type {
  ClassifiedEntity,
  ClassifyEntityType,
  ClassifyResponse,
} from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";
import { ENTITY_TYPES } from "../EntityTypeFilter/EntityTypeFilter";
import {
  EntityGroupCard,
  type EntityGroupCardEntry,
} from "../EntityGroupCard/EntityGroupCard";
import { EntityTable } from "../EntityTable/EntityTable";
import { JsonView } from "../JsonView/JsonView";

const uniqueId = "7b3d8a5e-2c1f-4d9c-8a4e-6b2d5f8c1a3e";

export const DATA_TEST_ID = {
  CONTAINER: `results-tabs-container-${uniqueId}`,
  PANEL_ENTITIES: `results-tabs-panel-entities-${uniqueId}`,
  PANEL_TABLE: `results-tabs-panel-table-${uniqueId}`,
  PANEL_JSON: `results-tabs-panel-json-${uniqueId}`,
  EMPTY_STATE: `results-tabs-empty-state-${uniqueId}`,
};

export type ResultsTabId = "entities" | "table" | "json";

export interface ResultsTabsProps {
  response: ClassifyResponse;
  /** Entity types currently visible — un-selected types are hidden from views. */
  selectedEntityTypes: ReadonlySet<ClassifyEntityType>;
  activeTabId: ResultsTabId;
  onActiveTabChange: (next: ResultsTabId) => void;
  selectedEntityIndex?: number | null;
  onEntityClick?: (entity: ClassifiedEntity, entityIndex: number) => void;
  className?: string;
}

function partitionByType(
  entities: ClassifiedEntity[],
  visibleTypes: ReadonlySet<ClassifyEntityType>,
): Record<ClassifyEntityType, EntityGroupCardEntry[]> {
  const initial: Record<ClassifyEntityType, EntityGroupCardEntry[]> = {
    occupation: [],
    skill: [],
    qualification: [],
  };
  entities.forEach((entity, entityIndex) => {
    if (!visibleTypes.has(entity.entity_type)) return;
    initial[entity.entity_type].push({ entity, entityIndex });
  });
  return initial;
}

export function ResultsTabs({
  response,
  selectedEntityTypes,
  activeTabId,
  onActiveTabChange,
  selectedEntityIndex = null,
  onEntityClick,
  className,
}: ResultsTabsProps) {
  const { t } = useTranslation();

  const visibleEntities = useMemo(
    () =>
      response.entities.filter((entity) =>
        selectedEntityTypes.has(entity.entity_type),
      ),
    [response.entities, selectedEntityTypes],
  );

  const grouped = useMemo(
    () => partitionByType(response.entities, selectedEntityTypes),
    [response.entities, selectedEntityTypes],
  );

  const tabItems: TabItem[] = [
    {
      id: "entities",
      label: t("classifier.resultsTabs.entities"),
      meta: visibleEntities.length,
    },
    {
      id: "table",
      label: t("classifier.resultsTabs.table"),
    },
    {
      id: "json",
      label: t("classifier.resultsTabs.json"),
    },
  ];

  const hasVisibleEntities = visibleEntities.length > 0;

  return (
    <section
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames("flex flex-col gap-4", className)}
    >
      <Tabs
        items={tabItems}
        value={activeTabId}
        onChange={(next) => onActiveTabChange(next as ResultsTabId)}
        aria-label={t("classifier.resultsTabs.ariaLabel")}
      />

      {activeTabId === "entities" && (
        <div
          data-testid={DATA_TEST_ID.PANEL_ENTITIES}
          role="tabpanel"
          className="flex flex-col gap-3"
        >
          {hasVisibleEntities ? (
            ENTITY_TYPES.filter((type) => grouped[type].length > 0).map(
              (type) => (
                <EntityGroupCard
                  key={type}
                  entityType={type}
                  entries={grouped[type]}
                  selectedEntityIndex={selectedEntityIndex}
                  onEntityClick={onEntityClick}
                />
              ),
            )
          ) : (
            <ResultsEmptyState />
          )}
        </div>
      )}

      {activeTabId === "table" && (
        <div data-testid={DATA_TEST_ID.PANEL_TABLE} role="tabpanel">
          {hasVisibleEntities ? (
            <EntityTable
              entities={visibleEntities}
              selectedEntityIndex={selectedEntityIndex}
              onEntityClick={onEntityClick}
            />
          ) : (
            <ResultsEmptyState />
          )}
        </div>
      )}

      {activeTabId === "json" && (
        <div data-testid={DATA_TEST_ID.PANEL_JSON} role="tabpanel">
          <JsonView value={response} />
        </div>
      )}
    </section>
  );
}

function ResultsEmptyState() {
  const { t } = useTranslation();
  return (
    <div
      data-testid={DATA_TEST_ID.EMPTY_STATE}
      className="rounded-md border border-dashed border-line bg-paper px-6 py-10 text-center"
    >
      <p className="m-0 font-mono text-xs text-muted">
        {t("classifier.results.emptyTitle")}
      </p>
      <p className="m-0 mt-1 text-xs text-muted-2">
        {t("classifier.results.emptyDescription")}
      </p>
    </div>
  );
}
