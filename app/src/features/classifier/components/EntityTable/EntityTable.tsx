/**
 * Flat-table view of classified entities. One row per entity, surface form
 * + best-match + score + a row-level Open button that fires onEntityClick
 * (the page opens the detail drawer).
 *
 * Header carries a single CSV download button so the user can export
 * everything with one click.
 */

import { useTranslation } from "react-i18next";
import { Button, Icon, Table } from "@/components";
import type { ClassifiedEntity } from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";
import { EntitySwatch } from "../EntitySwatch/EntitySwatch";
import { entitiesToCsv } from "./entitiesToCsv";

const uniqueId = "1f7c4d8e-2a5b-4c9d-8e3f-6b1a2d5c8f4e";

export const DATA_TEST_ID = {
  CONTAINER: `entity-table-container-${uniqueId}`,
  DOWNLOAD_BUTTON: `entity-table-download-button-${uniqueId}`,
  ROW: `entity-table-row-${uniqueId}`,
  OPEN_BUTTON: `entity-table-open-button-${uniqueId}`,
};

export interface EntityTableProps {
  entities: ClassifiedEntity[];
  selectedEntityIndex?: number | null;
  onEntityClick?: (entity: ClassifiedEntity, entityIndex: number) => void;
  /** Filename for the CSV (without extension). */
  csvFilename?: string;
  className?: string;
}

export function EntityTable({
  entities,
  selectedEntityIndex = null,
  onEntityClick,
  csvFilename = "classification",
  className,
}: EntityTableProps) {
  const { t } = useTranslation();

  function handleDownload() {
    const csv = entitiesToCsv(entities);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${csvFilename}.csv`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames("flex min-h-0 flex-col gap-3", className)}
    >
      <div className="flex justify-end">
        <Button
          size="sm"
          variant="default"
          leading={<Icon name="download" />}
          onClick={handleDownload}
          disabled={entities.length === 0}
          data-testid={DATA_TEST_ID.DOWNLOAD_BUTTON}
        >
          {t("classifier.results.downloadCsv")}
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto rounded-md border border-line bg-paper">
        <Table>
          <Table.Head>
            <Table.Row>
              <Table.HeaderCell>
                {t("classifier.resultsTable.headerType")}
              </Table.HeaderCell>
              <Table.HeaderCell>
                {t("classifier.resultsTable.headerSurfaceForm")}
              </Table.HeaderCell>
              <Table.HeaderCell>
                {t("classifier.resultsTable.headerTopMatch")}
              </Table.HeaderCell>
              <Table.HeaderCell className="w-[80px]">
                {t("classifier.resultsTable.headerScore")}
              </Table.HeaderCell>
              <Table.HeaderCell className="w-[1%] text-right" />
            </Table.Row>
          </Table.Head>
          <Table.Body>
            {entities.map((entity, entityIndex) => {
              const topMatch = entity.matches[0];
              const isSelected = entityIndex === selectedEntityIndex;
              return (
                <Table.Row
                  key={`${entityIndex}-${entity.span.start}`}
                  data-testid={DATA_TEST_ID.ROW}
                  data-entity-index={entityIndex}
                  data-selected={isSelected ? "true" : "false"}
                  hover={Boolean(onEntityClick)}
                  className={isSelected ? "bg-cream-200" : undefined}
                  onClick={() => onEntityClick?.(entity, entityIndex)}
                >
                  <Table.Cell>
                    <span className="inline-flex items-center gap-2">
                      <EntitySwatch entityType={entity.entity_type} />
                      <span className="font-mono text-[11px] uppercase tracking-wider text-muted">
                        {t(
                          `classifier.entityTypeFilter.types.${entity.entity_type}`,
                        )}
                      </span>
                    </span>
                  </Table.Cell>
                  <Table.Cell>
                    <span className="font-mono text-[13px] text-navy">
                      {entity.surface_form}
                    </span>
                  </Table.Cell>
                  <Table.Cell>
                    <span className="text-xs text-muted">
                      {topMatch?.entity.preferred_label ??
                        t("classifier.results.noMatches")}
                    </span>
                  </Table.Cell>
                  <Table.Cell>
                    <span className="font-mono text-[11px] text-muted">
                      {topMatch
                        ? `${(topMatch.similarity_score * 100).toFixed(0)}%`
                        : "—"}
                    </span>
                  </Table.Cell>
                  <Table.Cell>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={(event) => {
                        event.stopPropagation();
                        onEntityClick?.(entity, entityIndex);
                      }}
                      data-testid={DATA_TEST_ID.OPEN_BUTTON}
                    >
                      {t("classifier.resultsTable.openButton")}
                    </Button>
                  </Table.Cell>
                </Table.Row>
              );
            })}
          </Table.Body>
        </Table>
      </div>
    </div>
  );
}
