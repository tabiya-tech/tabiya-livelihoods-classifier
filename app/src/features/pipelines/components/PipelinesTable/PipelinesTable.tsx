/**
 * Tabular listing of the caller's pipelines. Each row exposes activate,
 * clone, and delete actions. Clone and delete are disabled for readonly
 * pipelines. All controls are disabled while a request is in-flight for
 * that row.
 *
 * Pure presentation; the snapshot lives in usePipelinesList.
 */

import { useTranslation } from "react-i18next";
import { Button, Icon, Table, Tag, Toggle } from "@/components";
import type { Pipeline } from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "b3c4d5e6-f7a8-4b9c-8d0e-1f2a3b4c5d6e";

export const DATA_TEST_ID = {
  CONTAINER: `pipelines-table-container-${uniqueId}`,
  ROW: `pipelines-table-row-${uniqueId}`,
  NAME_CELL: `pipelines-table-name-cell-${uniqueId}`,
  DEFAULT_BADGE: `pipelines-table-default-badge-${uniqueId}`,
  ACTIVE_BADGE: `pipelines-table-active-badge-${uniqueId}`,
  ACTIVE_TOGGLE: `pipelines-table-active-toggle-${uniqueId}`,
  UPDATED_CELL: `pipelines-table-updated-cell-${uniqueId}`,
  EDIT_BUTTON: `pipelines-table-edit-button-${uniqueId}`,
  CLONE_BUTTON: `pipelines-table-clone-button-${uniqueId}`,
  DELETE_BUTTON: `pipelines-table-delete-button-${uniqueId}`,
};

export interface PipelinesTableProps {
  pipelines: Pipeline[];
  /** pipeline_id whose request is in-flight, or null. */
  pendingId: string | null;
  onActivate: (pipelineId: string) => void;
  onEdit?: (pipelineId: string) => void;
  onClone: (pipelineId: string) => void;
  onDelete: (pipelineId: string) => void;
  className?: string;
}

function formatIsoTimestamp(isoString: string, locale: string): string {
  return new Intl.DateTimeFormat(locale, {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(new Date(isoString));
}

export function PipelinesTable({
  pipelines,
  pendingId,
  onActivate,
  onEdit,
  onClone,
  onDelete,
  className,
}: PipelinesTableProps) {
  const { t, i18n } = useTranslation();
  const locale = i18n.language;

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "overflow-hidden rounded-md border border-line bg-paper",
        className,
      )}
    >
      <Table>
        <Table.Head>
          <Table.Row>
            <Table.HeaderCell>{t("pipelines.list.table.headerName")}</Table.HeaderCell>
            <Table.HeaderCell>{t("pipelines.list.table.headerActive")}</Table.HeaderCell>
            <Table.HeaderCell>{t("pipelines.list.table.headerUpdated")}</Table.HeaderCell>
            <Table.HeaderCell className="w-[1%] text-right">
              {t("pipelines.list.table.headerActions")}
            </Table.HeaderCell>
          </Table.Row>
        </Table.Head>
        <Table.Body>
          {pipelines.map((pipeline) => {
            const isPending = pipeline.pipeline_id === pendingId;
            const isActionDisabled = pipeline.is_readonly || isPending;

            return (
              <Table.Row
                key={pipeline.pipeline_id}
                data-testid={DATA_TEST_ID.ROW}
                data-pipeline-id={pipeline.pipeline_id}
              >
                <Table.Cell data-testid={DATA_TEST_ID.NAME_CELL}>
                  <span className="flex items-center gap-2">
                    <span className="font-medium text-navy text-[13px]">
                      {pipeline.name}
                    </span>
                    {pipeline.is_default && (
                      <Tag
                        tone="teal"
                        size="sm"
                        data-testid={DATA_TEST_ID.DEFAULT_BADGE}
                      >
                        {t("pipelines.list.badges.default")}
                      </Tag>
                    )}
                    {pipeline.is_active && (
                      <Tag
                        tone="lime"
                        size="sm"
                        data-testid={DATA_TEST_ID.ACTIVE_BADGE}
                      >
                        {t("pipelines.list.tabs.activeBadge")}
                      </Tag>
                    )}
                  </span>
                </Table.Cell>
                <Table.Cell>
                  <Toggle
                    checked={pipeline.is_active}
                    disabled={isPending}
                    label={t("pipelines.list.actions.activate")}
                    onChange={() => onActivate(pipeline.pipeline_id)}
                    data-testid={DATA_TEST_ID.ACTIVE_TOGGLE}
                    data-pipeline-id={pipeline.pipeline_id}
                  />
                </Table.Cell>
                <Table.Cell data-testid={DATA_TEST_ID.UPDATED_CELL}>
                  <span className="text-xs text-muted">
                    {formatIsoTimestamp(pipeline.updated_at, locale)}
                  </span>
                </Table.Cell>
                <Table.Cell className="text-right">
                  <span className="flex items-center justify-end gap-2">
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={isPending}
                      onClick={() => onEdit?.(pipeline.pipeline_id)}
                      data-testid={DATA_TEST_ID.EDIT_BUTTON}
                      data-pipeline-id={pipeline.pipeline_id}
                      leading={<Icon name="arrowRight" size={12} />}
                    >
                      {t("pipelines.list.actions.edit")}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={isActionDisabled}
                      onClick={() => onClone(pipeline.pipeline_id)}
                      data-testid={DATA_TEST_ID.CLONE_BUTTON}
                      data-pipeline-id={pipeline.pipeline_id}
                      leading={<Icon name="copy" size={12} />}
                    >
                      {t("pipelines.list.actions.clone")}
                    </Button>
                    <Button
                      size="sm"
                      variant="danger"
                      disabled={isActionDisabled}
                      onClick={() => onDelete(pipeline.pipeline_id)}
                      data-testid={DATA_TEST_ID.DELETE_BUTTON}
                      data-pipeline-id={pipeline.pipeline_id}
                      leading={<Icon name="trash" size={12} />}
                    >
                      {t("pipelines.list.actions.delete")}
                    </Button>
                  </span>
                </Table.Cell>
              </Table.Row>
            );
          })}
        </Table.Body>
      </Table>
    </div>
  );
}
