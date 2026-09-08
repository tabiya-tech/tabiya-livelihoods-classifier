import { useTranslation } from "react-i18next";
import { Eyebrow } from "@/components";
import { Button } from "@/components/Button/Button";
import { EmptyState } from "@/components/EmptyState/EmptyState";
import { Spinner } from "@/components/Spinner/Spinner";
import { Table } from "@/components/Table/Table";
import { useClassifications } from "@/features/dashboard/hooks/useClassifications";
import { usePipelineNames } from "@/features/dashboard/hooks/usePipelineNames";

const uniqueId = "d7c4e891-5a23-4b67-f018-3e9a2c5d8b12";

export const DATA_TEST_ID = {
  CONTAINER: `history-page-container-${uniqueId}`,
  TABLE: `history-page-table-${uniqueId}`,
  LOAD_MORE: `history-page-load-more-${uniqueId}`,
  EMPTY: `history-page-empty-${uniqueId}`,
  SPINNER: `history-page-spinner-${uniqueId}`,
};

function formatDate(isoString: string): string {
  return new Date(isoString).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  });
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function HistoryPage() {
  const { t } = useTranslation();
  const { status, items, nextCursor, loadMore } = useClassifications({
    limit: 20,
  });
  const pipelineNames = usePipelineNames();

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className="mx-auto w-full max-w-5xl px-4 py-6 sm:px-8 sm:py-8 lg:px-10 lg:py-10"
    >
      <Eyebrow>{t("history.eyebrow")}</Eyebrow>
      <h1 className="h-page mt-2">{t("history.title")}</h1>
      <p className="mt-2 text-sm text-muted">{t("history.intro")}</p>

      <div className="mt-8">
        {status === "loading" ? (
          <div
            data-testid={DATA_TEST_ID.SPINNER}
            className="flex h-40 items-center justify-center"
          >
            <Spinner size={20} />
          </div>
        ) : items.length === 0 ? (
          <EmptyState
            data-testid={DATA_TEST_ID.EMPTY}
            title={t("history.table.empty")}
            description={t("history.table.emptyDescription")}
          />
        ) : (
          <>
            <div className="overflow-x-auto rounded-md border border-line">
              <Table data-testid={DATA_TEST_ID.TABLE}>
                <Table.Head>
                  <Table.Row>
                    <Table.HeaderCell>
                      {t("history.table.headerDate")}
                    </Table.HeaderCell>
                    <Table.HeaderCell>
                      {t("history.table.headerPipeline")}
                    </Table.HeaderCell>
                    <Table.HeaderCell>
                      {t("history.table.headerEntities")}
                    </Table.HeaderCell>
                    <Table.HeaderCell>
                      {t("history.table.headerDuration")}
                    </Table.HeaderCell>
                  </Table.Row>
                </Table.Head>
                <Table.Body>
                  {items.map((item) => (
                    <Table.Row key={item.classification_id}>
                      <Table.Cell>{formatDate(item.created_at)}</Table.Cell>
                      <Table.Cell>
                        {pipelineNames.get(item.pipeline_id) ?? (
                          <span className="font-mono text-[12px] text-muted">
                            {item.pipeline_id}
                          </span>
                        )}
                      </Table.Cell>
                      <Table.Cell>{item.entity_count}</Table.Cell>
                      <Table.Cell className="text-muted">
                        {formatDuration(item.processing_time_ms)}
                      </Table.Cell>
                    </Table.Row>
                  ))}
                </Table.Body>
              </Table>
            </div>
            {nextCursor && (
              <div className="mt-4 flex justify-center">
                <Button
                  data-testid={DATA_TEST_ID.LOAD_MORE}
                  variant="ghost"
                  onClick={loadMore}
                >
                  {t("history.loadMore")}
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
