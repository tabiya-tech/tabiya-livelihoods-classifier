import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { Eyebrow, StatCard, UsageChart } from "@/components";
import { Card, CardHead } from "@/components/Card/Card";
import { Spinner } from "@/components/Spinner/Spinner";
import { Table } from "@/components/Table/Table";
import { useFirebaseAuth } from "@/lib/auth/useFirebaseAuth";
import { useUsage } from "../hooks/useUsage";
import { useClassifications } from "../hooks/useClassifications";
import { useApiKeys } from "@/features/keys/hooks/useApiKeys";
import { usePipelineNames } from "../hooks/usePipelineNames";
import { routerPaths } from "@/routes/routerPaths";

const uniqueId = "f0e3a6a2-2b18-4f0b-9d8b-7d2c9f1a8b3e";

export const DATA_TEST_ID = {
  CONTAINER: `dashboard-page-container-${uniqueId}`,
  WELCOME_MESSAGE: `dashboard-page-welcome-${uniqueId}`,
  STAT_CALLS: `dashboard-page-stat-calls-${uniqueId}`,
  STAT_KEYS: `dashboard-page-stat-keys-${uniqueId}`,
  USAGE_SECTION: `dashboard-page-usage-section-${uniqueId}`,
  RECENT_SECTION: `dashboard-page-recent-section-${uniqueId}`,
  RECENT_TABLE: `dashboard-page-recent-table-${uniqueId}`,
};

function formatDate(isoString: string): string {
  return new Date(isoString).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function DashboardPage() {
  const { t } = useTranslation();
  const { user } = useFirebaseAuth();
  const displayName = user?.email ?? t("dashboard.fallbackName");

  const usage = useUsage({ days: 30 });
  const classifications = useClassifications({ limit: 5 });
  const apiKeys = useApiKeys();
  const pipelineNames = usePipelineNames();

  // Derive calls-this-week and delta from the usage data.
  // The backend omits zero-count days, so slice(-7) would count the last 7
  // non-empty entries — not the last 7 calendar days. Use UTC date strings
  // instead so today always counts, even when adjacent days had zero calls.
  const todayUtc = new Date().toISOString().slice(0, 10);
  const sevenDaysAgoUtc = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
    .toISOString()
    .slice(0, 10);
  const fourteenDaysAgoUtc = new Date(Date.now() - 14 * 24 * 60 * 60 * 1000)
    .toISOString()
    .slice(0, 10);

  const callsThisWeek = usage.data
    .filter((entry) => entry.date > sevenDaysAgoUtc && entry.date <= todayUtc)
    .reduce((sum, entry) => sum + entry.count, 0);
  const callsLastWeek = usage.data
    .filter(
      (entry) =>
        entry.date > fourteenDaysAgoUtc && entry.date <= sevenDaysAgoUtc
    )
    .reduce((sum, entry) => sum + entry.count, 0);
  const delta = callsThisWeek - callsLastWeek;
  const deltaText = t("dashboard.stats.callsDelta", {
    delta: delta >= 0 ? `+${delta}` : String(delta),
  });

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className="mx-auto w-full max-w-5xl px-4 py-6 sm:px-8 sm:py-8 lg:px-10 lg:py-10"
    >
      <Eyebrow>{t("dashboard.eyebrow")}</Eyebrow>
      <h1 data-testid={DATA_TEST_ID.WELCOME_MESSAGE} className="h-page mt-2">
        {t("dashboard.welcomeBack", { name: displayName })}
      </h1>

      {/* Stat cards */}
      <div className="mt-8 grid grid-cols-2 gap-4 sm:grid-cols-2 lg:grid-cols-2">
        <StatCard
          data-testid={DATA_TEST_ID.STAT_CALLS}
          label={t("dashboard.stats.callsThisWeek")}
          value={
            usage.status === "loading" ? (
              <Spinner size={20} />
            ) : (
              callsThisWeek
            )
          }
          delta={usage.status === "ready" ? deltaText : undefined}
          deltaPositive={delta >= 0}
        />
        <StatCard
          data-testid={DATA_TEST_ID.STAT_KEYS}
          label={t("dashboard.stats.activeKeys")}
          value={
            apiKeys.status === "loading" ? (
              <Spinner size={20} />
            ) : (
              apiKeys.keys.length
            )
          }
        />
      </div>

      {/* Usage chart */}
      <Card className="mt-6" data-testid={DATA_TEST_ID.USAGE_SECTION}>
        <CardHead title={t("dashboard.usageChart.title")} />
        {usage.status === "loading" ? (
          <div className="flex h-40 items-center justify-center">
            <Spinner size={20} />
          </div>
        ) : (
          <UsageChart data={usage.data} />
        )}
      </Card>

      {/* Recent classifications */}
      <Card className="mt-6" flush data-testid={DATA_TEST_ID.RECENT_SECTION}>
        <CardHead
          className="px-5 pt-5"
          title={t("dashboard.recentClassifications.title")}
          action={
            <Link
              to={routerPaths.HISTORY}
              className="font-mono text-[11px] text-navy underline-offset-2 hover:underline"
            >
              {t("dashboard.recentClassifications.viewAll")}
            </Link>
          }
        />
        {classifications.status === "loading" ? (
          <div className="flex h-24 items-center justify-center">
            <Spinner size={16} />
          </div>
        ) : classifications.items.length === 0 ? (
          <p className="px-5 pb-5 font-mono text-[13px] text-muted">
            {t("dashboard.recentClassifications.empty")}
          </p>
        ) : (
          <div
            data-testid={DATA_TEST_ID.RECENT_TABLE}
            className="overflow-x-auto"
          >
            <Table>
              <Table.Head>
                <Table.Row>
                  <Table.HeaderCell>{t("history.table.headerDate")}</Table.HeaderCell>
                  <Table.HeaderCell>{t("history.table.headerPipeline")}</Table.HeaderCell>
                  <Table.HeaderCell>{t("history.table.headerEntities")}</Table.HeaderCell>
                  <Table.HeaderCell>{t("history.table.headerDuration")}</Table.HeaderCell>
                </Table.Row>
              </Table.Head>
              <Table.Body>
                {classifications.items.map((item) => (
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
        )}
      </Card>
    </div>
  );
}
