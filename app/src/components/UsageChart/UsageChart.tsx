import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { useTranslation } from "react-i18next";
import { colors } from "@/theme/theme";
import type { DailyCount } from "@/lib/api";
import { EmptyState } from "@/components/EmptyState/EmptyState";

const uniqueId = "b4e9c127-3f81-4a56-8e03-6d1f7a2c9b84";

export const DATA_TEST_ID = {
  CONTAINER: `usage-chart-container-${uniqueId}`,
  CHART: `usage-chart-chart-${uniqueId}`,
  EMPTY: `usage-chart-empty-${uniqueId}`,
};

export interface UsageChartProps {
  data: DailyCount[];
  className?: string;
}

function formatAxisDate(dateStr: string): string {
  const date = new Date(dateStr + "T00:00:00Z");
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

export function UsageChart({ data, className }: UsageChartProps) {
  const { t } = useTranslation();

  if (data.length === 0) {
    return (
      <div data-testid={DATA_TEST_ID.EMPTY} className={className}>
        <EmptyState
          title={t("dashboard.usageChart.emptyTitle")}
          description={t("dashboard.usageChart.emptyDescription")}
        />
      </div>
    );
  }

  const chartData = data.map((entry) => ({
    date: formatAxisDate(entry.date),
    count: entry.count,
  }));

  // Show a tick every ~7 days to avoid crowding on 30-day view.
  const tickInterval = Math.max(0, Math.floor(data.length / 5) - 1);

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={className}
      style={{ height: 160 }}
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data-testid={DATA_TEST_ID.CHART}
          data={chartData}
          margin={{ top: 4, right: 4, left: -20, bottom: 0 }}
          barCategoryGap="30%"
        >
          <CartesianGrid
            vertical={false}
            stroke={colors.line.DEFAULT}
            strokeDasharray="3 3"
          />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: colors.muted.DEFAULT, fontFamily: "IBM Plex Mono, monospace" }}
            tickLine={false}
            axisLine={false}
            interval={tickInterval}
          />
          <YAxis
            tick={{ fontSize: 10, fill: colors.muted.DEFAULT, fontFamily: "IBM Plex Mono, monospace" }}
            tickLine={false}
            axisLine={false}
            allowDecimals={false}
          />
          <Tooltip
            cursor={{ fill: colors.cream.DEFAULT }}
            contentStyle={{
              background: colors.paper,
              border: `1px solid ${colors.line.DEFAULT}`,
              borderRadius: 6,
              fontSize: 12,
              fontFamily: "IBM Plex Mono, monospace",
            }}
            formatter={(value) => [
              value ?? 0,
              t("dashboard.usageChart.tooltipLabel"),
            ]}
            labelStyle={{ color: colors.muted.DEFAULT }}
          />
          <Bar
            dataKey="count"
            fill={colors.navy.DEFAULT}
            radius={[3, 3, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
