import { useEffect, useState } from "react";
import { getUsage, type DailyCount } from "@/lib/api";

export type UsageStatus = "loading" | "ready" | "error";

export interface UsageSnapshot {
  status: UsageStatus;
  data: DailyCount[];
  error: Error | null;
}

export interface UseUsageOptions {
  days?: number;
  fetchUsage?: (days: number) => Promise<{ days: number; data: DailyCount[] }>;
}

export function useUsage({
  days = 30,
  fetchUsage = getUsage,
}: UseUsageOptions = {}): UsageSnapshot {
  const [snapshot, setSnapshot] = useState<UsageSnapshot>({
    status: "loading",
    data: [],
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    setSnapshot({ status: "loading", data: [], error: null });

    fetchUsage(days)
      .then((response) => {
        if (!cancelled) {
          setSnapshot({ status: "ready", data: response.data, error: null });
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          const error =
            caught instanceof Error ? caught : new Error(String(caught));
          setSnapshot({ status: "error", data: [], error });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [days, fetchUsage]);

  return snapshot;
}
