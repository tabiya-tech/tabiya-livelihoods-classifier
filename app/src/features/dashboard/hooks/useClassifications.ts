import { useCallback, useEffect, useState } from "react";
import {
  listClassifications,
  type ClassificationSummary,
  type ClassificationsPage,
} from "@/lib/api";

export type ClassificationsStatus = "loading" | "ready" | "error";

export interface ClassificationsSnapshot {
  status: ClassificationsStatus;
  items: ClassificationSummary[];
  nextCursor: string | null;
  error: Error | null;
  loadMore: () => void;
}

export interface UseClassificationsOptions {
  limit?: number;
  fetchClassifications?: (params: {
    limit?: number;
    cursor?: string;
  }) => Promise<ClassificationsPage>;
}

export function useClassifications({
  limit = 20,
  fetchClassifications = listClassifications,
}: UseClassificationsOptions = {}): ClassificationsSnapshot {
  const [items, setItems] = useState<ClassificationSummary[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [status, setStatus] = useState<ClassificationsStatus>("loading");
  const [error, setError] = useState<Error | null>(null);

  const load = useCallback(
    async (cursor?: string) => {
      setStatus(cursor ? "ready" : "loading");
      try {
        const response = await fetchClassifications({ limit, cursor });
        setItems((previous) =>
          cursor ? [...previous, ...response.items] : response.items,
        );
        setNextCursor(response.next_cursor);
        setStatus("ready");
        setError(null);
      } catch (caught: unknown) {
        const fetchError =
          caught instanceof Error ? caught : new Error(String(caught));
        setError(fetchError);
        setStatus("error");
      }
    },
    [limit, fetchClassifications],
  );

  useEffect(() => {
    void load();
  }, [load]);

  const loadMore = useCallback(() => {
    if (nextCursor) void load(nextCursor);
  }, [load, nextCursor]);

  return { status, items, nextCursor, error, loadMore };
}
