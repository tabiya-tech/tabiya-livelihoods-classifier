/**
 * Polls the backend's /v2/classify/health endpoint and exposes its status to the UI.
 *
 * - Polls every `pollIntervalMs` (default 5 min).
 * - First call happens on mount; the component sees `status: "unknown"` until
 *   the first response arrives.
 * - Network/parse failures map to "down".
 * - Non-200 with a body map to "degraded".
 * - `refresh()` triggers an immediate check and resets the poll timer.
 *
 * Storybook / exploratory tooling can short-circuit the live poll by wrapping
 * a subtree in `<ApiHealthOverrideProvider value={…} />`. Production never
 * mounts the provider.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { ApiError, getHealth, type HealthResponse } from "@/lib/api";

export type ApiHealthStatus = "healthy" | "degraded" | "down" | "unknown";

export interface ApiHealthSnapshot {
  status: ApiHealthStatus;
  version?: string;
  /** True while an explicit refresh() call is in-flight. */
  isRefreshing: boolean;
  /** Last time we received a non-error response. */
  lastCheckedAt: Date | null;
}

export interface UseApiHealthOptions {
  /** Poll interval in milliseconds. Defaults to 300 000 (5 min). */
  pollIntervalMs?: number;
  /** Test seam — override the health fetcher. Defaults to the real getHealth. */
  fetchHealth?: () => Promise<HealthResponse>;
}

export interface UseApiHealthResult extends ApiHealthSnapshot {
  /** Trigger an immediate check and reset the poll timer. */
  refresh: () => void;
}

const DEFAULT_POLL_INTERVAL_MS = 300_000;

function deriveStatusFromResponse(response: HealthResponse): ApiHealthStatus {
  if (response.status === "healthy") return "healthy";
  return "degraded";
}

// ── Override plumbing (Storybook / exploratory tooling only) ──────────────

const ApiHealthOverrideContext = createContext<ApiHealthSnapshot | null>(null);

export interface ApiHealthOverrideProviderProps {
  value: ApiHealthSnapshot;
  children: ReactNode;
}

/**
 * Wrap a subtree to short-circuit `useApiHealth` and return a deterministic
 * snapshot. Intended for Storybook and one-off exploratory harnesses —
 * production never mounts this.
 */
export function ApiHealthOverrideProvider({
  value,
  children,
}: ApiHealthOverrideProviderProps) {
  return (
    <ApiHealthOverrideContext.Provider value={value}>
      {children}
    </ApiHealthOverrideContext.Provider>
  );
}

// ── Hook ──────────────────────────────────────────────────────────────────

function useLiveApiHealth({
  pollIntervalMs,
  fetchHealth,
}: Required<UseApiHealthOptions>): UseApiHealthResult {
  const [snapshot, setSnapshot] = useState<ApiHealthSnapshot>({
    status: "unknown",
    isRefreshing: false,
    lastCheckedAt: null,
  });

  // Incrementing this triggers an immediate check + poll-timer reset.
  const [refreshTick, setRefreshTick] = useState(0);

  // Stable ref to the fetcher so the polling effect doesn't restart on every
  // parent render that happens to pass a new function identity.
  const fetchHealthRef = useRef(fetchHealth);
  fetchHealthRef.current = fetchHealth;

  const isManualRefresh = refreshTick > 0;

  useEffect(() => {
    let cancelled = false;

    async function checkHealthOnce() {
      if (isManualRefresh) {
        setSnapshot((prev) => ({ ...prev, isRefreshing: true }));
      }
      try {
        const response = await fetchHealthRef.current();
        if (cancelled) return;
        setSnapshot({
          status: deriveStatusFromResponse(response),
          version: response.version,
          isRefreshing: false,
          lastCheckedAt: new Date(),
        });
      } catch (caught) {
        if (cancelled) return;
        const isApiError = caught instanceof ApiError;
        setSnapshot((previous) => ({
          status: isApiError ? "degraded" : "down",
          version: previous.version,
          isRefreshing: false,
          lastCheckedAt: previous.lastCheckedAt,
        }));
      }
    }

    checkHealthOnce();
    const intervalHandle = setInterval(checkHealthOnce, pollIntervalMs);

    return () => {
      cancelled = true;
      clearInterval(intervalHandle);
    };
  }, [pollIntervalMs, refreshTick]); // refreshTick re-mounts the effect, resetting the timer

  const refresh = useCallback(() => {
    setRefreshTick((tick) => tick + 1);
  }, []);

  return { ...snapshot, refresh };
}

export function useApiHealth({
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  fetchHealth = getHealth,
}: UseApiHealthOptions = {}): UseApiHealthResult {
  // Rules-of-hooks: both branches always call hooks. Override shadows live.
  const overrideValue = useContext(ApiHealthOverrideContext);
  const liveValue = useLiveApiHealth({ pollIntervalMs, fetchHealth });
  // When overriding (Storybook), refresh is a no-op.
  const noOpRefresh = useCallback(() => {}, []);
  if (overrideValue) return { ...overrideValue, isRefreshing: false, refresh: noOpRefresh };
  return liveValue;
}
