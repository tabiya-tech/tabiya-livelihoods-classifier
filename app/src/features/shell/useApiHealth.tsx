/**
 * Polls the backend's /v2/classify/health endpoint and exposes its status to the UI.
 *
 * - Polls every `pollIntervalMs` (default 30s).
 * - First call happens on mount; the component sees `status: "unknown"` until
 *   the first response arrives.
 * - Network/parse failures map to "down".
 * - Non-200 with a body map to "degraded".
 *
 * Storybook / exploratory tooling can short-circuit the live poll by wrapping
 * a subtree in `<ApiHealthOverrideProvider value={…} />`. Production never
 * mounts the provider.
 */

import {
  createContext,
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
  /** Last time we received a non-error response. */
  lastCheckedAt: Date | null;
}

export interface UseApiHealthOptions {
  /** Poll interval in milliseconds. Defaults to 30 000 (30s). */
  pollIntervalMs?: number;
  /** Test seam — override the health fetcher. Defaults to the real getHealth. */
  fetchHealth?: () => Promise<HealthResponse>;
}

const DEFAULT_POLL_INTERVAL_MS = 30_000;

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
}: Required<UseApiHealthOptions>): ApiHealthSnapshot {
  const [snapshot, setSnapshot] = useState<ApiHealthSnapshot>({
    status: "unknown",
    lastCheckedAt: null,
  });

  // Stable ref to the fetcher so the polling effect doesn't restart on every
  // parent render that happens to pass a new function identity.
  const fetchHealthRef = useRef(fetchHealth);
  fetchHealthRef.current = fetchHealth;

  useEffect(() => {
    let cancelled = false;

    async function checkHealthOnce() {
      try {
        const response = await fetchHealthRef.current();
        if (cancelled) return;
        setSnapshot({
          status: deriveStatusFromResponse(response),
          version: response.version,
          lastCheckedAt: new Date(),
        });
      } catch (caught) {
        if (cancelled) return;
        const isApiError = caught instanceof ApiError;
        setSnapshot((previous) => ({
          status: isApiError ? "degraded" : "down",
          version: previous.version,
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
  }, [pollIntervalMs]);

  return snapshot;
}

export function useApiHealth({
  pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
  fetchHealth = getHealth,
}: UseApiHealthOptions = {}): ApiHealthSnapshot {
  // Rules-of-hooks: both branches always call hooks. Override shadows live.
  const overrideValue = useContext(ApiHealthOverrideContext);
  const liveValue = useLiveApiHealth({ pollIntervalMs, fetchHealth });
  return overrideValue ?? liveValue;
}
