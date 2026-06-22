/**
 * v1 endpoints — user config, API keys, usage, health.
 * Base URL: VITE_API_BASE_URL (the classify backend).
 */

import { request } from "./fetcher";

// ── Health ─────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: "healthy" | "degraded" | string;
  service?: string;
  version?: string;
  dependencies?: Record<string, string>;
}

/** Public health probe — no auth required by the backend. */
export function getHealth(): Promise<HealthResponse> {
  // Send through the same wrapper for consistency; the backend ignores the
  // bearer token on /v1/health.
  return request<HealthResponse>("/v1/health");
}

// ── User config ────────────────────────────────────────────────────────────

export interface UserConfig {
  ner_type: string;
  nel_type: string;
  ner_model_name: string;
  nel_model_name: string;
  taxonomy_model_id: string;
}

export function getUserConfig(): Promise<UserConfig> {
  return request<UserConfig>("/v1/user/config");
}

export function saveUserConfig(config: Partial<UserConfig>): Promise<void> {
  return request<void>("/v1/user/config", {
    method: "PUT",
    body: JSON.stringify(config),
  });
}

// ── Usage ──────────────────────────────────────────────────────────────────

export interface UsagePoint {
  /** ISO date string (YYYY-MM-DD). */
  date: string;
  count: number;
}

export function getUsage(): Promise<UsagePoint[]> {
  return request<UsagePoint[]>("/v1/user/usage");
}
