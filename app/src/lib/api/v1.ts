/**
 * Health probe. classify_v2 exposes /v2/classify/health (no auth required)
 * and the topbar pill polls it via useApiHealth.
 *
 * Historical note: this file used to host the classify_v1 user-config and
 * usage helpers. classify_v1 was retired during the redesign; only the
 * health endpoint remains, but the filename is kept stable to avoid churn
 * elsewhere.
 */

import { request } from "./fetcher";

export interface HealthResponse {
  status: "healthy" | "degraded" | string;
  service?: string;
  version?: string;
  dependencies?: Record<string, string>;
}

/** Public health probe — no auth required by the backend. */
export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/v2/classify/health");
}
