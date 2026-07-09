/** Health probe — classify_v2 exposes /v2/classify/health (no auth required). */

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
