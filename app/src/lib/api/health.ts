/** Health probe — classify_v2 exposes /v2/classify/health (no auth required). */

import { API_BASE_URL, ApiError } from "./fetcher";

export interface HealthResponse {
  status: "healthy" | "degraded" | string;
  service?: string;
  version?: string;
  dependencies?: Record<string, string>;
}

/**
 * Public health probe. Intentionally does NOT use the authenticated `request()`
 * helper: the endpoint requires no auth, and sending an `Authorization` header
 * would make this a non-"simple" cross-origin request, triggering a CORS
 * preflight. The API Gateway's health route has no OPTIONS method, so that
 * preflight 405s. A plain GET with no custom headers stays "simple" — no
 * preflight, no CORS failure.
 */
export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/v2/classify/health`, {
    method: "GET",
  });
  if (!response.ok) {
    throw new ApiError(response.status, await response.text());
  }
  return (await response.json()) as HealthResponse;
}
