/**
 * Authenticated fetch wrapper for the Tabiya Classifier backends.
 *
 * - Reads the Firebase ID token from `auth.currentUser` and sends it as
 *   `Authorization: Bearer …`.
 * - If the backend returns 401, refreshes the ID token once (forces a fresh
 *   token from Firebase) and retries the request a single time. A second 401
 *   bubbles up as an `ApiError`.
 * - Surfaces non-2xx responses as `ApiError` so callers can branch on status.
 */

import { auth } from "../firebase";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:5001";
export const NEL_V2_API_BASE_URL =
  import.meta.env.VITE_NEL_V2_API_BASE_URL ?? API_BASE_URL;

export class ApiError extends Error {
  readonly status: number;
  readonly body: string;
  constructor(status: number, body: string) {
    super(`API ${status}: ${body}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

/** Allows tests to substitute the token source without touching Firebase. */
export interface RequestContext {
  /** Fetches a Firebase ID token. Pass `forceRefresh=true` to mint a fresh one. */
  getIdToken: (forceRefresh?: boolean) => Promise<string>;
  /** Override for tests. Defaults to the global `fetch`. */
  fetchImpl?: typeof fetch;
}

function defaultIdTokenSource(forceRefresh = false): Promise<string> {
  const user = auth.currentUser;
  if (!user) {
    return Promise.reject(new Error("Not authenticated"));
  }
  return user.getIdToken(forceRefresh);
}

export interface RequestOptions extends RequestInit {
  /** Base URL override. Defaults to `API_BASE_URL`. */
  baseUrl?: string;
  /** Test seam — see `RequestContext`. Production callers leave this unset. */
  context?: RequestContext;
}

/**
 * Send an authenticated request and parse the JSON response.
 * Throws `ApiError` for non-2xx responses (after the single 401-retry).
 */
export async function request<TResponse>(
  path: string,
  options: RequestOptions = {},
): Promise<TResponse> {
  const baseUrl = options.baseUrl ?? API_BASE_URL;
  const fetchImpl = options.context?.fetchImpl ?? fetch;
  const getIdToken = options.context?.getIdToken ?? defaultIdTokenSource;

  const buildInit = (idToken: string): RequestInit => ({
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${idToken}`,
      ...(options.headers ?? {}),
    },
  });

  // First attempt with the cached token.
  let idToken = await getIdToken(false);
  let response = await fetchImpl(`${baseUrl}${path}`, buildInit(idToken));

  // 401 → refresh the token once and retry. A second 401 surfaces as ApiError.
  if (response.status === 401) {
    idToken = await getIdToken(true);
    response = await fetchImpl(`${baseUrl}${path}`, buildInit(idToken));
  }

  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(response.status, body);
  }

  // 204 No Content → return undefined (cast to TResponse for the void case).
  if (response.status === 204) {
    return undefined as TResponse;
  }
  return await response.json() as Promise<TResponse>;
}
