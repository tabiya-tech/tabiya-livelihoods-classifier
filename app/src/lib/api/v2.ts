/**
 * v2 endpoints. Two backends:
 *  - NEL v2 (model selection, taxonomy, user config) — VITE_NEL_V2_API_BASE_URL
 *  - Classify v2 (api-keys, classify) — VITE_API_BASE_URL
 */

import { NEL_V2_API_BASE_URL, request } from "./fetcher";

// ── User config (NEL v2) ───────────────────────────────────────────────────

export interface V2UserConfig {
  taxonomy_model_id: string;
  nel_model_id: string;
}

export function getV2UserConfig(): Promise<V2UserConfig> {
  return request<V2UserConfig>("/v2/nel/user/config", {
    baseUrl: NEL_V2_API_BASE_URL,
  });
}

export function saveV2UserConfig(config: V2UserConfig): Promise<V2UserConfig> {
  return request<V2UserConfig>("/v2/nel/user/config", {
    baseUrl: NEL_V2_API_BASE_URL,
    method: "PUT",
    body: JSON.stringify(config),
  });
}

// ── NEL models ─────────────────────────────────────────────────────────────

export interface NelModel {
  model_id: string;
  dimensions: number;
  description: string;
}

export function listNelModels(): Promise<NelModel[]> {
  return request<NelModel[]>("/v2/nel/models", {
    baseUrl: NEL_V2_API_BASE_URL,
  });
}

// ── Taxonomy models ────────────────────────────────────────────────────────

export interface TaxonomyModel {
  id: string;
  name: string;
  version: string;
  description: string;
  released: boolean;
}

export function listTaxonomyModels(): Promise<TaxonomyModel[]> {
  return request<TaxonomyModel[]>("/v2/nel/taxonomy-models", {
    baseUrl: NEL_V2_API_BASE_URL,
  });
}

// ── API keys (Classify v2) ─────────────────────────────────────────────────

export interface ApiKeyMetadata {
  key_id: string;
  user_id: string;
  label: string;
  /** Unix epoch seconds. */
  created_at: number;
  /** Unix epoch seconds; null until the key has been used. */
  last_used_at: number | null;
  revoked: boolean;
}

export interface ListApiKeysResponse {
  keys: ApiKeyMetadata[];
}

export interface CreateApiKeyResponse {
  /** The plaintext key. Returned exactly once — show it to the user immediately. */
  key: string;
  meta: ApiKeyMetadata;
}

export function listApiKeys(): Promise<ListApiKeysResponse> {
  return request<ListApiKeysResponse>("/v2/user/api-keys");
}

export function createApiKey(label: string): Promise<CreateApiKeyResponse> {
  return request<CreateApiKeyResponse>("/v2/user/api-keys", {
    method: "POST",
    body: JSON.stringify({ label }),
  });
}

export function deleteApiKey(keyId: string): Promise<void> {
  return request<void>(`/v2/user/api-keys/${keyId}`, { method: "DELETE" });
}
