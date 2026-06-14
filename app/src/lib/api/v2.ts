/**
 * v2 endpoints — NEL service (model selection, taxonomy, user config).
 * Base URL: VITE_NEL_V2_API_BASE_URL.
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
