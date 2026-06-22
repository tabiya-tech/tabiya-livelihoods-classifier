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

// ── Classify (Classify v2) ─────────────────────────────────────────────────

export type ClassifyEntityType = "occupation" | "skill" | "qualification";

export interface ClassifyEntitySpan {
  /** Character offset of the start of the surface form (inclusive). */
  start: number;
  /** Character offset of the end of the surface form (exclusive). */
  end: number;
}

interface ClassifyEntityBase {
  uuid: string;
  origin_uuid: string;
  uuid_history: string[];
  preferred_label: string;
  origin_uri: string;
  alt_labels: string[];
  description: string;
}

export interface ClassifyOccupationEntity extends ClassifyEntityBase {
  /** ESCO occupation code. Some taxonomies omit it. */
  esco_code?: string | null;
}

export interface ClassifySkillEntity extends ClassifyEntityBase {
  skill_type?: string | null;
  reuse_level?: string | null;
}

export interface ClassifyQualificationEntity extends ClassifyEntityBase {
  eqf_level?: string | null;
  country?: string | null;
}

export type ClassifyMatchEntity =
  | ClassifyOccupationEntity
  | ClassifySkillEntity
  | ClassifyQualificationEntity;

export interface ClassifyOccupationMatch {
  entity_type: "occupation";
  similarity_score: number;
  entity: ClassifyOccupationEntity;
}

export interface ClassifySkillMatch {
  entity_type: "skill";
  similarity_score: number;
  entity: ClassifySkillEntity;
}

export interface ClassifyQualificationMatch {
  entity_type: "qualification";
  similarity_score: number;
  entity: ClassifyQualificationEntity;
}

export type ClassifyMatch =
  | ClassifyOccupationMatch
  | ClassifySkillMatch
  | ClassifyQualificationMatch;

export interface ClassifiedEntity {
  entity_type: ClassifyEntityType;
  surface_form: string;
  span: ClassifyEntitySpan;
  /** ESCO matches ordered by similarity_score descending. */
  matches: ClassifyMatch[];
}

export interface ClassifyOptions {
  /** Restrict extraction to specific entity types. Omit to extract all. */
  extract_entities?: ClassifyEntityType[];
  /** Max matches per entity (1–50). Default 5. */
  top_k?: number;
  /** Minimum cosine similarity to include (0.0–1.0). Default 0.0. */
  min_similarity?: number;
}

export interface ClassifyRequest {
  /** Raw job ad text. Use this OR title + description. */
  text?: string;
  title?: string;
  description?: string;
  options?: ClassifyOptions;
}

export interface ClassifyMetadata {
  classifier_version: string;
  ner_model: string;
  nel_model_id: string;
  taxonomy_model_id: string;
  processing_time_ms: number;
}

export interface ClassifyResponse {
  entities: ClassifiedEntity[];
  metadata: ClassifyMetadata;
}

export function classify(payload: ClassifyRequest): Promise<ClassifyResponse> {
  return request<ClassifyResponse>("/v2/classify", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
