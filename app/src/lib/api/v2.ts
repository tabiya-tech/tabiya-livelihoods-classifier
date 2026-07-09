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

/**
 * Entity types the NER model can emit. `occupation`, `skill`, and
 * `qualification` are linkable against ESCO; `experience` and `domain` are
 * recognised + displayed but never have matches (backend skips NEL for them).
 */
export type ClassifyEntityType =
  | "occupation"
  | "skill"
  | "qualification"
  | "experience"
  | "domain";

/** The subset of {@link ClassifyEntityType} the backend tries to link to ESCO. */
export const LINKABLE_ENTITY_TYPES = [
  "occupation",
  "skill",
  "qualification",
] as const satisfies readonly ClassifyEntityType[];

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
  /**
   * Optional pipeline_id — forwards which pipeline the executor should use.
   * When omitted, the backend falls back to the caller's active pipeline.
   */
  pipeline_id?: string;
  options?: ClassifyOptions;
}

export interface PipelineStageSummary {
  plugin_id: string;
  category: string;
}

export interface ClassifyPipelineSummary {
  pipeline_id: string;
  name: string;
  stages: PipelineStageSummary[];
}

export interface ClassifyMetadata {
  classifier_version: string;
  ner_model: string;
  nel_model_id: string;
  taxonomy_model_id: string;
  processing_time_ms: number;
  /** Present once the backend runs classify through the pipeline executor. */
  pipeline?: ClassifyPipelineSummary | null;
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

// ── Plugins (Classify v2) ──────────────────────────────────────────────────

/** Runtime status of a plugin as seen by the orchestrator. */
export type PluginStatus = "enabled" | "degraded" | "unavailable";

export type PluginCategory = "source" | "core" | "transform" | "sink";

/** Extra fields the backend may attach under `x-tabiya-*`. */
export interface PluginCapabilities {
  "x-tabiya-contract-version"?: string | null;
  "x-tabiya-streams"?: boolean | null;
  "x-tabiya-idempotent"?: boolean | null;
  "x-tabiya-cancellable"?: boolean | null;
  "x-tabiya-batch-max"?: number | null;
}

/** Palette-sized manifest projection returned by `GET /v2/plugins`. */
export interface PluginSummary {
  plugin_id: string;
  name: string;
  version: string;
  category?: PluginCategory | null;
  summary: string;
  detail?: string | null;
  icon: string;
  status: PluginStatus;
  coming_soon: boolean;
  last_error?: string | null;
}

export interface ListPluginsResponse {
  plugins: PluginSummary[];
}

export interface PluginSlot {
  type: "None" | "RawText" | "RawTextStream" | "Entities" | "LinkedEntities";
  cardinality?: "single" | "none";
}

/** Full plugin manifest, as returned by `GET /v2/plugins/{plugin_id}`. */
export interface PluginManifest extends PluginCapabilities {
  plugin_id: string;
  name: string;
  version: string;
  category: PluginCategory;
  summary: string;
  detail?: string | null;
  icon: string;
  input_slot: PluginSlot;
  output_slot: PluginSlot;
  config_schema: Record<string, unknown>;
  timeout_ms: number;
}

/** Response body of `GET /v2/plugins/{plugin_id}`. */
export interface PluginDetail {
  plugin_id: string;
  status: PluginStatus;
  coming_soon: boolean;
  last_error?: string | null;
  manifest: PluginManifest | null;
}

export interface PluginOptionItem {
  value: string;
  label: string;
}

export interface PluginOptionsResponse {
  field: string;
  options: PluginOptionItem[];
}

export function listPlugins(): Promise<ListPluginsResponse> {
  return request<ListPluginsResponse>("/v2/plugins");
}

export function getPlugin(pluginId: string): Promise<PluginDetail> {
  return request<PluginDetail>(`/v2/plugins/${encodeURIComponent(pluginId)}`);
}

export function getPluginOptions(
  pluginId: string,
  field: string,
): Promise<PluginOptionsResponse> {
  return request<PluginOptionsResponse>(
    `/v2/plugins/${encodeURIComponent(pluginId)}/options/${encodeURIComponent(field)}`,
  );
}

// ── Pipelines (Classify v2) ────────────────────────────────────────────────

export interface PipelineStage {
  plugin_id: string;
  config: Record<string, unknown>;
}

export interface Pipeline {
  pipeline_id: string;
  user_id: string;
  name: string;
  stages: PipelineStage[];
  is_active: boolean;
  is_default: boolean;
  is_readonly: boolean;
  /** ISO-8601 UTC timestamp. */
  created_at: string;
  /** ISO-8601 UTC timestamp. */
  updated_at: string;
}

export interface ListPipelinesResponse {
  pipelines: Pipeline[];
}

export interface CreatePipelineRequest {
  name: string;
  stages: PipelineStage[];
}

export interface UpdatePipelineRequest {
  name: string;
  stages: PipelineStage[];
}

/** One issue surfaced by the backend validator (design §7). */
export interface PipelineValidationIssue {
  code: string;
  message: string;
  stage_index?: number | null;
  plugin_id?: string | null;
  detail?: Record<string, unknown> | null;
}

export interface ValidatePipelineRequest {
  stages: PipelineStage[];
}

export interface ValidatePipelineResponse {
  valid: boolean;
  issues: PipelineValidationIssue[];
}

export function listPipelines(): Promise<ListPipelinesResponse> {
  return request<ListPipelinesResponse>("/v2/pipelines");
}

export function getPipeline(pipelineId: string): Promise<Pipeline> {
  return request<Pipeline>(
    `/v2/pipelines/${encodeURIComponent(pipelineId)}`,
  );
}

export function createPipeline(
  payload: CreatePipelineRequest,
): Promise<Pipeline> {
  return request<Pipeline>("/v2/pipelines", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updatePipeline(
  pipelineId: string,
  payload: UpdatePipelineRequest,
): Promise<Pipeline> {
  return request<Pipeline>(
    `/v2/pipelines/${encodeURIComponent(pipelineId)}`,
    { method: "PUT", body: JSON.stringify(payload) },
  );
}

export function deletePipeline(pipelineId: string): Promise<void> {
  return request<void>(
    `/v2/pipelines/${encodeURIComponent(pipelineId)}`,
    { method: "DELETE" },
  );
}

export function activatePipeline(pipelineId: string): Promise<Pipeline> {
  return request<Pipeline>(
    `/v2/pipelines/${encodeURIComponent(pipelineId)}/activate`,
    { method: "POST" },
  );
}

export function clonePipeline(pipelineId: string): Promise<Pipeline> {
  return request<Pipeline>(
    `/v2/pipelines/${encodeURIComponent(pipelineId)}/clone`,
    { method: "POST" },
  );
}

export function validatePipeline(
  payload: ValidatePipelineRequest,
): Promise<ValidatePipelineResponse> {
  return request<ValidatePipelineResponse>("/v2/pipelines/validate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
