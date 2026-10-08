/**
 * Canonical fixtures for /v2/pipelines. Two pipelines by default:
 *   - "Default Tabiya" — readonly, default, active, canonical 4-stage chain.
 *   - "Recruiter tuning" — user-edited variant with narrower entity_types
 *     and a higher top_k, so stories can show the diff at a glance.
 */

import type { Pipeline, PipelineStage } from "@/lib/api";

const NOW = "2026-07-08T12:00:00.000Z";

export const fixtureDefaultTabiyaStages: PipelineStage[] = [
  { plugin_id: "tabiya.source.text.v1", config: { text: "" } },
  { plugin_id: "tabiya.ner.v1", config: {} },
  {
    plugin_id: "tabiya.nel.v1",
    config: {
      nel_model_id: "all-MiniLM-L6-v2",
      taxonomy_model_id: "esco-v1.2",
      top_k: 5,
      min_similarity: 0.0,
    },
  },
  { plugin_id: "tabiya.sink.results.v1", config: {} },
];

export const fixtureRecruiterTuningStages: PipelineStage[] = [
  { plugin_id: "tabiya.source.text.v1", config: { text: "" } },
  {
    plugin_id: "tabiya.ner.v1",
    config: { entity_types: ["occupation", "skill"] },
  },
  {
    plugin_id: "tabiya.nel.v1",
    config: {
      nel_model_id: "all-MiniLM-L6-v2",
      taxonomy_model_id: "esco-v1.2",
      top_k: 10,
      min_similarity: 0.4,
    },
  },
  { plugin_id: "tabiya.sink.results.v1", config: {} },
];

export const fixtureDefaultTabiyaPipeline: Pipeline = {
  pipeline_id: "pipeline-default",
  user_id: "local-user",
  name: "Default Tabiya",
  stages: fixtureDefaultTabiyaStages,
  is_active: true,
  is_default: true,
  is_readonly: true,
  created_at: NOW,
  updated_at: NOW,
};

export const fixtureRecruiterTuningPipeline: Pipeline = {
  pipeline_id: "pipeline-recruiter-tuning",
  user_id: "local-user",
  name: "Recruiter tuning",
  stages: fixtureRecruiterTuningStages,
  is_active: false,
  is_default: false,
  is_readonly: false,
  created_at: NOW,
  updated_at: NOW,
};

export const fixturePipelines: Pipeline[] = [
  fixtureDefaultTabiyaPipeline,
  fixtureRecruiterTuningPipeline,
];
