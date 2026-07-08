/**
 * MSW handlers for /v2/pipelines against an in-memory store.
 *
 * Behaviour matches the real backend for the paths the frontend exercises:
 *   - Create / update / clone assign a new pipeline_id.
 *   - Activate flips exactly one pipeline to is_active=true; the others
 *     for the same user go to false.
 *   - Update + delete on `is_readonly: true` return 409.
 *   - `POST /v2/pipelines/validate` runs a minimal in-fixture validator
 *     (source-first, sink-last, ≥2 stages, coming_soon rejected) so
 *     Storybook can drive the badge without hitting a live backend.
 *
 * Tests reset the store via {@link resetPipelinesHandlersStore}.
 */

import { http, HttpResponse } from "msw";
import { API_BASE_URL } from "@/lib/api";
import type {
  CreatePipelineRequest,
  Pipeline,
  PipelineStage,
  PipelineValidationIssue,
  UpdatePipelineRequest,
  ValidatePipelineRequest,
} from "@/lib/api";
import {
  fixturePipelines,
  fixtureDefaultTabiyaPipeline,
} from "../fixtures/pipelines";
import { fixturePluginSummaries } from "../fixtures/plugins";

let currentPipelines: Pipeline[] = fixturePipelines.map((pipeline) => ({
  ...pipeline,
  stages: pipeline.stages.map((stage) => ({ ...stage })),
}));
let nextPipelineSequence = 1000;

/** Reset the store back to the shipped fixtures. */
export function resetPipelinesHandlersStore() {
  currentPipelines = fixturePipelines.map((pipeline) => ({
    ...pipeline,
    stages: pipeline.stages.map((stage) => ({ ...stage })),
  }));
  nextPipelineSequence = 1000;
}

/** Replace the store with the supplied list (used for stories). */
export function seedPipelinesHandlersStore(seed: Pipeline[]) {
  currentPipelines = seed.map((pipeline) => ({
    ...pipeline,
    stages: pipeline.stages.map((stage) => ({ ...stage })),
  }));
}

const listUrl = `${API_BASE_URL}/v2/pipelines`;
const itemUrl = `${API_BASE_URL}/v2/pipelines/:pipelineId`;
const validateUrl = `${API_BASE_URL}/v2/pipelines/validate`;
const activateUrl = `${API_BASE_URL}/v2/pipelines/:pipelineId/activate`;
const cloneUrl = `${API_BASE_URL}/v2/pipelines/:pipelineId/clone`;

function nowIso(): string {
  return new Date().toISOString();
}

function nextPipelineId(): string {
  const value = `pipeline-${String(nextPipelineSequence).padStart(4, "0")}`;
  nextPipelineSequence += 1;
  return value;
}

function summaryByPluginId(pluginId: string) {
  return fixturePluginSummaries.find(
    (summary) => summary.plugin_id === pluginId,
  );
}

/**
 * Frontend-side validator used by the MSW handler for both the standalone
 * /validate route and mutations. Deliberately a subset of the backend's
 * rules — Storybook stories only need to drive the happy-path badge +
 * a couple of failure modes. The real backend catches anything richer.
 */
function validateStages(stages: PipelineStage[]): PipelineValidationIssue[] {
  const issues: PipelineValidationIssue[] = [];
  if (stages.length < 2) {
    issues.push({
      code: "too_few_stages",
      message: "A pipeline needs at least a Source and a Sink stage.",
      stage_index: null,
      plugin_id: null,
    });
  }
  stages.forEach((stage, stageIndex) => {
    const summary = summaryByPluginId(stage.plugin_id);
    if (!summary) {
      issues.push({
        code: "unknown_plugin",
        message: `Plugin '${stage.plugin_id}' is not in the catalog.`,
        stage_index: stageIndex,
        plugin_id: stage.plugin_id,
      });
      return;
    }
    if (summary.coming_soon) {
      issues.push({
        code: "coming_soon_plugin",
        message: `Plugin '${stage.plugin_id}' is marked Coming Soon.`,
        stage_index: stageIndex,
        plugin_id: stage.plugin_id,
      });
    }
  });
  if (stages.length >= 2) {
    const first = summaryByPluginId(stages[0].plugin_id);
    if (first && first.category !== "source") {
      issues.push({
        code: "not_a_source",
        message: `First stage must be a Source; got ${first.category}.`,
        stage_index: 0,
        plugin_id: first.plugin_id,
      });
    }
    const last = summaryByPluginId(stages[stages.length - 1].plugin_id);
    if (last && last.category !== "sink") {
      issues.push({
        code: "not_a_sink",
        message: `Last stage must be a Sink; got ${last.category}.`,
        stage_index: stages.length - 1,
        plugin_id: last.plugin_id,
      });
    }
  }
  return issues;
}

function pipelineByIdForUser(pipelineId: string): Pipeline | undefined {
  return currentPipelines.find((pipeline) => pipeline.pipeline_id === pipelineId);
}

export const pipelinesHandlers = [
  http.get(listUrl, () =>
    HttpResponse.json({ pipelines: currentPipelines }),
  ),

  http.post(listUrl, async ({ request }) => {
    const body = (await request.json()) as CreatePipelineRequest;
    const issues = validateStages(body.stages);
    if (issues.length > 0) {
      return HttpResponse.json(
        { detail: { message: "Pipeline validation failed.", issues } },
        { status: 422 },
      );
    }
    const inserted: Pipeline = {
      pipeline_id: nextPipelineId(),
      user_id: "local-user",
      name: body.name,
      stages: body.stages.map((stage) => ({ ...stage })),
      is_active: false,
      is_default: false,
      is_readonly: false,
      created_at: nowIso(),
      updated_at: nowIso(),
    };
    currentPipelines = [...currentPipelines, inserted];
    return HttpResponse.json(inserted, { status: 201 });
  }),

  http.post(validateUrl, async ({ request }) => {
    const body = (await request.json()) as ValidatePipelineRequest;
    const issues = validateStages(body.stages);
    return HttpResponse.json({ valid: issues.length === 0, issues });
  }),

  http.get(itemUrl, ({ params }) => {
    const pipeline = pipelineByIdForUser(String(params.pipelineId));
    if (!pipeline) {
      return HttpResponse.json(
        { detail: "Pipeline not found" },
        { status: 404 },
      );
    }
    return HttpResponse.json(pipeline);
  }),

  http.put(itemUrl, async ({ params, request }) => {
    const pipelineId = String(params.pipelineId);
    const existing = pipelineByIdForUser(pipelineId);
    if (!existing) {
      return HttpResponse.json(
        { detail: "Pipeline not found" },
        { status: 404 },
      );
    }
    if (existing.is_readonly) {
      return HttpResponse.json(
        { detail: `Pipeline '${pipelineId}' is read-only.` },
        { status: 409 },
      );
    }
    const body = (await request.json()) as UpdatePipelineRequest;
    const issues = validateStages(body.stages);
    if (issues.length > 0) {
      return HttpResponse.json(
        { detail: { message: "Pipeline validation failed.", issues } },
        { status: 422 },
      );
    }
    const updated: Pipeline = {
      ...existing,
      name: body.name,
      stages: body.stages.map((stage) => ({ ...stage })),
      updated_at: nowIso(),
    };
    currentPipelines = currentPipelines.map((pipeline) =>
      pipeline.pipeline_id === pipelineId ? updated : pipeline,
    );
    return HttpResponse.json(updated);
  }),

  http.delete(itemUrl, ({ params }) => {
    const pipelineId = String(params.pipelineId);
    const existing = pipelineByIdForUser(pipelineId);
    if (!existing) {
      return HttpResponse.json(
        { detail: "Pipeline not found" },
        { status: 404 },
      );
    }
    if (existing.is_readonly) {
      return HttpResponse.json(
        { detail: `Pipeline '${pipelineId}' is read-only.` },
        { status: 409 },
      );
    }
    currentPipelines = currentPipelines.filter(
      (pipeline) => pipeline.pipeline_id !== pipelineId,
    );
    return new HttpResponse(null, { status: 204 });
  }),

  http.post(activateUrl, ({ params }) => {
    const pipelineId = String(params.pipelineId);
    const target = pipelineByIdForUser(pipelineId);
    if (!target) {
      return HttpResponse.json(
        { detail: "Pipeline not found" },
        { status: 404 },
      );
    }
    const issues = validateStages(target.stages);
    if (issues.length > 0) {
      return HttpResponse.json(
        { detail: { message: "Pipeline validation failed.", issues } },
        { status: 422 },
      );
    }
    currentPipelines = currentPipelines.map((pipeline) => ({
      ...pipeline,
      is_active: pipeline.pipeline_id === pipelineId,
      updated_at:
        pipeline.pipeline_id === pipelineId ? nowIso() : pipeline.updated_at,
    }));
    return HttpResponse.json(
      currentPipelines.find((pipeline) => pipeline.pipeline_id === pipelineId),
    );
  }),

  http.post(cloneUrl, ({ params }) => {
    const source = pipelineByIdForUser(String(params.pipelineId));
    if (!source) {
      return HttpResponse.json(
        { detail: "Pipeline not found" },
        { status: 404 },
      );
    }
    const cloneName = source.name.endsWith(" (copy)")
      ? source.name
      : `${source.name} (copy)`;
    const cloned: Pipeline = {
      pipeline_id: nextPipelineId(),
      user_id: source.user_id,
      name: cloneName,
      stages: source.stages.map((stage) => ({ ...stage })),
      is_active: false,
      is_default: false,
      is_readonly: false,
      created_at: nowIso(),
      updated_at: nowIso(),
    };
    currentPipelines = [...currentPipelines, cloned];
    return HttpResponse.json(cloned, { status: 201 });
  }),
];

// Re-exports used by consumers who don't want to reach into ../fixtures.
export { fixtureDefaultTabiyaPipeline };
