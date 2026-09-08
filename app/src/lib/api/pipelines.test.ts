/**
 * Client tests for /v2/pipelines* against the shared MSW handlers.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { auth } from "@/lib/firebase";
import {
  activatePipeline,
  clonePipeline,
  createPipeline,
  deletePipeline,
  getPipeline,
  listPipelines,
  updatePipeline,
  validatePipeline,
} from "@/lib/api";
import type { PipelineStage } from "@/lib/api";
import { resetPipelinesHandlersStore } from "@/mocks/handlers";
import { fixtureDefaultTabiyaStages } from "@/mocks/fixtures/pipelines";

let originalCurrentUser: unknown;

beforeEach(() => {
  originalCurrentUser = (auth as { currentUser: unknown }).currentUser;
  (auth as { currentUser: unknown }).currentUser = {
    getIdToken: async () => "test-id-token",
  };
});

afterEach(() => {
  (auth as { currentUser: unknown }).currentUser = originalCurrentUser;
  resetPipelinesHandlersStore();
});

const givenValidStages: PipelineStage[] = fixtureDefaultTabiyaStages;

const givenInvalidStages: PipelineStage[] = [
  { plugin_id: "tabiya.source.text.v1", config: { text: "" } },
];

describe("listPipelines", () => {
  it("returns the two seeded pipelines by default", async () => {
    // GIVEN the MSW store seeded with Default Tabiya + Recruiter tuning
    // WHEN we list
    const response = await listPipelines();

    // THEN both come back in insertion order
    const expectedIds = ["pipeline-default", "pipeline-recruiter-tuning"];
    expect(response.pipelines.map((pipeline) => pipeline.pipeline_id)).toEqual(
      expectedIds,
    );
  });
});

describe("createPipeline", () => {
  it("returns the newly-inserted pipeline with a generated id", async () => {
    // GIVEN a valid create request
    const givenName = "My custom pipeline";

    // WHEN we create it
    const inserted = await createPipeline({
      name: givenName,
      stages: givenValidStages,
    });

    // THEN it comes back with an id and the request name preserved
    expect(inserted.name).toBe(givenName);
    expect(inserted.pipeline_id).toMatch(/^pipeline-/);
    expect(inserted.is_active).toBe(false);
    expect(inserted.is_readonly).toBe(false);
  });

  it("rejects with 422 when the pipeline fails validation", async () => {
    // GIVEN a single-stage pipeline
    // WHEN we create it
    // THEN the client rejects with ApiError status=422
    await expect(
      createPipeline({ name: "Bad", stages: givenInvalidStages }),
    ).rejects.toMatchObject({ status: 422 });
  });
});

describe("validatePipeline", () => {
  it("returns valid=true for a canonical pipeline", async () => {
    // GIVEN the canonical stages
    // WHEN we validate them
    const response = await validatePipeline({ stages: givenValidStages });

    // THEN valid=true, no issues
    expect(response.valid).toBe(true);
    expect(response.issues).toEqual([]);
  });

  it("returns valid=false with issues for an invalid pipeline", async () => {
    // GIVEN a too-short stage list
    // WHEN we validate
    const response = await validatePipeline({ stages: givenInvalidStages });

    // THEN valid=false and at least one issue mentions the min-stage rule
    expect(response.valid).toBe(false);
    const expectedCode = "too_few_stages";
    expect(response.issues.map((issue) => issue.code)).toContain(expectedCode);
  });
});

describe("getPipeline", () => {
  it("returns the pipeline document for a known id", async () => {
    // GIVEN a persisted pipeline
    const givenPipelineId = "pipeline-default";

    // WHEN we fetch it
    const pipeline = await getPipeline(givenPipelineId);

    // THEN the document matches the id we requested
    expect(pipeline.pipeline_id).toBe(givenPipelineId);
    expect(pipeline.is_default).toBe(true);
  });

  it("rejects with ApiError on 404 for missing id", async () => {
    await expect(getPipeline("ghost")).rejects.toMatchObject({ status: 404 });
  });
});

describe("updatePipeline", () => {
  it("returns the updated pipeline with the new name", async () => {
    // GIVEN a non-readonly pipeline exists
    const givenPipelineId = "pipeline-recruiter-tuning";
    const givenName = "Renamed";

    // WHEN we update it
    const updated = await updatePipeline(givenPipelineId, {
      name: givenName,
      stages: givenValidStages,
    });

    // THEN the pipeline reflects the new name
    expect(updated.name).toBe(givenName);
    expect(updated.pipeline_id).toBe(givenPipelineId);
  });

  it("rejects with 409 on readonly pipeline", async () => {
    // GIVEN the readonly Default Tabiya
    const givenPipelineId = "pipeline-default";

    // WHEN we try to update it
    // THEN 409
    await expect(
      updatePipeline(givenPipelineId, {
        name: "hijacked",
        stages: givenValidStages,
      }),
    ).rejects.toMatchObject({ status: 409 });
  });
});

describe("deletePipeline", () => {
  it("returns without error on a non-readonly pipeline", async () => {
    // GIVEN a non-readonly pipeline
    const givenPipelineId = "pipeline-recruiter-tuning";

    // WHEN we delete it
    await deletePipeline(givenPipelineId);

    // THEN the list no longer contains it
    const listed = await listPipelines();
    expect(
      listed.pipelines.some(
        (pipeline) => pipeline.pipeline_id === givenPipelineId,
      ),
    ).toBe(false);
  });

  it("rejects with 409 on readonly pipeline", async () => {
    await expect(deletePipeline("pipeline-default")).rejects.toMatchObject({
      status: 409,
    });
  });
});

describe("activatePipeline", () => {
  it("flips exactly one pipeline to is_active=true", async () => {
    // GIVEN Default Tabiya is active by default
    const givenPipelineId = "pipeline-recruiter-tuning";

    // WHEN we activate the other pipeline
    const activated = await activatePipeline(givenPipelineId);

    // THEN it's active and Default Tabiya isn't anymore
    expect(activated.is_active).toBe(true);
    const listed = await listPipelines();
    const otherActive = listed.pipelines.filter(
      (pipeline) => pipeline.is_active,
    );
    expect(otherActive.map((pipeline) => pipeline.pipeline_id)).toEqual([
      givenPipelineId,
    ]);
  });

  it("rejects with 404 for missing id", async () => {
    await expect(activatePipeline("ghost")).rejects.toMatchObject({
      status: 404,
    });
  });
});

describe("clonePipeline", () => {
  it("returns a new pipeline whose name gets the (copy) suffix", async () => {
    // GIVEN a source pipeline
    const givenPipelineId = "pipeline-recruiter-tuning";

    // WHEN we clone it
    const cloned = await clonePipeline(givenPipelineId);

    // THEN the clone has a new id, a suffixed name, and is not active/readonly/default
    expect(cloned.pipeline_id).not.toBe(givenPipelineId);
    expect(cloned.name.endsWith("(copy)")).toBe(true);
    expect(cloned.is_active).toBe(false);
    expect(cloned.is_readonly).toBe(false);
    expect(cloned.is_default).toBe(false);
  });
});
