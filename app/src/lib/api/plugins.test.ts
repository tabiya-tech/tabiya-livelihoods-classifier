/**
 * Client tests for /v2/plugins* against the shared MSW handlers.
 *
 * Every test uses GIVEN/WHEN/THEN inline comments and named
 * `given*` / `expected*` variables per project convention.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { auth } from "@/lib/firebase";
import { getPlugin, getPluginOptions, listPlugins } from "@/lib/api";

// Patch `auth.currentUser` so `defaultIdTokenSource` in fetcher.ts finds a
// user and returns a deterministic token. Restored after each test so the
// suite is independent of any cross-file Firebase state.
let originalCurrentUser: unknown;

beforeEach(() => {
  originalCurrentUser = (auth as { currentUser: unknown }).currentUser;
  (auth as { currentUser: unknown }).currentUser = {
    getIdToken: async () => "test-id-token",
  };
});

afterEach(() => {
  (auth as { currentUser: unknown }).currentUser = originalCurrentUser;
});

describe("listPlugins", () => {
  it("returns the full palette from the shipped fixtures", async () => {
    // GIVEN the MSW handlers are seeded from `fixturePluginSummaries`.

    // WHEN we call the client
    const response = await listPlugins();

    // THEN we get the catalog entries in the same order the backend serves,
    // including the coming-soon plugins (which now ship real manifests).
    const expectedIds = [
      "tabiya.ner.v1",
      "tabiya.nel.v1",
      "tabiya.source.text.v1",
      "tabiya.source.json_entities.v1",
      "tabiya.sink.results.v1",
      "tabiya.source.scraper.v1",
      "tabiya.transform.stopwords.v1",
      "tabiya.sink.database.v1",
      "tabiya.branching.language_router.v1",
    ];
    expect(response.plugins.map((plugin) => plugin.plugin_id)).toEqual(
      expectedIds,
    );
  });
});

describe("getPlugin", () => {
  it("returns the full manifest for an enabled plugin", async () => {
    // GIVEN an enabled plugin_id
    const givenPluginId = "tabiya.ner.v1";

    // WHEN we fetch its detail
    const response = await getPlugin(givenPluginId);

    // THEN status is enabled and the manifest matches
    const expectedStatus = "enabled";
    expect(response.status).toBe(expectedStatus);
    expect(response.manifest).not.toBeNull();
    expect(response.manifest?.plugin_id).toBe(givenPluginId);
  });

  it("returns manifest=null with status=unavailable for coming_soon plugins", async () => {
    // GIVEN a coming_soon plugin_id
    const givenPluginId = "tabiya.source.scraper.v1";

    // WHEN we fetch its detail
    const response = await getPlugin(givenPluginId);

    // THEN we get status=unavailable and no manifest
    const expectedStatus = "unavailable";
    expect(response.status).toBe(expectedStatus);
    expect(response.coming_soon).toBe(true);
    expect(response.manifest).toBeNull();
  });

  it("rejects with ApiError on 404 for unknown plugin_id", async () => {
    // GIVEN a plugin_id that isn't in the catalog
    const givenPluginId = "tabiya.ghost.v1";

    // WHEN we fetch its detail
    // THEN the client rejects with a status-404 ApiError
    await expect(getPlugin(givenPluginId)).rejects.toMatchObject({
      status: 404,
    });
  });
});

describe("getPluginOptions", () => {
  it("returns the normalised {value,label} options for a real x-source", async () => {
    // GIVEN a plugin + field with fixtures wired
    const givenPluginId = "tabiya.nel.v1";
    const givenField = "nel_model_id";

    // WHEN we fetch its options
    const response = await getPluginOptions(givenPluginId, givenField);

    // THEN the response echoes the field and returns the seeded options
    expect(response.field).toBe(givenField);
    const expectedValues = ["all-MiniLM-L6-v2", "sentence-t5-base"];
    expect(response.options.map((option) => option.value)).toEqual(
      expectedValues,
    );
  });

  it("rejects with ApiError on 404 for unknown field", async () => {
    // GIVEN a valid plugin_id but a field without x-source configured
    const givenPluginId = "tabiya.nel.v1";
    const givenField = "does_not_exist";

    // WHEN we fetch options
    // THEN 404 surfaces as ApiError
    await expect(getPluginOptions(givenPluginId, givenField)).rejects.toMatchObject(
      {
        status: 404,
      },
    );
  });
});
