/**
 * MSW handlers for the NEL v2 backend.
 *
 * - `GET /v2/nel/models` and `GET /v2/nel/taxonomy-models` are static reads
 *   off the canonical fixtures.
 * - `GET /v2/nel/user/config` and `PUT /v2/nel/user/config` share an
 *   in-memory store so a PUT actually mutates and the next GET reflects it.
 *   Tests reset the store via {@link resetNelV2HandlersStore}.
 */

import { HttpResponse, http } from "msw";
import { NEL_V2_API_BASE_URL } from "@/lib/api";
import type { V2UserConfig } from "@/lib/api";
import {
  fixtureNelModels,
  fixtureTaxonomyModels,
  fixtureUserConfig,
} from "../fixtures/nelV2";

/** In-memory user-config so PUT survives until the next test reset. */
let currentUserConfig: V2UserConfig = { ...fixtureUserConfig };

/**
 * Resets the in-memory user-config back to the canonical fixture.
 * Vitest's afterEach calls server.resetHandlers() but that doesn't clear
 * module state — tests that mutate user config should call this in their
 * own setup.
 */
export function resetNelV2HandlersStore() {
  currentUserConfig = { ...fixtureUserConfig };
}

const nelModelsUrl = `${NEL_V2_API_BASE_URL}/v2/nel/models`;
const taxonomyModelsUrl = `${NEL_V2_API_BASE_URL}/v2/nel/taxonomy-models`;
const userConfigUrl = `${NEL_V2_API_BASE_URL}/v2/nel/user/config`;

export const nelV2Handlers = [
  http.get(nelModelsUrl, () => HttpResponse.json(fixtureNelModels)),
  http.get(taxonomyModelsUrl, () => HttpResponse.json(fixtureTaxonomyModels)),
  http.get(userConfigUrl, () => HttpResponse.json(currentUserConfig)),
  http.put(userConfigUrl, async ({ request }) => {
    const body = (await request.json()) as V2UserConfig;
    currentUserConfig = body;
    return HttpResponse.json(currentUserConfig);
  }),
];
