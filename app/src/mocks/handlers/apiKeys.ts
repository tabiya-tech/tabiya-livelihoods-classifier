/**
 * MSW handlers for /v2/user/api-keys against an in-memory store so a POST
 * actually mutates and the next GET reflects it.
 *
 * Tests reset the store via {@link resetApiKeysHandlersStore}.
 */

import { http, HttpResponse } from "msw";
import { API_BASE_URL } from "@/lib/api";
import type { ApiKeyMetadata } from "@/lib/api";
import {
  fixtureApiKeyMetadata,
  fixtureCreatedApiKey,
} from "../fixtures/apiKeys";

let currentKeys: ApiKeyMetadata[] = [...fixtureApiKeyMetadata];

/** Reset the in-memory store back to the canonical fixtures. */
export function resetApiKeysHandlersStore() {
  currentKeys = [...fixtureApiKeyMetadata];
}

/** Replace the in-memory store with the supplied list (used for stories). */
export function seedApiKeysHandlersStore(seed: ApiKeyMetadata[]) {
  currentKeys = [...seed];
}

const listUrl = `${API_BASE_URL}/v2/user/api-keys`;
const itemUrl = `${API_BASE_URL}/v2/user/api-keys/:keyId`;

let nextKeyId = 1000;

export const apiKeysHandlers = [
  http.get(listUrl, () => HttpResponse.json({ keys: currentKeys })),
  http.post(listUrl, async ({ request }) => {
    const body = (await request.json()) as { label: string };
    const issuedKeyId = `key-${String(nextKeyId).padStart(4, "0")}`;
    nextKeyId += 1;
    const meta: ApiKeyMetadata = {
      key_id: issuedKeyId,
      user_id: "local-user",
      label: body.label,
      created_at: Math.floor(Date.now() / 1000),
      last_used_at: null,
      revoked: false,
    };
    currentKeys = [...currentKeys, meta];
    return HttpResponse.json(
      { key: fixtureCreatedApiKey, meta },
      { status: 201 },
    );
  }),
  http.delete(itemUrl, ({ params }) => {
    const targetKeyId = String(params.keyId);
    const before = currentKeys.length;
    currentKeys = currentKeys.filter((meta) => meta.key_id !== targetKeyId);
    if (currentKeys.length === before) {
      return new HttpResponse(null, { status: 404 });
    }
    return new HttpResponse(null, { status: 204 });
  }),
];
