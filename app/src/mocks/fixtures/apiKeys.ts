/**
 * Canonical fixtures for the /v2/user/api-keys endpoints. Shared between
 * Storybook (via msw-storybook-addon) and Vitest tests.
 */

import type { ApiKeyMetadata } from "@/lib/api";

export const fixtureApiKeyMetadata: ApiKeyMetadata[] = [
  {
    key_id: "key-001",
    user_id: "local-user",
    label: "analyst-laptop",
    created_at: 1_700_000_000,
    last_used_at: 1_700_100_000,
    revoked: false,
  },
  {
    key_id: "key-002",
    user_id: "local-user",
    label: "ci-pipeline",
    created_at: 1_700_050_000,
    last_used_at: null,
    revoked: false,
  },
];

/** A plaintext key value to surface from POST. Looks like a real GCP key. */
export const fixtureCreatedApiKey = "AIzaSyDEMO0000000000000000000000000000000";
