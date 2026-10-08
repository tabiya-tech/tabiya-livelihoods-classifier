/**
 * Storybook-only override plumbing for the API Keys page's hooks.
 *
 * Production never wraps the page in these providers. Each hook reads its
 * override context; when present it short-circuits the real fetch and
 * returns the canned snapshot the story provided.
 */

import { createContext, type ReactNode } from "react";
import type { ApiKeysSnapshot } from "./useApiKeys";
import type { CreateApiKeyState } from "./useCreateApiKey";
import type { RevokeApiKeyState } from "./useRevokeApiKey";

export const ApiKeysOverrideContext = createContext<ApiKeysSnapshot | null>(
  null,
);
export const CreateApiKeyOverrideContext =
  createContext<CreateApiKeyState | null>(null);
export const RevokeApiKeyOverrideContext =
  createContext<RevokeApiKeyState | null>(null);

export interface ApiKeysOverridesProviderProps {
  apiKeys?: ApiKeysSnapshot;
  createApiKey?: CreateApiKeyState;
  revokeApiKey?: RevokeApiKeyState;
  children: ReactNode;
}

export function ApiKeysOverridesProvider({
  apiKeys,
  createApiKey,
  revokeApiKey,
  children,
}: ApiKeysOverridesProviderProps) {
  return (
    <ApiKeysOverrideContext.Provider value={apiKeys ?? null}>
      <CreateApiKeyOverrideContext.Provider value={createApiKey ?? null}>
        <RevokeApiKeyOverrideContext.Provider value={revokeApiKey ?? null}>
          {children}
        </RevokeApiKeyOverrideContext.Provider>
      </CreateApiKeyOverrideContext.Provider>
    </ApiKeysOverrideContext.Provider>
  );
}
