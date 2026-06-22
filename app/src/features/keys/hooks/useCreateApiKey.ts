/**
 * Owns the "create key" lifecycle. The just-issued plaintext key is held
 * here so the page can render the reveal-once banner; the caller dismisses
 * via {@link clearJustIssuedKey}.
 */

import { useCallback, useContext, useState } from "react";
import { createApiKey, type CreateApiKeyResponse } from "@/lib/api";
import { CreateApiKeyOverrideContext } from "./apiKeysOverrides";

export type CreateApiKeyStatus = "idle" | "submitting" | "success" | "error";

export interface CreateApiKeyState {
  status: CreateApiKeyStatus;
  error: Error | null;
  /** The plaintext key + metadata returned by the most recent successful POST. */
  justIssued: CreateApiKeyResponse | null;
  /** Submit a new key. Rejects on backend failure. */
  submit: (label: string) => Promise<CreateApiKeyResponse>;
  /** Dismiss the reveal-once banner without affecting other state. */
  clearJustIssuedKey: () => void;
}

export interface UseCreateApiKeyOptions {
  /** Test seam — override the backend mutation. Defaults to the real client. */
  createKey?: (label: string) => Promise<CreateApiKeyResponse>;
  /**
   * Optional side-effect after a successful create — typically the page's
   * refetch on the list. Caller is responsible for awaiting.
   */
  onSuccess?: (response: CreateApiKeyResponse) => void | Promise<void>;
}

export function useCreateApiKey({
  createKey = createApiKey,
  onSuccess,
}: UseCreateApiKeyOptions = {}): CreateApiKeyState {
  const override = useContext(CreateApiKeyOverrideContext);
  const [status, setStatus] = useState<CreateApiKeyStatus>("idle");
  const [error, setError] = useState<Error | null>(null);
  const [justIssued, setJustIssued] = useState<CreateApiKeyResponse | null>(
    null,
  );

  const submit = useCallback(
    async (label: string) => {
      setStatus("submitting");
      setError(null);
      try {
        const response = await createKey(label);
        setJustIssued(response);
        setStatus("success");
        if (onSuccess) await onSuccess(response);
        return response;
      } catch (caught: unknown) {
        const caughtError =
          caught instanceof Error ? caught : new Error(String(caught));
        setError(caughtError);
        setStatus("error");
        throw caughtError;
      }
    },
    [createKey, onSuccess],
  );

  const clearJustIssuedKey = useCallback(() => {
    setJustIssued(null);
    setStatus((previous) => (previous === "success" ? "idle" : previous));
  }, []);

  if (override) return override;
  return { status, error, justIssued, submit, clearJustIssuedKey };
}
