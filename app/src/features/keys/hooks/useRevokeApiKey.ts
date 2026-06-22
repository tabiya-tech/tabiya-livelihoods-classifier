/**
 * "Revoke key" lifecycle. Tracks which key id is in-flight so the row can
 * show a spinner and the modal confirm button can disable itself.
 */

import { useCallback, useContext, useState } from "react";
import { deleteApiKey } from "@/lib/api";
import { RevokeApiKeyOverrideContext } from "./apiKeysOverrides";

export type RevokeApiKeyStatus = "idle" | "submitting" | "error";

export interface RevokeApiKeyState {
  status: RevokeApiKeyStatus;
  error: Error | null;
  /** key_id currently being revoked, or null. */
  pendingKeyId: string | null;
  /** Submit a revoke. Rejects on backend failure. */
  revoke: (keyId: string) => Promise<void>;
}

export interface UseRevokeApiKeyOptions {
  /** Test seam — override the backend mutation. Defaults to the real client. */
  revokeKey?: (keyId: string) => Promise<void>;
  /** Optional side-effect after a successful revoke — typically a list refetch. */
  onSuccess?: (keyId: string) => void | Promise<void>;
}

export function useRevokeApiKey({
  revokeKey = deleteApiKey,
  onSuccess,
}: UseRevokeApiKeyOptions = {}): RevokeApiKeyState {
  const override = useContext(RevokeApiKeyOverrideContext);
  const [status, setStatus] = useState<RevokeApiKeyStatus>("idle");
  const [error, setError] = useState<Error | null>(null);
  const [pendingKeyId, setPendingKeyId] = useState<string | null>(null);

  const revoke = useCallback(
    async (keyId: string) => {
      setStatus("submitting");
      setPendingKeyId(keyId);
      setError(null);
      try {
        await revokeKey(keyId);
        setStatus("idle");
        setPendingKeyId(null);
        if (onSuccess) await onSuccess(keyId);
      } catch (caught: unknown) {
        const caughtError =
          caught instanceof Error ? caught : new Error(String(caught));
        setError(caughtError);
        setStatus("error");
        setPendingKeyId(null);
        throw caughtError;
      }
    },
    [revokeKey, onSuccess],
  );

  if (override) return override;
  return { status, error, pendingKeyId, revoke };
}
