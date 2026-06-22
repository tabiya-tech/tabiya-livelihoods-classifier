import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useRevokeApiKey } from "./useRevokeApiKey";

describe("useRevokeApiKey", () => {
  it("calls the backend and clears pendingKeyId on success", async () => {
    // GIVEN a revoker that resolves
    const revokeKey = vi.fn(async () => undefined);
    const givenKeyId = "key-123";

    // WHEN we render and revoke a key
    const { result } = renderHook(() => useRevokeApiKey({ revokeKey }));
    await act(async () => {
      await result.current.revoke(givenKeyId);
    });

    // THEN the revoker was called with that id and state is idle
    expect(revokeKey).toHaveBeenCalledWith(givenKeyId);
    expect(result.current.status).toBe("idle");
    expect(result.current.pendingKeyId).toBeNull();
  });

  it("surfaces backend errors via status='error' and rethrows", async () => {
    // GIVEN a revoker that rejects
    const givenError = new Error("nope");
    const revokeKey = vi.fn(async () => {
      throw givenError;
    });

    // WHEN we revoke
    const { result } = renderHook(() => useRevokeApiKey({ revokeKey }));
    await act(async () => {
      await expect(result.current.revoke("key-x")).rejects.toThrow("nope");
    });

    // THEN the error surfaces and pendingKeyId resets
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe(givenError);
    expect(result.current.pendingKeyId).toBeNull();
  });

  it("fires onSuccess after a successful revoke", async () => {
    // GIVEN an onSuccess spy
    const revokeKey = vi.fn(async () => undefined);
    const onSuccess = vi.fn();
    const givenKeyId = "key-777";

    // WHEN we revoke
    const { result } = renderHook(() =>
      useRevokeApiKey({ revokeKey, onSuccess }),
    );
    await act(async () => {
      await result.current.revoke(givenKeyId);
    });

    // THEN onSuccess fires with the revoked key id
    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
    expect(onSuccess).toHaveBeenCalledWith(givenKeyId);
  });
});
