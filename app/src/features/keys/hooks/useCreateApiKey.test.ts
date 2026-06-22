import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { CreateApiKeyResponse } from "@/lib/api";
import { useCreateApiKey } from "./useCreateApiKey";

const givenResponse: CreateApiKeyResponse = {
  key: "AIzaSyDEMO",
  meta: {
    key_id: "key-new",
    user_id: "u",
    label: "fresh",
    created_at: 1,
    last_used_at: null,
    revoked: false,
  },
};

describe("useCreateApiKey", () => {
  it("returns the issued key and parks it on justIssued for the banner", async () => {
    // GIVEN a backend that resolves to the canonical response
    const createKey = vi.fn(async () => givenResponse);

    // WHEN we render and submit a label
    const { result } = renderHook(() => useCreateApiKey({ createKey }));
    let returned: CreateApiKeyResponse | undefined;
    await act(async () => {
      returned = await result.current.submit("fresh");
    });

    // THEN the response is returned AND parked on justIssued
    expect(returned).toEqual(givenResponse);
    expect(result.current.status).toBe("success");
    expect(result.current.justIssued).toEqual(givenResponse);
  });

  it("transitions to error and throws when the backend rejects", async () => {
    // GIVEN a backend that rejects
    const givenError = new Error("quota");
    const createKey = vi.fn(async () => {
      throw givenError;
    });

    // WHEN we submit
    const { result } = renderHook(() => useCreateApiKey({ createKey }));
    await act(async () => {
      await expect(result.current.submit("nope")).rejects.toThrow("quota");
    });

    // THEN the status reflects the failure and no key is parked
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe(givenError);
    expect(result.current.justIssued).toBeNull();
  });

  it("invokes the onSuccess callback after a successful submit", async () => {
    // GIVEN an onSuccess spy
    const createKey = vi.fn(async () => givenResponse);
    const onSuccess = vi.fn();

    // WHEN we submit
    const { result } = renderHook(() =>
      useCreateApiKey({ createKey, onSuccess }),
    );
    await act(async () => {
      await result.current.submit("fresh");
    });

    // THEN onSuccess fires with the response
    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
    expect(onSuccess).toHaveBeenCalledWith(givenResponse);
  });

  it("clears justIssued back to null when clearJustIssuedKey is called", async () => {
    // GIVEN a successful submit that parked a key
    const createKey = vi.fn(async () => givenResponse);
    const { result } = renderHook(() => useCreateApiKey({ createKey }));
    await act(async () => {
      await result.current.submit("fresh");
    });
    expect(result.current.justIssued).not.toBeNull();

    // WHEN clearJustIssuedKey is called
    act(() => result.current.clearJustIssuedKey());

    // THEN the banner data is gone and status returns to idle
    expect(result.current.justIssued).toBeNull();
    expect(result.current.status).toBe("idle");
  });
});
