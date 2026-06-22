import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ApiKeyMetadata } from "@/lib/api";
import { useApiKeys } from "./useApiKeys";

const givenKeys: ApiKeyMetadata[] = [
  {
    key_id: "key-001",
    user_id: "u",
    label: "laptop",
    created_at: 1,
    last_used_at: null,
    revoked: false,
  },
];

describe("useApiKeys", () => {
  it("resolves to status='ready' with the fetched keys", async () => {
    // GIVEN a fetcher that resolves to a single key
    const fetchKeys = vi.fn(async () => ({ keys: givenKeys }));

    // WHEN we render the hook
    const { result } = renderHook(() => useApiKeys({ fetchKeys }));

    // THEN we end in ready with the supplied keys
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.keys).toEqual(givenKeys);
    expect(result.current.error).toBeNull();
  });

  it("transitions to status='error' on fetch failure", async () => {
    // GIVEN a fetcher that rejects
    const givenError = new Error("boom");
    const fetchKeys = vi.fn(async () => {
      throw givenError;
    });

    // WHEN we render the hook
    const { result } = renderHook(() => useApiKeys({ fetchKeys }));

    // THEN we end in error with that exception surfaced
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.keys).toEqual([]);
    expect(result.current.error).toBe(givenError);
  });

  it("re-runs the fetcher when refetch is invoked", async () => {
    // GIVEN a fetcher we can inspect for invocation count
    const fetchKeys = vi.fn(async () => ({ keys: givenKeys }));
    const { result } = renderHook(() => useApiKeys({ fetchKeys }));
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(fetchKeys).toHaveBeenCalledTimes(1);

    // WHEN refetch is called
    await act(async () => {
      await result.current.refetch();
    });

    // THEN the fetcher fires again
    expect(fetchKeys).toHaveBeenCalledTimes(2);
  });
});
