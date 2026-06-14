import { describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { ApiError } from "@/lib/api";
import { useApiHealth } from "./useApiHealth";

describe("useApiHealth", () => {
  it("starts with status='unknown' until the first response", () => {
    // GIVEN a fetchHealth that never resolves
    const fetchHealth = vi.fn(() => new Promise<never>(() => {}));

    // WHEN we mount the hook
    const { result } = renderHook(() =>
      useApiHealth({ fetchHealth, pollIntervalMs: 10_000 }),
    );

    // THEN the initial snapshot reports an unknown status with no timestamp
    expect(result.current.status).toBe("unknown");
    expect(result.current.lastCheckedAt).toBeNull();
  });

  it("reports status='healthy' when /v1/health returns healthy", async () => {
    // GIVEN a successful healthy response with a version
    const givenVersion = "1.0.0";
    const fetchHealth = vi.fn(async () => ({
      status: "healthy" as const,
      version: givenVersion,
    }));

    // WHEN we mount the hook
    const { result } = renderHook(() =>
      useApiHealth({ fetchHealth, pollIntervalMs: 10_000 }),
    );

    // THEN the hook eventually exposes a healthy snapshot with the given version
    await waitFor(() => expect(result.current.status).toBe("healthy"));
    expect(result.current.version).toBe(givenVersion);
    expect(result.current.lastCheckedAt).toBeInstanceOf(Date);
  });

  it("reports status='degraded' when an ApiError is thrown", async () => {
    // GIVEN a fetchHealth that throws an ApiError (e.g. 503)
    const fetchHealth = vi.fn(async () => {
      throw new ApiError(503, "service unavailable");
    });

    // WHEN we mount the hook
    const { result } = renderHook(() =>
      useApiHealth({ fetchHealth, pollIntervalMs: 10_000 }),
    );

    // THEN the snapshot reports degraded
    await waitFor(() => expect(result.current.status).toBe("degraded"));
  });

  it("reports status='down' on network errors", async () => {
    // GIVEN a fetchHealth that throws a generic network error
    const fetchHealth = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });

    // WHEN we mount the hook
    const { result } = renderHook(() =>
      useApiHealth({ fetchHealth, pollIntervalMs: 10_000 }),
    );

    // THEN the snapshot reports down
    await waitFor(() => expect(result.current.status).toBe("down"));
  });
});
