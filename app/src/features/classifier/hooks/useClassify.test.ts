import { describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import type { ClassifyResponse } from "@/lib/api";
import { useClassify } from "./useClassify";

const givenResponse: ClassifyResponse = {
  entities: [],
  metadata: {
    classifier_version: "2.0.0",
    ner_model: "ner",
    nel_model_id: "nel",
    taxonomy_model_id: "tax",
    processing_time_ms: 1,
  },
};

describe("useClassify", () => {
  it("starts idle with no response or error", () => {
    // GIVEN a fresh hook
    const { result } = renderHook(() => useClassify());

    // THEN status=idle, response and error are null
    expect(result.current.status).toBe("idle");
    expect(result.current.response).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("transitions to running then done on success and stores the response", async () => {
    // GIVEN a backend that resolves with the canonical response
    const classifyImpl = vi.fn(async () => givenResponse);

    // WHEN run is called
    const { result } = renderHook(() => useClassify({ classifyImpl }));
    await act(async () => {
      await result.current.run({ text: "anything" });
    });

    // THEN status=done and response is the supplied value
    expect(result.current.status).toBe("done");
    expect(result.current.response).toEqual(givenResponse);
    expect(result.current.error).toBeNull();
  });

  it("transitions to error and rethrows when the backend rejects", async () => {
    // GIVEN a backend that rejects
    const givenError = new Error("backend down");
    const classifyImpl = vi.fn(async () => {
      throw givenError;
    });

    // WHEN run is called
    const { result } = renderHook(() => useClassify({ classifyImpl }));
    await act(async () => {
      await expect(result.current.run({ text: "x" })).rejects.toThrow(
        "backend down",
      );
    });

    // THEN status=error, error is the thrown exception, response stays null
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe(givenError);
    expect(result.current.response).toBeNull();
  });

  it("ignores the older response when a second run supersedes the first", async () => {
    // GIVEN two backend implementations that resolve at controlled times
    let resolveFirst: (value: ClassifyResponse) => void = () => undefined;
    let resolveSecond: (value: ClassifyResponse) => void = () => undefined;
    const firstResponse: ClassifyResponse = {
      ...givenResponse,
      metadata: { ...givenResponse.metadata, processing_time_ms: 1 },
    };
    const secondResponse: ClassifyResponse = {
      ...givenResponse,
      metadata: { ...givenResponse.metadata, processing_time_ms: 2 },
    };
    const classifyImpl = vi
      .fn<(payload: unknown) => Promise<ClassifyResponse>>()
      .mockImplementationOnce(
        () => new Promise((resolve) => { resolveFirst = resolve; }),
      )
      .mockImplementationOnce(
        () => new Promise((resolve) => { resolveSecond = resolve; }),
      );

    const { result } = renderHook(() => useClassify({ classifyImpl }));

    // WHEN we kick off two runs in sequence, then resolve the SECOND first,
    // then resolve the (now-stale) first
    let firstRunPromise!: Promise<ClassifyResponse>;
    let secondRunPromise!: Promise<ClassifyResponse>;
    act(() => {
      firstRunPromise = result.current.run({ text: "first" });
      secondRunPromise = result.current.run({ text: "second" });
    });
    await act(async () => {
      resolveSecond(secondResponse);
      await secondRunPromise;
    });
    await act(async () => {
      resolveFirst(firstResponse);
      await firstRunPromise;
    });

    // THEN the response remains the SECOND one — the stale first never overwrites
    expect(result.current.response).toBe(secondResponse);
    expect(result.current.status).toBe("done");
  });

  it("reset clears state and ignores any still-in-flight runs", async () => {
    // GIVEN a backend that never resolves until we say so
    let resolveLater: (value: ClassifyResponse) => void = () => undefined;
    const classifyImpl = vi.fn(
      () => new Promise<ClassifyResponse>((resolve) => { resolveLater = resolve; }),
    );

    const { result } = renderHook(() => useClassify({ classifyImpl }));

    // WHEN we kick off a run, reset, then resolve the in-flight call
    let pending!: Promise<ClassifyResponse>;
    act(() => {
      pending = result.current.run({ text: "x" });
    });
    act(() => result.current.reset());
    expect(result.current.status).toBe("idle");
    await act(async () => {
      resolveLater(givenResponse);
      await pending;
    });

    // THEN the hook remains idle and never adopted that stale response
    expect(result.current.status).toBe("idle");
    expect(result.current.response).toBeNull();
  });
});
