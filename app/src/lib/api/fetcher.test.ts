import { describe, expect, it, vi } from "vitest";
import { ApiError, request } from "./fetcher";

type FetchSignature = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

/** Helper to build a fake `fetch` that returns a sequence of pre-canned responses. */
function buildSequentialFetch(responses: Response[]) {
  let callCount = 0;
  const fetchImpl = vi.fn<FetchSignature>(async () => {
    const next = responses[callCount];
    callCount += 1;
    if (!next) throw new Error("fetch called more times than expected");
    return next;
  });
  return fetchImpl;
}

function buildJsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function buildEmptyResponse(status: number): Response {
  return new Response(null, { status });
}

describe("fetcher.request", () => {
  it("attaches a Bearer token from the context and returns parsed JSON", async () => {
    // GIVEN an id token source and an endpoint that returns JSON
    const givenIdToken = "id-token-cached";
    const givenResponseBody = { ok: true, value: 42 };
    const getIdToken = vi.fn(async () => givenIdToken);
    const fetchImpl = buildSequentialFetch([buildJsonResponse(givenResponseBody)]);

    // WHEN we call request with that context
    const actualResponseBody = await request<typeof givenResponseBody>("/v2/example", {
      context: { getIdToken, fetchImpl },
    });

    // THEN the response is parsed and the Authorization header carried the cached token
    expect(actualResponseBody).toEqual(givenResponseBody);
    expect(getIdToken).toHaveBeenCalledWith(false);
    const [, init] = fetchImpl.mock.calls[0];
    expect((init?.headers as Record<string, string>).Authorization).toBe(
      `Bearer ${givenIdToken}`,
    );
  });

  it("refreshes the id token once and retries when the first response is 401", async () => {
    // GIVEN two id tokens (cached + refreshed) and a 401-then-200 sequence
    const givenCachedIdToken = "cached-token";
    const givenRefreshedIdToken = "refreshed-token";
    const givenResponseBody = { ok: true };
    const getIdToken = vi
      .fn<(forceRefresh?: boolean) => Promise<string>>()
      .mockResolvedValueOnce(givenCachedIdToken)
      .mockResolvedValueOnce(givenRefreshedIdToken);
    const fetchImpl = buildSequentialFetch([
      buildEmptyResponse(401),
      buildJsonResponse(givenResponseBody),
    ]);

    // WHEN we call request
    const actualResponseBody = await request<typeof givenResponseBody>("/v2/example", {
      context: { getIdToken, fetchImpl },
    });

    // THEN the second attempt succeeded with the refreshed token
    expect(actualResponseBody).toEqual(givenResponseBody);
    expect(getIdToken.mock.calls).toEqual([[false], [true]]);

    const [, secondInit] = fetchImpl.mock.calls[1];
    expect((secondInit?.headers as Record<string, string>).Authorization).toBe(
      `Bearer ${givenRefreshedIdToken}`,
    );
  });

  it("surfaces ApiError when a second 401 follows the refresh", async () => {
    // GIVEN two 401 responses in a row
    const getIdToken = vi.fn(async () => "any-token");
    const fetchImpl = buildSequentialFetch([
      buildEmptyResponse(401),
      buildEmptyResponse(401),
    ]);

    // WHEN we call request
    const requestPromise = request<unknown>("/v2/example", {
      context: { getIdToken, fetchImpl },
    });

    // THEN the call rejects with an ApiError whose status reflects the failure
    await expect(requestPromise).rejects.toBeInstanceOf(ApiError);
    await expect(requestPromise).rejects.toMatchObject({ status: 401 });
  });

  it("does not refresh the token on non-401 errors", async () => {
    // GIVEN an id token source and a 500 response
    const getIdToken = vi.fn(async () => "any-token");
    const fetchImpl = buildSequentialFetch([buildEmptyResponse(500)]);

    // WHEN we call request
    const requestPromise = request<unknown>("/v2/example", {
      context: { getIdToken, fetchImpl },
    });

    // THEN the call rejects with ApiError and the token was fetched exactly once
    await expect(requestPromise).rejects.toBeInstanceOf(ApiError);
    expect(getIdToken).toHaveBeenCalledTimes(1);
    expect(getIdToken).toHaveBeenCalledWith(false);
  });

  it("omits the Authorization header when no token is available (no signed-in user)", async () => {
    // GIVEN a token source that yields null (e.g. Storybook / not signed in)
    const getIdToken = vi.fn(async () => null);
    const fetchImpl = buildSequentialFetch([buildJsonResponse({ ok: true })]);

    // WHEN we call request
    await request<{ ok: boolean }>("/v2/example", {
      context: { getIdToken, fetchImpl },
    });

    // THEN the request still goes out, with NO Authorization header
    const [, init] = fetchImpl.mock.calls[0];
    expect((init?.headers as Record<string, string>).Authorization).toBeUndefined();
  });

  it("returns undefined for 204 No Content responses", async () => {
    // GIVEN a 204 response
    const getIdToken = vi.fn(async () => "tok");
    const fetchImpl = buildSequentialFetch([buildEmptyResponse(204)]);

    // WHEN we call request
    const actualResponseBody = await request<void>("/v2/no-body", {
      context: { getIdToken, fetchImpl },
    });

    // THEN the call resolves to undefined without trying to parse JSON
    expect(actualResponseBody).toBeUndefined();
  });
});
