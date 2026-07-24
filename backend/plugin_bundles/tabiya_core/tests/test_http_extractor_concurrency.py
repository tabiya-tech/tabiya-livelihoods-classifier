"""Concurrency guard for the NER plugin's HTTP extractor.

`HttpEntityExtractor` calls the legacy NER service with an
`httpx.AsyncClient`. Because it's async, many `extract()` calls awaited
concurrently share one event loop and their downstream HTTP round-trips
overlap. If the client were ever swapped back to a blocking `httpx.Client`,
each call would block the loop and the calls would serialize.

This test pins the behaviour with a `MockTransport` handler that awaits a
fixed sleep per request, fires N concurrent `extract()` calls, and asserts
the total wall-clock stays far below N × the per-call sleep. A blocking
client cannot even be awaited concurrently, so a regression fails loudly.

Every test uses GIVEN/WHEN/THEN inline comments and named `given*` /
`expected*` variables.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from tabiya_core.plugins.ner.http_extractor import HttpEntityExtractor


def _make_sleeping_client(sleep_seconds: float) -> httpx.AsyncClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(sleep_seconds)
        return httpx.Response(200, json={"entities": [], "metadata": {}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_concurrent_extracts_overlap_instead_of_serializing() -> None:
    # GIVEN an extractor whose downstream NER call takes 0.3s and N concurrent calls
    givenBaseUrl = "http://ner-service:5002"
    givenSleepSeconds = 0.3
    givenConcurrentCalls = 10
    givenText = "Data scientist wanted"
    givenExtractor = HttpEntityExtractor(
        base_url=givenBaseUrl,
        http_client=_make_sleeping_client(givenSleepSeconds),
    )

    # AND a serial run would take N × sleep; a 3× ceiling still proves overlap
    # (10 × 0.3s = 3.0s serial vs. ~0.3s concurrent).
    expectedMaxWallClockSeconds = givenSleepSeconds * 3

    # WHEN we await all extracts concurrently on one event loop
    start = time.perf_counter()
    results = await asyncio.gather(
        *[
            givenExtractor.extract(givenText, model_id="tabiya/roberta-base-job-ner")
            for _ in range(givenConcurrentCalls)
        ]
    )
    actualWallClockSeconds = time.perf_counter() - start
    await givenExtractor.close()

    # THEN every call returned (empty entities from the fake service)
    expectedResultCount = givenConcurrentCalls
    assert len(results) == expectedResultCount

    # AND the total time is far below the serial floor — the async client let
    # the downstream round-trips overlap rather than blocking the loop.
    assert actualWallClockSeconds < expectedMaxWallClockSeconds
