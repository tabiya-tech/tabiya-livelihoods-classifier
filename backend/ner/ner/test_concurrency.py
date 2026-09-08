"""Concurrency guard for the NER route.

The model forward pass is synchronous, CPU-bound and GIL-holding, so
`main.extract_entities` offloads it with `asyncio.to_thread(...)`. If that
offload is ever removed and the blocking call runs inline on the event
loop, one slow request stalls the whole process and concurrent requests
serialize.

This test pins that behaviour: a deliberately *blocking* (`time.sleep`)
fake service stands in for real inference; N requests are fired
concurrently against one ASGI app (one event loop); the total wall-clock
is asserted to be far below N × the per-request block time. With the
thread offload the blocking calls run on the thread pool and overlap;
without it they serialize and the assertion fails loudly.

Every test uses GIVEN/WHEN/THEN inline comments and named `given*` /
`expected*` variables.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from ner.get_ner_service import get_ner_service
from ner.main import app
from ner.models import NERMetadata, NERResponse
from ner.service import INERService


class _BlockingNERService(INERService):
    """Fake service whose extract call blocks the calling thread for a fixed time.

    This mimics the real torch forward pass: synchronous and GIL-holding.
    The route must run it off the event loop for concurrent requests to
    overlap.
    """

    def __init__(self, block_seconds: float) -> None:
        self._block_seconds = block_seconds

    def extract_entities(self, text, entity_types=None, language=None) -> NERResponse:
        time.sleep(self._block_seconds)
        return NERResponse(
            entities=[],
            metadata=NERMetadata(
                model_name="fake/blocking-ner",
                entity_count=0,
                processing_time_ms=self._block_seconds * 1000.0,
            ),
        )


def _make_async_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://ner.test")


@pytest.fixture
def blocking_service():
    givenBlockSeconds = 0.3
    givenService = _BlockingNERService(block_seconds=givenBlockSeconds)
    app.dependency_overrides[get_ner_service] = lambda: givenService
    yield givenService, givenBlockSeconds
    app.dependency_overrides.clear()


async def test_concurrent_requests_overlap_instead_of_serializing(blocking_service) -> None:
    # GIVEN a service that blocks 0.3s per call and N concurrent requests
    _, givenBlockSeconds = blocking_service
    givenConcurrentRequests = 10
    givenText = "Data scientist wanted"

    # AND a serial execution would take N × block time; a generous ceiling of
    # 3× a single block still proves interleaving happened (10 × 0.3s = 3.0s
    # serial vs. ~0.3s concurrent — 0.9s leaves ample slack for scheduling).
    expectedMaxWallClockSeconds = givenBlockSeconds * 3

    # WHEN we fire all requests concurrently on one event loop
    async with _make_async_client() as client:
        start = time.perf_counter()
        responses = await asyncio.gather(
            *[
                client.post("/v1/ner", json={"text": givenText})
                for _ in range(givenConcurrentRequests)
            ]
        )
        actualWallClockSeconds = time.perf_counter() - start

    # THEN every request succeeded
    expectedStatus = 200
    assert all(response.status_code == expectedStatus for response in responses)

    # AND the total time is far below the serial floor — the blocking calls
    # were offloaded and overlapped rather than stalling the loop one by one.
    assert actualWallClockSeconds < expectedMaxWallClockSeconds


async def test_health_check_stays_responsive_while_inference_is_in_flight(blocking_service) -> None:
    # GIVEN a slow (blocking) inference request already in flight
    _, givenBlockSeconds = blocking_service
    givenText = "Head Chef who can plan menus"

    # AND a ceiling well under the block time — a responsive health check must
    # return long before the in-flight inference finishes.
    expectedHealthMaxSeconds = givenBlockSeconds / 2

    async with _make_async_client() as client:
        # WHEN we start the slow inference without awaiting it, then hit /health
        inferenceTask = asyncio.ensure_future(
            client.post("/v1/ner", json={"text": givenText})
        )
        await asyncio.sleep(0)  # let the inference request reach the route

        start = time.perf_counter()
        healthResponse = await client.get("/v1/health")
        actualHealthSeconds = time.perf_counter() - start

        await inferenceTask  # drain the in-flight request

    # THEN the health check returned promptly — the event loop was never
    # blocked by the inference running on the thread pool.
    assert actualHealthSeconds < expectedHealthMaxSeconds
