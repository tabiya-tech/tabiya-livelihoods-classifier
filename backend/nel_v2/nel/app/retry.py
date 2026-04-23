"""Shared retry infrastructure.

Usage:
    retry = RetryPolicy(attempts=3, backoff=2.0, on=(httpx.TransportError,))
    result = await retry.run(lambda: some_async_call())

Pass RetryPolicy(attempts=1) in tests to disable retries.
"""

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

_logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions that are always considered transient (extended per-site via RetryPolicy.on)
_DEFAULT_TRANSIENT: tuple[type[BaseException], ...] = ()


class RetryPolicy:
    def __init__(
        self,
        attempts: int = 3,
        backoff: float = 2.0,
        on: tuple[type[BaseException], ...] = _DEFAULT_TRANSIENT,
    ):
        """
        Args:
            attempts:  Total number of attempts (1 = no retry).
            backoff:   Seconds to wait before the next attempt; doubles each time.
            on:        Exception types that trigger a retry. Any other exception
                       propagates immediately without retrying.
        """
        if attempts < 1:
            raise ValueError("attempts must be >= 1")
        self.attempts = attempts
        self.backoff = backoff
        self.on = on

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Call fn(), retrying on self.on exceptions up to self.attempts times."""
        wait = self.backoff
        last_exc: BaseException | None = None

        for attempt in range(1, self.attempts + 1):
            try:
                return await fn()
            except BaseException as exc:
                if not self.on or not isinstance(exc, self.on):
                    raise
                last_exc = exc
                if attempt < self.attempts:
                    _logger.warning(
                        "Attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                        attempt, self.attempts, type(exc).__name__, exc, wait,
                    )
                    await asyncio.sleep(wait)
                    wait *= 2
                else:
                    _logger.error(
                        "All %d attempts failed (%s: %s)",
                        self.attempts, type(exc).__name__, exc,
                    )

        raise last_exc  # type: ignore[misc]
