import pytest
from nel.app.retry import RetryPolicy


class TestRetryPolicy:
    async def test_succeeds_on_first_attempt(self):
        # GIVEN a function that always succeeds
        calls = []
        async def fn():
            calls.append(1)
            return "ok"

        # WHEN run with retry
        result = await RetryPolicy(attempts=3, backoff=0.0, on=(Exception,)).run(fn)

        # THEN called once and result returned
        assert result == "ok"
        assert len(calls) == 1

    async def test_retries_on_matching_exception(self):
        # GIVEN a function that fails twice then succeeds
        calls = []
        async def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("transient")
            return "ok"

        result = await RetryPolicy(attempts=3, backoff=0.0, on=(ValueError,)).run(fn)

        assert result == "ok"
        assert len(calls) == 3

    async def test_raises_immediately_on_non_matching_exception(self):
        # GIVEN a function that raises a TypeError (not in the retry list)
        calls = []
        async def fn():
            calls.append(1)
            raise TypeError("permanent")

        with pytest.raises(TypeError):
            await RetryPolicy(attempts=3, backoff=0.0, on=(ValueError,)).run(fn)

        # THEN only called once — no retry
        assert len(calls) == 1

    async def test_raises_after_all_attempts_exhausted(self):
        # GIVEN a function that always fails
        calls = []
        async def fn():
            calls.append(1)
            raise ValueError("always fails")

        with pytest.raises(ValueError):
            await RetryPolicy(attempts=3, backoff=0.0, on=(ValueError,)).run(fn)

        assert len(calls) == 3

    async def test_attempts_1_means_no_retry(self):
        # GIVEN attempts=1
        calls = []
        async def fn():
            calls.append(1)
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await RetryPolicy(attempts=1, backoff=0.0, on=(ValueError,)).run(fn)

        assert len(calls) == 1

    def test_invalid_attempts_raises(self):
        with pytest.raises(ValueError):
            RetryPolicy(attempts=0)
