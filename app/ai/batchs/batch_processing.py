
import asyncio
import logging
import random
import time

from collections import deque
from collections.abc import Awaitable, Callable

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------
# BATCH CONFIGURATION
# ---------------------------------------------------------

LOCAL_EMBEDDING_SETTINGS = {
    "batch_size": 32,
    "max_concurrency": 1,
    "requests_per_minute": None,
    "tokens_per_minute": None,
    "max_retries": 1,
}

GEMINI_FREE_TIER_SETTINGS = {
    "batch_size": 20,
    "max_concurrency": 2,
    "requests_per_minute": 80,
    "tokens_per_minute": 25_000,
    "max_retries": 5,
}


# ---------------------------------------------------------
# RETRY AND RATE-LIMIT SETTINGS
# ---------------------------------------------------------

BASE_RETRY_DELAY = 5.0
MAX_RETRY_DELAY = 60.0
RATE_LIMIT_WINDOW = 60.0
CHARS_PER_TOKEN = 4


class BatchProcess:

    def __init__(
        self,
        batch_size: int,
        max_concurrency: int,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        max_retries: int = 3,
        window_seconds: float = RATE_LIMIT_WINDOW
    ):

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")

        if requests_per_minute is not None and requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")

        if tokens_per_minute is not None and tokens_per_minute < 1:
            raise ValueError("tokens_per_minute must be positive")

        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.max_retries = max_retries
        self.window_seconds = window_seconds

        self._window = deque()
        self._lock = asyncio.Lock()

    # -----------------------------------------------------
    # BATCH CREATION
    # -----------------------------------------------------

    def create_batches(self, items: list) -> list[list]:

        return [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

    # -----------------------------------------------------
    # TOKEN ESTIMATION
    # -----------------------------------------------------

    @staticmethod
    def _estimate_tokens(batch: list) -> int:

        return max(
            1,
            sum(len(str(item)) for item in batch) // CHARS_PER_TOKEN
        )

    # -----------------------------------------------------
    # RATE LIMIT MANAGEMENT
    # -----------------------------------------------------

    async def _wait_for_rate_limit(
        self,
        tokens: int,
        stop: asyncio.Event
    ) -> None:

        if (
            self.requests_per_minute is None
            and self.tokens_per_minute is None
        ):
            if stop.is_set():
                raise asyncio.CancelledError()

            return

        if (
            self.tokens_per_minute is not None
            and tokens > self.tokens_per_minute
        ):
            raise ValueError(
                f"Estimated batch tokens ({tokens}) exceed the "
                f"configured token limit ({self.tokens_per_minute}). "
                "Reduce batch_size."
            )

        while True:

            if stop.is_set():
                raise asyncio.CancelledError()

            async with self._lock:

                now = time.monotonic()

                while (
                    self._window
                    and now - self._window[0][0] >= self.window_seconds
                ):
                    self._window.popleft()

                used_tokens = sum(
                    entry[1] for entry in self._window
                )

                requests_ok = (
                    self.requests_per_minute is None
                    or len(self._window) < self.requests_per_minute
                )

                tokens_ok = (
                    self.tokens_per_minute is None
                    or used_tokens + tokens <= self.tokens_per_minute
                )

                if requests_ok and tokens_ok:

                    self._window.append((now, tokens))
                    return

                wait_time = max(
                    0.01,
                    self._window[0][0] + self.window_seconds - now
                )

            logger.info(
                "Rate limit reached. Waiting %.1f seconds.",
                wait_time
            )

            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=wait_time
                )

                raise asyncio.CancelledError()

            except asyncio.TimeoutError:
                continue

    # -----------------------------------------------------
    # RETRY CLASSIFICATION
    # -----------------------------------------------------

    @staticmethod
    def _is_retryable(error: Exception) -> bool:

        # Validation and programming errors should not be retried.
        if isinstance(error, (ValueError, TypeError)):
            return False

        # Gemini SDK errors may expose HTTP status through code.
        status_code = getattr(error, "code", None)

        if status_code is None:
            status_code = getattr(error, "status_code", None)

        if status_code is not None:

            try:
                status_code = int(status_code)
            except (ValueError, TypeError):
                return False

            return (
                status_code == 429
                or status_code in (500, 502, 503, 504)
            )

        # Network-related exceptions are generally transient.
        if isinstance(error, (TimeoutError, ConnectionError)):
            return True

        return False

    # -----------------------------------------------------
    # PROCESS ONE BATCH
    # -----------------------------------------------------

    async def _process_batch(
        self,
        func: Callable[[list], Awaitable[list]],
        batch: list,
        index: int,
        total: int,
        semaphore: asyncio.Semaphore,
        stop: asyncio.Event
    ) -> list:

        async with semaphore:

            if stop.is_set():
                raise asyncio.CancelledError()

            tokens = self._estimate_tokens(batch)

            for attempt in range(1, self.max_retries + 1):

                await self._wait_for_rate_limit(tokens, stop)

                try:

                    results = await func(batch)

                    if len(results) != len(batch):
                        raise ValueError(
                            f"Expected {len(batch)} results, "
                            f"received {len(results)}."
                        )

                    logger.info(
                        "Batch %s/%s completed (%s items).",
                        index,
                        total,
                        len(batch)
                    )

                    return results

                except asyncio.CancelledError:
                    raise

                except Exception as error:

                    if (
                        attempt >= self.max_retries
                        or not self._is_retryable(error)
                    ):

                        logger.error(
                            "Batch %s/%s failed on attempt %s: %s",
                            index,
                            total,
                            attempt,
                            error
                        )

                        stop.set()
                        raise

                    backoff = min(
                        MAX_RETRY_DELAY,
                        BASE_RETRY_DELAY * (2 ** (attempt - 1))
                    )

                    sleep_time = random.uniform(0, backoff)

                    logger.warning(
                        "Batch %s/%s failed on attempt %s. "
                        "Retrying in %.1f seconds: %s",
                        index,
                        total,
                        attempt,
                        sleep_time,
                        error
                    )

                    try:
                        await asyncio.wait_for(
                            stop.wait(),
                            timeout=sleep_time
                        )

                        raise asyncio.CancelledError()

                    except asyncio.TimeoutError:
                        continue

            raise RuntimeError(
                f"Batch {index}/{total} exhausted retries."
            )

    # -----------------------------------------------------
    # MAIN BATCH PROCESSOR
    # -----------------------------------------------------

    async def run(
        self,
        items: list,
        func: Callable[[list], Awaitable[list]]
    ) -> list:

        if not items:
            return []

        batches = self.create_batches(items)

        total = len(batches)

        semaphore = asyncio.Semaphore(self.max_concurrency)
        stop = asyncio.Event()

        started_at = time.monotonic()

        logger.info(
            "Batch processing started: %s items, %s batches, "
            "batch_size=%s, concurrency=%s",
            len(items),
            total,
            self.batch_size,
            self.max_concurrency
        )

        tasks = [
            asyncio.create_task(
                self._process_batch(
                    func,
                    batch,
                    index,
                    total,
                    semaphore,
                    stop
                )
            )
            for index, batch in enumerate(batches, start=1)
        ]

        try:

            batch_results = await asyncio.gather(*tasks)

        except BaseException:

            stop.set()

            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(
                *tasks,
                return_exceptions=True
            )

            raise

        results = [
            item
            for batch_result in batch_results
            for item in batch_result
        ]

        logger.info(
            "Batch processing completed: %s items in %.1f seconds.",
            len(results),
            time.monotonic() - started_at
        )

        return results
