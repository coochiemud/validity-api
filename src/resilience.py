"""
Validity — API Resilience
resilience.py

Retry logic, rate limit handling, and timeout management
for all OpenAI API calls in the pipeline.

Provides:
- @retry_openai decorator for wrapping any OpenAI call
- ResilientOpenAIClient — drop-in wrapper with automatic retry
- Structured error classification

Design:
- Exponential backoff: 1s, 2s, 4s (3 attempts total)
- Retries on: rate limits, timeouts, server errors (5xx)
- Does not retry on: auth errors, invalid requests (4xx except 429)
- Logs every retry attempt with reason and wait time
- Returns None on exhausted retries (caller decides how to handle)
"""

from __future__ import annotations
import functools
import logging
import os
import time
from typing import Any, Callable, Optional, TypeVar

from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    APIStatusError,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger("validity.resilience")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ── Retry configuration ───────────────────────────────────────────────────────

DEFAULT_MAX_RETRIES  = 3
DEFAULT_BASE_WAIT    = 1.0   # seconds
DEFAULT_MAX_WAIT     = 30.0  # seconds — cap on exponential growth

# Errors that are retryable
RETRYABLE_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


# ── Retry decorator ───────────────────────────────────────────────────────────

F = TypeVar("F", bound=Callable[..., Any])


def retry_openai(
    max_retries: int   = DEFAULT_MAX_RETRIES,
    base_wait:   float = DEFAULT_BASE_WAIT,
    max_wait:    float = DEFAULT_MAX_WAIT,
):
    """
    Decorator that wraps an OpenAI API call with exponential backoff retry.

    Usage:
        @retry_openai(max_retries=3)
        def call_api(...):
            return client.chat.completions.create(...)

    Returns None if all retries are exhausted.
    Non-retryable errors (auth, bad request) are raised immediately.
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)

                except RETRYABLE_ERRORS as e:
                    last_error = e
                    wait_time  = min(base_wait * (2 ** (attempt - 1)), max_wait)

                    # Rate limit errors may include a retry-after header
                    if isinstance(e, RateLimitError):
                        retry_after = _extract_retry_after(e)
                        if retry_after:
                            wait_time = max(wait_time, retry_after)

                    if attempt < max_retries:
                        logger.warning(
                            f"{fn.__name__} attempt {attempt}/{max_retries} failed "
                            f"({type(e).__name__}: {str(e)[:80]}). "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"{fn.__name__} failed after {max_retries} attempts. "
                            f"Last error: {type(e).__name__}: {str(e)[:120]}"
                        )

                except APIStatusError as e:
                    # 4xx errors (except 429) are not retryable
                    if e.status_code == 429:
                        # 429 is rate limit — retryable
                        last_error = e
                        wait_time  = min(base_wait * (2 ** (attempt - 1)), max_wait)
                        if attempt < max_retries:
                            logger.warning(
                                f"{fn.__name__} rate limited (429). "
                                f"Retrying in {wait_time:.1f}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            logger.error(f"{fn.__name__} rate limited — retries exhausted.")
                    else:
                        # Non-retryable — raise immediately
                        logger.error(
                            f"{fn.__name__} non-retryable API error "
                            f"(status={e.status_code}): {str(e)[:120]}"
                        )
                        raise

                except Exception as e:
                    # Unexpected errors — raise immediately
                    logger.error(f"{fn.__name__} unexpected error: {type(e).__name__}: {str(e)[:120]}")
                    raise

            return None  # All retries exhausted

        return wrapper  # type: ignore
    return decorator


def _extract_retry_after(error: RateLimitError) -> Optional[float]:
    """
    Extract the retry-after value from a RateLimitError response header.
    Returns None if not present.
    """
    try:
        headers = getattr(error, "response", None)
        if headers is not None:
            retry_after = headers.headers.get("retry-after")
            if retry_after:
                return float(retry_after)
    except Exception:
        pass
    return None


# ── Resilient OpenAI client ───────────────────────────────────────────────────

class ResilientOpenAIClient:
    """
    Drop-in wrapper around OpenAI client that adds automatic retry
    to chat completion calls.

    Usage:
        client = ResilientOpenAIClient()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[...],
        )
        # Returns None if all retries exhausted, otherwise normal response
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        max_retries: int           = DEFAULT_MAX_RETRIES,
        base_wait:   float         = DEFAULT_BASE_WAIT,
        max_wait:    float         = DEFAULT_MAX_WAIT,
        timeout:     float         = 60.0,
    ):
        self._client = OpenAI(
            api_key = api_key or os.environ.get("OPENAI_API_KEY"),
            timeout = timeout,
        )
        self._max_retries = max_retries
        self._base_wait   = base_wait
        self._max_wait    = max_wait
        self.chat         = _ResilientChat(self._client, max_retries, base_wait, max_wait)


class _ResilientChat:
    def __init__(self, client, max_retries, base_wait, max_wait):
        self._client      = client
        self._max_retries = max_retries
        self._base_wait   = base_wait
        self._max_wait    = max_wait
        self.completions  = _ResilientCompletions(client, max_retries, base_wait, max_wait)


class _ResilientCompletions:
    def __init__(self, client, max_retries, base_wait, max_wait):
        self._client      = client
        self._max_retries = max_retries
        self._base_wait   = base_wait
        self._max_wait    = max_wait

    def create(self, **kwargs):
        """
        Call chat.completions.create with automatic retry.
        Returns None if all retries are exhausted.
        """
        last_error = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)

            except RETRYABLE_ERRORS as e:
                last_error = e
                wait_time  = min(self._base_wait * (2 ** (attempt - 1)), self._max_wait)

                if isinstance(e, RateLimitError):
                    retry_after = _extract_retry_after(e)
                    if retry_after:
                        wait_time = max(wait_time, retry_after)

                if attempt < self._max_retries:
                    logger.warning(
                        f"chat.completions.create attempt {attempt}/{self._max_retries} "
                        f"failed ({type(e).__name__}). Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"chat.completions.create failed after {self._max_retries} attempts. "
                        f"Last error: {type(e).__name__}: {str(e)[:120]}"
                    )

            except APIStatusError as e:
                if e.status_code == 429:
                    last_error = e
                    wait_time  = min(self._base_wait * (2 ** (attempt - 1)), self._max_wait)
                    if attempt < self._max_retries:
                        logger.warning(f"Rate limited (429). Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error("Rate limited — retries exhausted.")
                else:
                    logger.error(f"Non-retryable API error (status={e.status_code})")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error: {type(e).__name__}: {str(e)[:120]}")
                raise

        return None


# ── Pipeline-level error classification ──────────────────────────────────────

class PipelineError(Exception):
    """Base class for Validity pipeline errors."""
    pass

class ExtractionError(PipelineError):
    """Stage 1 extraction failed."""
    pass

class TranslationError(PipelineError):
    """Stage 2 translation failed."""
    pass

class EncodingError(PipelineError):
    """Z3 encoding failed."""
    pass

class SolverError(PipelineError):
    """Z3 solver failed."""
    pass


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("── Smoke test: ResilientOpenAIClient ─────────────────────────\n")

    client = ResilientOpenAIClient(timeout=30.0)

    response = client.chat.completions.create(
        model       = "gpt-4o",
        messages    = [{"role": "user", "content": "Say 'Validity resilience test passed' and nothing else."}],
        temperature = 0,
        max_tokens  = 20,
    )

    if response is None:
        print("FAILED — API call returned None after retries.")
        sys.exit(1)

    result = response.choices[0].message.content.strip()
    print(f"Response: {result}")
    print(f"\nResilientOpenAIClient smoke test complete.")

    print("\n── Smoke test: retry_openai decorator ───────────────────────\n")

    attempt_count = 0

    @retry_openai(max_retries=3, base_wait=0.1)
    def always_succeeds():
        global attempt_count
        attempt_count += 1
        return "success"

    result = always_succeeds()
    print(f"Result: {result}  Attempts: {attempt_count}")
    assert result == "success" and attempt_count == 1

    print("retry_openai decorator smoke test complete.")
