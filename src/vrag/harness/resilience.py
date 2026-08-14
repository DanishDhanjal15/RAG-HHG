"""Retries, timeouts, circuit breakers, and the error taxonomy.

Everything that leaves the process -- speech-to-text, the LLM polish path -- is
wrapped here. The contract these primitives enforce is simple and absolute: a
dependency failure produces a **typed refusal**, never an exception that escapes
to the caller as a 500. The API has exactly one response shape, and a broken
Sarvam key has to fit inside it.

The circuit breaker deserves the most attention. Retrying into a dependency that
is already down converts one slow request into three slow requests and turns a
partial outage into a total one. After ``failure_threshold`` consecutive failures
the breaker opens and every call fails *immediately* with ``SttUnavailable``, so
the user gets an honest error in 1 ms instead of a spinner for 24 s. After
``reset_after_s`` it half-opens and lets a single probe through.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Error taxonomy
# --------------------------------------------------------------------------- #
class VragError(Exception):
    """Base for every error the pipeline knows how to turn into a refusal."""

    retryable: bool = False


class TransientError(VragError):
    """Worth retrying: timeout, 429, 5xx, connection reset."""

    retryable = True


class PermanentError(VragError):
    """Not worth retrying: 401, 400, malformed audio."""

    retryable = False


class SttUnavailable(VragError):
    """Speech-to-text could not be reached, or the breaker is open."""


class StageTimeout(TransientError):
    def __init__(self, stage: str, timeout_s: float) -> None:
        super().__init__(f"stage {stage!r} exceeded {timeout_s:.2f}s")
        self.stage = stage


def classify_http(status: int) -> type[VragError]:
    if status in (408, 409, 425, 429) or status >= 500:
        return TransientError
    return PermanentError


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #
class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    reset_after_s: float = 30.0

    _failures: int = 0
    _opened_at: float = 0.0
    _state: BreakerState = BreakerState.CLOSED

    @property
    def state(self) -> BreakerState:
        if self._state is BreakerState.OPEN and (
            time.monotonic() - self._opened_at >= self.reset_after_s
        ):
            self._state = BreakerState.HALF_OPEN
        return self._state

    def allow(self) -> bool:
        return self.state is not BreakerState.OPEN

    def on_success(self) -> None:
        self._failures = 0
        self._state = BreakerState.CLOSED

    def on_failure(self) -> None:
        self._failures += 1
        # A failed probe in half-open re-opens immediately: one success is needed
        # to close, one failure is enough to re-open. Recovery should be earned.
        if self._state is BreakerState.HALF_OPEN or self._failures >= self.failure_threshold:
            self._state = BreakerState.OPEN
            self._opened_at = time.monotonic()


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #
@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.25
    max_delay_s: float = 4.0
    jitter: float = 0.3

    def delay_for(self, attempt: int) -> float:
        """Exponential backoff with proportional jitter.

        Jitter is not decoration: without it, N clients that failed together
        retry together, and the retry storm is what keeps the dependency down.
        """
        raw = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        return raw * (1.0 + random.uniform(-self.jitter, self.jitter))


@dataclass
class CallResult:
    attempts: int = 0
    total_wait_s: float = 0.0
    breaker_state: str = BreakerState.CLOSED
    errors: list[str] = field(default_factory=list)


async def call_with_resilience(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    breaker: CircuitBreaker | None = None,
    timeout_s: float | None = None,
    stage: str = "external",
) -> tuple[T, CallResult]:
    """Run ``fn`` under timeout + retry + circuit breaker.

    Returns the value and a ``CallResult`` describing what it took to get it --
    attempts and waiting are recorded as span attributes, so a request that
    succeeded only after two retries is visibly different from one that succeeded
    first time.
    """
    result = CallResult()

    if breaker is not None and not breaker.allow():
        result.breaker_state = breaker.state
        raise SttUnavailable(
            f"{stage}: circuit breaker open after {breaker.failure_threshold} "
            f"consecutive failures; retrying in "
            f"{max(0.0, breaker.reset_after_s - (time.monotonic() - breaker._opened_at)):.0f}s"
        )

    last: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        result.attempts = attempt
        try:
            value = (
                await asyncio.wait_for(fn(), timeout=timeout_s)
                if timeout_s
                else await fn()
            )
            if breaker is not None:
                breaker.on_success()
                result.breaker_state = breaker.state
            return value, result

        except TimeoutError as exc:
            last = StageTimeout(stage, timeout_s or 0.0)
            result.errors.append(str(last))
            _ = exc
        except VragError as exc:
            last = exc
            result.errors.append(f"{type(exc).__name__}: {exc}")
            if not exc.retryable:
                break
        except Exception as exc:  # noqa: BLE001 -- unknown failures are treated as transient
            last = TransientError(f"{type(exc).__name__}: {exc}")
            result.errors.append(str(last))

        if attempt < policy.max_attempts:
            delay = policy.delay_for(attempt)
            result.total_wait_s += delay
            await asyncio.sleep(delay)

    if breaker is not None:
        breaker.on_failure()
        result.breaker_state = breaker.state

    raise last or TransientError(f"{stage}: exhausted {policy.max_attempts} attempts")


def guard_sync(fn: Callable[[], T], *, stage: str, default: T) -> T:
    """Run an in-process stage, converting any failure into ``default``.

    Used for optional local stages (reranking, the grounding check) where a model
    that fails to load should degrade the answer rather than fail the request.
    The caller records a degradation so the loss is never invisible.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default
