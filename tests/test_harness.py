"""Harness tests: the budget manager and the resilience primitives.

These are the pieces that turn "<200 ms" from a hope into an invariant, and that
turn a dependency outage into a typed refusal instead of a 500. Both are pure
logic with no index dependency, so they are tested directly.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from vrag.config import BudgetCfg, StageBudget
from vrag.harness.budget import BudgetRegistry, LatencyBudget, StageStats
from vrag.harness.resilience import (
    BreakerState,
    CircuitBreaker,
    PermanentError,
    RetryPolicy,
    StageTimeout,
    SttUnavailable,
    TransientError,
    call_with_resilience,
    classify_http,
    guard_sync,
)


def budget_cfg(core_ms: float = 200.0, margin: float = 15.0) -> BudgetCfg:
    return BudgetCfg(
        core_budget_ms=core_ms,
        safety_margin_ms=margin,
        stages={
            "embed_query": StageBudget(required=True, soft_ms=20),
            "rerank": StageBudget(required=False, soft_ms=90),
            "grounding_guard": StageBudget(required=False, soft_ms=20),
        },
    )


# --------------------------------------------------------------------------- #
class TestStageStats:
    def test_too_few_samples_returns_none(self):
        """An estimate from 3 samples is noise; the manager must fall back to config."""
        stats = StageStats()
        for _ in range(3):
            stats.add(10.0)
        assert stats.percentile(90) is None

    def test_percentile_after_enough_samples(self):
        stats = StageStats()
        for value in range(1, 101):
            stats.add(float(value))
        p90 = stats.percentile(90)
        assert p90 is not None
        assert 88 <= p90 <= 92

    def test_window_is_bounded(self):
        stats = StageStats(window=50)
        for value in range(200):
            stats.add(float(value))
        assert stats.count == 50
        # Only recent samples survive, so the estimate tracks current conditions
        # rather than the whole history of the process.
        assert stats.percentile(50) > 150


class TestBudgetRegistry:
    def test_estimate_falls_back_to_config_when_cold(self):
        registry = BudgetRegistry()
        assert registry.estimate("rerank", fallback_ms=90.0) == 90.0

    def test_estimate_uses_measurement_once_warm(self):
        registry = BudgetRegistry()
        for _ in range(50):
            registry.record("rerank", 30.0)
        assert registry.estimate("rerank", fallback_ms=90.0) == pytest.approx(30.0)

    def test_snapshot_reports_percentiles(self):
        registry = BudgetRegistry()
        for value in range(1, 101):
            registry.record("dense_search", float(value))
        snap = registry.snapshot()["dense_search"]
        assert snap["n"] == 100
        assert snap["p50"] < snap["p90"] <= snap["p100"]


class TestLatencyBudget:
    def test_required_stages_always_run(self):
        registry = BudgetRegistry()
        budget = LatencyBudget(budget_cfg(), registry)
        budget.started_at = time.perf_counter() - 10.0  # wildly over budget
        assert budget.should_run("embed_query") is True

    def test_unknown_stages_are_treated_as_required(self):
        budget = LatencyBudget(budget_cfg(), BudgetRegistry())
        assert budget.should_run("some_new_stage") is True

    def test_optional_stage_runs_when_it_fits(self):
        registry = BudgetRegistry()
        for _ in range(20):
            registry.record("rerank", 30.0)
        budget = LatencyBudget(budget_cfg(), registry)
        assert budget.should_run("rerank") is True
        assert budget.degradations == []

    def test_optional_stage_skipped_when_it_does_not_fit(self):
        registry = BudgetRegistry()
        for _ in range(20):
            registry.record("rerank", 150.0)   # measured p90 exceeds the budget
        budget = LatencyBudget(budget_cfg(core_ms=100.0), registry)
        assert budget.should_run("rerank") is False
        assert len(budget.degradations) == 1
        assert budget.degradations[0].stage == "rerank"

    def test_skip_is_reported_with_numbers(self):
        """A degradation the caller cannot interpret is not much better than a
        silent one."""
        registry = BudgetRegistry()
        for _ in range(20):
            registry.record("rerank", 150.0)
        budget = LatencyBudget(budget_cfg(core_ms=100.0), registry)
        budget.should_run("rerank")
        degradation = budget.degradations[0]
        assert degradation.expected_cost_ms == pytest.approx(150.0, abs=1.0)
        assert degradation.remaining_budget_ms < 100.0
        assert "150" in degradation.reason

    def test_elapsed_time_shrinks_the_budget(self):
        registry = BudgetRegistry()
        for _ in range(20):
            registry.record("rerank", 60.0)
        budget = LatencyBudget(budget_cfg(core_ms=200.0), registry)
        assert budget.should_run("rerank") is True

        budget.started_at = time.perf_counter() - 0.150  # 150ms already gone
        assert budget.should_run("rerank") is False

    def test_uses_measured_p90_over_configured_estimate(self):
        """The config value is a cold-start guess; measurement must win once it
        exists, or the manager keeps budgeting against a number that was never
        true on this machine."""
        registry = BudgetRegistry()
        for _ in range(20):
            registry.record("rerank", 10.0)   # config says 90, reality is 10
        budget = LatencyBudget(budget_cfg(core_ms=60.0), registry)
        assert budget.should_run("rerank") is True

    def test_within_budget_flag(self):
        budget = LatencyBudget(budget_cfg(core_ms=200.0), BudgetRegistry())
        assert budget.within_budget is True
        budget.started_at = time.perf_counter() - 0.5
        assert budget.within_budget is False

    def test_manual_degradation_uses_the_same_channel(self):
        budget = LatencyBudget(budget_cfg(), BudgetRegistry())
        budget.note_degradation("rerank", "model failed to load")
        assert len(budget.degradations) == 1
        assert "load" in budget.degradations[0].reason


# --------------------------------------------------------------------------- #
class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        breaker = CircuitBreaker(name="t", failure_threshold=3)
        for _ in range(2):
            breaker.on_failure()
        assert breaker.allow() is True
        breaker.on_failure()
        assert breaker.allow() is False
        assert breaker.state is BreakerState.OPEN

    def test_success_resets_the_counter(self):
        breaker = CircuitBreaker(name="t", failure_threshold=3)
        breaker.on_failure()
        breaker.on_failure()
        breaker.on_success()
        breaker.on_failure()
        assert breaker.allow() is True

    def test_half_opens_after_reset_window(self):
        # The reset window is advanced by rewinding _opened_at rather than by
        # sleeping. A sleep-based version of this test is flaky under load: the
        # breaker can cross its own reset window while the test is still setting
        # up, and then the "still OPEN" assertion fails for reasons that have
        # nothing to do with the breaker.
        breaker = CircuitBreaker(name="t", failure_threshold=1, reset_after_s=30.0)
        breaker.on_failure()
        assert breaker.state is BreakerState.OPEN
        assert breaker.allow() is False

        breaker._opened_at -= 31.0        # window has now elapsed
        assert breaker.state is BreakerState.HALF_OPEN
        assert breaker.allow() is True

    def test_failed_probe_reopens_immediately(self):
        """Recovery must be earned: one success closes, one failure re-opens."""
        breaker = CircuitBreaker(name="t", failure_threshold=5, reset_after_s=30.0)
        for _ in range(5):
            breaker.on_failure()
        breaker._opened_at -= 31.0
        assert breaker.state is BreakerState.HALF_OPEN

        breaker.on_failure()
        assert breaker.state is BreakerState.OPEN
        assert breaker.allow() is False

    def test_successful_probe_closes_the_breaker(self):
        breaker = CircuitBreaker(name="t", failure_threshold=1, reset_after_s=30.0)
        breaker.on_failure()
        breaker._opened_at -= 31.0
        assert breaker.state is BreakerState.HALF_OPEN

        breaker.on_success()
        assert breaker.state is BreakerState.CLOSED
        assert breaker.allow() is True


class TestRetryPolicy:
    def test_delay_grows_exponentially(self):
        policy = RetryPolicy(base_delay_s=0.1, jitter=0.0)
        assert policy.delay_for(1) == pytest.approx(0.1)
        assert policy.delay_for(2) == pytest.approx(0.2)
        assert policy.delay_for(3) == pytest.approx(0.4)

    def test_delay_is_capped(self):
        policy = RetryPolicy(base_delay_s=1.0, max_delay_s=2.0, jitter=0.0)
        assert policy.delay_for(10) == pytest.approx(2.0)

    def test_jitter_spreads_retries(self):
        """Without jitter, clients that failed together retry together, and the
        retry storm is what keeps the dependency down."""
        policy = RetryPolicy(base_delay_s=1.0, jitter=0.3)
        delays = {policy.delay_for(1) for _ in range(50)}
        assert len(delays) > 1
        assert all(0.7 <= d <= 1.3 for d in delays)


class TestCallWithResilience:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        async def ok() -> str:
            return "value"

        value, result = await call_with_resilience(ok, policy=RetryPolicy())
        assert value == "value"
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_retries_transient_then_succeeds(self):
        calls = {"n": 0}

        async def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise TransientError("boom")
            return "value"

        value, result = await call_with_resilience(
            flaky, policy=RetryPolicy(max_attempts=5, base_delay_s=0.01)
        )
        assert value == "value"
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_permanent_error_is_not_retried(self):
        calls = {"n": 0}

        async def bad() -> str:
            calls["n"] += 1
            raise PermanentError("401")

        with pytest.raises(PermanentError):
            await call_with_resilience(bad, policy=RetryPolicy(max_attempts=5,
                                                               base_delay_s=0.01))
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_open_breaker_fails_fast(self):
        breaker = CircuitBreaker(name="t", failure_threshold=1)
        breaker.on_failure()
        calls = {"n": 0}

        async def never_called() -> str:
            calls["n"] += 1
            return "value"

        started = time.perf_counter()
        with pytest.raises(SttUnavailable):
            await call_with_resilience(never_called, policy=RetryPolicy(), breaker=breaker)
        # The point of the breaker: an honest error in ~0ms beats a spinner for 24s.
        assert time.perf_counter() - started < 0.05
        assert calls["n"] == 0

    @pytest.mark.asyncio
    async def test_timeout_is_enforced(self):
        async def slow() -> str:
            await asyncio.sleep(5)
            return "value"

        with pytest.raises(StageTimeout):
            await call_with_resilience(
                slow, policy=RetryPolicy(max_attempts=1), timeout_s=0.05
            )

    @pytest.mark.asyncio
    async def test_breaker_trips_after_exhausting_retries(self):
        breaker = CircuitBreaker(name="t", failure_threshold=1)

        async def always_fails() -> str:
            raise TransientError("down")

        with pytest.raises(TransientError):
            await call_with_resilience(
                always_fails,
                policy=RetryPolicy(max_attempts=2, base_delay_s=0.01),
                breaker=breaker,
            )
        assert breaker.allow() is False


class TestErrorClassification:
    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
    def test_retryable_statuses(self, status):
        assert classify_http(status) is TransientError

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_permanent_statuses(self, status):
        assert classify_http(status) is PermanentError


class TestGuardSync:
    def test_returns_default_on_failure(self):
        def boom() -> str:
            raise RuntimeError("model failed to load")

        assert guard_sync(boom, stage="rerank", default="fallback") == "fallback"

    def test_passes_value_through_on_success(self):
        assert guard_sync(lambda: "ok", stage="rerank", default="fallback") == "ok"
