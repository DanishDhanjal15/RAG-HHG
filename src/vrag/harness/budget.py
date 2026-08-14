"""The latency budget manager -- what turns "<200 ms" from a hope into an invariant.

A pipeline that simply runs every stage and reports the total has an *average*
latency. It has no ceiling: one slow reranker call, one cold page, one noisy
neighbour on a shared container, and the request blows through the SLA with
nothing to show for it.

This module inverts that. Each stage is declared ``required`` or not. Before an
optional stage runs, the manager compares the remaining budget against that
stage's **measured rolling p90** and skips it if it will not fit. The request then
completes with slightly worse retrieval instead of a blown deadline, and the skip
is reported to the caller as a ``Degradation`` -- so a fast answer is never
*silently* a worse answer.

Two details that matter:

* **p90, not the mean.** A stage whose mean is 40 ms and whose p90 is 110 ms will
  overrun roughly one request in ten if you budget on the mean. The ceiling has to
  be planned against the tail.
* **Statistics are process-global and warm up from config.** With no history a
  stage is estimated from its configured ``soft_ms``; after a few hundred requests
  the estimate is entirely measured. This is why the benchmark runs a warmup pass
  before recording -- the first requests budget from guesses.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from vrag.config import BudgetCfg
from vrag.schemas import Degradation


class StageStats:
    """Bounded ring buffer of recent durations for one stage."""

    __slots__ = ("_samples", "_lock")

    def __init__(self, window: int = 512) -> None:
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = Lock()

    def add(self, ms: float) -> None:
        with self._lock:
            self._samples.append(ms)

    def percentile(self, p: float) -> float | None:
        with self._lock:
            if len(self._samples) < 8:  # too few samples to trust a tail estimate
                return None
            ordered = sorted(self._samples)
        idx = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
        return ordered[idx]

    @property
    def count(self) -> int:
        return len(self._samples)


class BudgetRegistry:
    """Process-wide stage timing statistics.

    Shared across requests on purpose: the point is to learn how long stages
    actually take *on this machine under this load*, which a single request
    cannot know.
    """

    def __init__(self) -> None:
        self._stats: dict[str, StageStats] = {}
        self._lock = Lock()

    def stats(self, stage: str) -> StageStats:
        with self._lock:
            stats = self._stats.get(stage)
            if stats is None:
                stats = StageStats()
                self._stats[stage] = stats
            return stats

    def record(self, stage: str, ms: float) -> None:
        self.stats(stage).add(ms)

    def estimate(self, stage: str, fallback_ms: float, p: float = 90.0) -> float:
        measured = self.stats(stage).percentile(p)
        return fallback_ms if measured is None else measured

    def snapshot(self) -> dict[str, dict[str, float | int | None]]:
        """Per-stage percentiles, or ``None`` where there is not enough data yet.

        Deliberately not ``0.0`` for the cold case. A reported p50 of 0.00 ms is
        indistinguishable from a genuinely instant stage, so a dashboard would
        show a confident wrong number instead of an honest gap. ``None``
        serialises to JSON ``null`` and renders as "--".
        """
        with self._lock:
            names = list(self._stats)
        return {
            name: {
                "n": self._stats[name].count,
                "p50": self._stats[name].percentile(50),
                "p90": self._stats[name].percentile(90),
                "p100": self._stats[name].percentile(100),
            }
            for name in names
        }

    @property
    def warm(self) -> bool:
        """True once every recorded stage has enough samples to estimate a tail.

        Until this is true the manager budgets from configured guesses rather
        than measurement, so early requests can overrun. The server warms at boot
        specifically to get past this before the first user request.
        """
        with self._lock:
            stats = list(self._stats.values())
        return bool(stats) and all(s.percentile(90) is not None for s in stats)


GLOBAL_REGISTRY = BudgetRegistry()


@dataclass
class LatencyBudget:
    """Per-request budget clock."""

    cfg: BudgetCfg
    registry: BudgetRegistry = field(default=GLOBAL_REGISTRY)
    started_at: float = field(default_factory=time.perf_counter)
    degradations: list[Degradation] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.started_at = time.perf_counter()
        self.degradations.clear()
        self.timings.clear()

    # -- clock --------------------------------------------------------------- #
    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0

    @property
    def remaining_ms(self) -> float:
        return self.cfg.core_budget_ms - self.cfg.safety_margin_ms - self.elapsed_ms

    # -- policy -------------------------------------------------------------- #
    def should_run(self, stage: str) -> bool:
        """Decide whether an optional stage fits in what is left of the budget."""
        spec = self.cfg.stages.get(stage)
        if spec is None or spec.required:
            return True

        expected = self.registry.estimate(stage, spec.soft_ms)
        remaining = self.remaining_ms
        if expected <= remaining:
            return True

        self.degradations.append(
            Degradation(
                stage=stage,
                reason=f"expected p90 {expected:.1f}ms exceeds {remaining:.1f}ms remaining",
                remaining_budget_ms=round(remaining, 2),
                expected_cost_ms=round(expected, 2),
            )
        )
        self.timings[stage] = 0.0
        return False

    def record(self, stage: str, ms: float) -> None:
        self.timings[stage] = round(ms, 3)
        self.registry.record(stage, ms)

    def note_degradation(self, stage: str, reason: str) -> None:
        """Record a degradation the budget did not cause (a dependency was
        unavailable, a model failed to load). Same reporting path so the caller
        sees one uniform list."""
        self.degradations.append(
            Degradation(
                stage=stage,
                reason=reason,
                remaining_budget_ms=round(self.remaining_ms, 2),
                expected_cost_ms=0.0,
            )
        )

    @property
    def within_budget(self) -> bool:
        return self.elapsed_ms <= self.cfg.core_budget_ms


@dataclass
class stage_timer:  # noqa: N801 -- used as a context manager, reads as a verb
    """``with stage_timer(budget, "dense_search"):`` -- times and records a stage."""

    budget: LatencyBudget
    name: str
    _t0: float = 0.0

    def __enter__(self) -> stage_timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.budget.record(self.name, (time.perf_counter() - self._t0) * 1000.0)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0
