"""Per-request span recording.

Every stage emits a span; the span log *is* the latency dataset. The benchmark
does not instrument the pipeline separately -- it runs real requests and reads
these spans back, so the published percentiles describe the code that actually
serves traffic rather than a parallel measurement path that might drift from it.

Writes are buffered and flushed from a background thread. A synchronous SQLite
insert on the hot path would add 0.5-2 ms to a 60 ms request just to record how
long the request took, which is a self-defeating way to measure latency.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from vrag.schemas import Span

SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    request_id   TEXT    NOT NULL,
    ts           REAL    NOT NULL,
    name         TEXT    NOT NULL,
    duration_ms  REAL    NOT NULL,
    ok           INTEGER NOT NULL,
    skipped      INTEGER NOT NULL,
    error        TEXT,
    attributes   TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_name ON spans(name);
CREATE INDEX IF NOT EXISTS idx_spans_request ON spans(request_id);

CREATE TABLE IF NOT EXISTS requests (
    request_id      TEXT PRIMARY KEY,
    ts              REAL NOT NULL,
    core_ms         REAL NOT NULL,
    total_ms        REAL NOT NULL,
    within_budget   INTEGER NOT NULL,
    abstained       INTEGER NOT NULL,
    refusal_reason  TEXT,
    lang            TEXT,
    strategy        TEXT,
    degradations    INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass
class RequestTrace:
    request_id: str
    spans: list[Span] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    def add(self, span: Span) -> None:
        self.spans.append(span)

    @contextmanager
    def span(self, name: str, **attributes: object):
        t0 = time.perf_counter()
        span = Span(name=name, duration_ms=0.0, attributes=dict(attributes))
        try:
            yield span
        except Exception as exc:
            span.ok = False
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = (time.perf_counter() - t0) * 1000.0
            self.spans.append(span)

    def skipped(self, name: str, reason: str) -> None:
        self.spans.append(Span(name=name, duration_ms=0.0, skipped=True,
                               attributes={"reason": reason}))

    @property
    def timings(self) -> dict[str, float]:
        return {s.name: round(s.duration_ms, 3) for s in self.spans}


class Tracer:
    """Buffered span writer with a background flusher."""

    def __init__(self, path: Path, flush_interval_s: float = 2.0, buffer_size: int = 4096):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: deque[tuple] = deque(maxlen=buffer_size * 4)
        self._requests: deque[tuple] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._interval = flush_interval_s

        with self._connect() as conn:
            conn.executescript(SCHEMA)

        self._thread = threading.Thread(target=self._loop, daemon=True, name="vrag-tracer")
        self._thread.start()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # -- write side ---------------------------------------------------------- #
    def record(self, trace: RequestTrace, envelope) -> None:  # noqa: ANN001
        ts = time.time()
        with self._lock:
            for span in trace.spans:
                self._buffer.append(
                    (
                        trace.request_id, ts, span.name, span.duration_ms,
                        int(span.ok), int(span.skipped), span.error,
                        json.dumps(span.attributes, default=str),
                    )
                )
            self._requests.append(
                (
                    trace.request_id, ts, envelope.core_latency_ms, envelope.total_latency_ms,
                    int(envelope.within_budget), int(envelope.abstained),
                    envelope.refusal_reason.value if envelope.refusal_reason else None,
                    envelope.detected_lang, envelope.strategy, len(envelope.degradations),
                )
            )

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.flush()

    def flush(self) -> None:
        with self._lock:
            spans = list(self._buffer)
            requests = list(self._requests)
            self._buffer.clear()
            self._requests.clear()

        if not spans and not requests:
            return

        try:
            with self._connect() as conn:
                if spans:
                    conn.executemany(
                        "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?)", spans
                    )
                if requests:
                    conn.executemany(
                        "INSERT OR REPLACE INTO requests VALUES (?,?,?,?,?,?,?,?,?,?)", requests
                    )
        except sqlite3.Error:
            # Losing telemetry must never fail a request. The rows are already
            # out of the buffer; dropping them is the correct trade.
            return

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        self.flush()

    # -- read side ----------------------------------------------------------- #
    def percentiles(self, percentiles: list[int]) -> dict[str, dict[str, float]]:
        """Per-stage percentiles straight from the span log."""
        self.flush()
        out: dict[str, dict[str, float]] = {}
        with self._connect() as conn:
            names = [r[0] for r in conn.execute(
                "SELECT DISTINCT name FROM spans WHERE skipped=0"
            )]
            for name in names:
                values = [
                    r[0] for r in conn.execute(
                        "SELECT duration_ms FROM spans WHERE name=? AND skipped=0 "
                        "ORDER BY duration_ms", (name,)
                    )
                ]
                if values:
                    out[name] = _percentiles(values, percentiles)

            core = [r[0] for r in conn.execute(
                "SELECT core_ms FROM requests ORDER BY core_ms"
            )]
            if core:
                out["__core__"] = _percentiles(core, percentiles)
            total = [r[0] for r in conn.execute(
                "SELECT total_ms FROM requests ORDER BY total_ms"
            )]
            if total:
                out["__total__"] = _percentiles(total, percentiles)
        return out


def _percentiles(sorted_values: list[float], percentiles: list[int]) -> dict[str, float]:
    n = len(sorted_values)
    out = {"n": float(n)}
    for p in percentiles:
        idx = min(n - 1, int(round((p / 100.0) * (n - 1))))
        out[f"p{p}"] = round(sorted_values[idx], 3)
    out["mean"] = round(sum(sorted_values) / n, 3)
    return out
