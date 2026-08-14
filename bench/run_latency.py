"""Latency analytics -- the requirement-4 deliverable.

Reports P50 / P70 / P90 / P95 / P99 / P100 per stage and per composite, over a
few hundred real requests.

Three deliberate choices:

* **It measures the real pipeline, not a parallel copy.** The benchmark calls
  ``Pipeline.answer_text()`` and reads the spans the pipeline already emits. A
  separate measurement path drifts from the code that actually serves traffic,
  and then the published numbers describe something nobody runs.
* **Warmup requests are discarded.** The first requests pay ONNX graph
  optimization, arena allocation, page-faulting the mmapped store, and -- crucially
  -- they run while the budget manager still has no timing history and is
  estimating stage costs from config rather than measurement. Including them would
  describe the first minute of a process's life, not steady state.
* **P70 and P100 are both reported, and P100 is labelled honestly.** The form asks
  for P50/P70/P100. P100 over N runs is a single sample -- the slowest one -- and is
  dominated by whatever the OS was doing at that instant. We report it as asked,
  alongside P95/P99, which are what you would actually engineer against.

STT is measured separately and clearly labelled. It is a network round trip to
Sarvam, it cannot fit in a 200 ms budget, and averaging it into the core number
would be dishonest in the flattering direction.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from vrag.config import get_config
from vrag.harness.pipeline import Pipeline
from vrag.ingest.normalize import load_queries

console = Console()


# --------------------------------------------------------------------------- #
@dataclass
class Series:
    name: str
    samples: list[float] = field(default_factory=list)
    skipped: int = 0

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    def percentiles(self, ps: list[int]) -> dict[str, float]:
        if not self.samples:
            return {f"p{p}": 0.0 for p in ps} | {"n": 0, "mean": 0.0}
        ordered = sorted(self.samples)
        n = len(ordered)
        out: dict[str, float] = {"n": n}
        for p in ps:
            idx = min(n - 1, int(round((p / 100.0) * (n - 1))))
            out[f"p{p}"] = round(ordered[idx], 2)
        out["mean"] = round(statistics.fmean(ordered), 2)
        return out


class Collector:
    def __init__(self) -> None:
        self.series: dict[str, Series] = {}

    def record(self, name: str, ms: float) -> None:
        self.series.setdefault(name, Series(name)).add(ms)

    def skip(self, name: str) -> None:
        self.series.setdefault(name, Series(name)).skipped += 1


# --------------------------------------------------------------------------- #
# Pipeline order. Must include every stage the pipeline emits -- a stage missing
# here is silently absent from the report, which is how `generate_semantic`
# (the single most expensive optional stage) went unreported.
STAGE_ORDER = [
    "input_guard", "embed_query", "dense_search", "sparse_search", "fuse",
    "rerank", "domain_guard", "generate", "generate_semantic", "grounding_guard",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--out", default="docs/LATENCY.md")
    parser.add_argument("--chart", action="store_true", help="also write a PNG chart")
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Directory of .wav clips. Measures the real STT round trip separately "
             "from the core (needs VRAG_SARVAM_API_KEY).",
    )
    args = parser.parse_args()

    cfg = get_config()
    n = args.n or cfg.bench.latency.n_queries
    warmup = args.warmup if args.warmup is not None else cfg.bench.latency.warmup
    ps = cfg.bench.latency.percentiles

    console.print("[bold]Booting pipeline[/] (this pays every one-time cost up front)")
    t0 = time.perf_counter()
    pipeline = Pipeline(cfg)
    boot_s = time.perf_counter() - t0
    console.print(f"  ready in {boot_s:.1f}s   chunks={len(pipeline.store):,}")

    queries = [q for q in load_queries(cfg) if q.query.strip()]
    random.Random(cfg.corpus.seed + 1).shuffle(queries)
    pool = queries[: n + warmup]
    if len(pool) < n + warmup:
        console.print(f"[yellow]only {len(pool)} queries available[/]")

    # -- warmup ------------------------------------------------------------- #
    console.print(f"[bold]Warmup[/] ({warmup} requests, discarded)")
    for record in pool[:warmup]:
        pipeline.answer_text(record.query, lang=record.lang)

    # -- measured ----------------------------------------------------------- #
    console.print(f"[bold]Measuring[/] ({len(pool[warmup:])} requests)")
    collector = Collector()
    abstained = 0
    degraded = 0
    over_budget = 0

    for i, record in enumerate(pool[warmup:], start=1):
        envelope = pipeline.answer_text(record.query, lang=record.lang)

        for span in envelope.spans:
            if span.skipped:
                collector.skip(span.name)
            else:
                collector.record(span.name, span.duration_ms)

        collector.record("__core__", envelope.core_latency_ms)
        collector.record("__total__", envelope.total_latency_ms)

        abstained += int(envelope.abstained)
        degraded += int(bool(envelope.degradations))
        over_budget += int(not envelope.within_budget)

        if i % 100 == 0:
            console.print(f"  {i}/{len(pool[warmup:])}")

    measured = len(pool[warmup:])

    # -- voice path (optional, real API calls) ------------------------------- #
    voice_stats: dict[str, float] | None = None
    if args.audio_dir:
        voice_stats = _measure_voice(pipeline, cfg, Path(args.audio_dir), ps)

    # -- render ------------------------------------------------------------- #
    budget = cfg.budget.core_budget_ms
    table = Table(title=f"Latency over {measured} requests (text input, RAG core)")
    table.add_column("stage")
    for p in ps:
        table.add_column(f"P{p}", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("skipped", justify="right")

    def row(key: str, label: str, style: str = "") -> None:
        series = collector.series.get(key)
        if series is None:
            return
        stats = series.percentiles(ps)
        table.add_row(
            f"[{style}]{label}[/{style}]" if style else label,
            *[f"{stats[f'p{p}']:.2f}" for p in ps],
            f"{stats['mean']:.2f}",
            str(series.skipped) if series.skipped else "-",
        )

    for name in STAGE_ORDER:
        row(name, f"  {name}")
    table.add_section()
    row("__core__", f"RAG CORE (budget {budget:.0f}ms)", "bold green")
    row("__total__", "end-to-end (text in)", "bold")

    console.print(table)

    core = collector.series["__core__"].percentiles(ps)
    console.print(
        f"\n[bold]Budget:[/] {core['p100']:.1f}ms P100 vs {budget:.0f}ms target  "
        + ("[bold green]PASS[/]" if core["p100"] <= budget else "[bold red]FAIL[/]")
    )
    console.print(
        f"abstained {abstained}/{measured} ({abstained / measured:.1%})   "
        f"degraded {degraded}/{measured} ({degraded / measured:.1%})   "
        f"over budget {over_budget}/{measured}"
    )

    if voice_stats:
        vtable = Table(title=f"Voice path ({int(voice_stats['n'])} clips, real Sarvam calls)")
        vtable.add_column("stage")
        for p in ps:
            vtable.add_column(f"P{p}", justify="right")
        vtable.add_row("stt (network)", *[f"{voice_stats[f'stt_p{p}']:.0f}" for p in ps])
        vtable.add_row("rag core", *[f"{voice_stats[f'core_p{p}']:.1f}" for p in ps])
        vtable.add_row("[bold]end-to-end[/]", *[f"{voice_stats[f'total_p{p}']:.0f}" for p in ps])
        console.print(vtable)

    _write_report(cfg, args.out, collector, ps, measured, budget, boot_s,
                  abstained, degraded, over_budget, voice_stats)
    if args.chart:
        _write_chart(cfg, collector, ps)

    pipeline.close()


# --------------------------------------------------------------------------- #
def _measure_voice(pipeline, cfg, audio_dir: Path, ps: list[int]) -> dict[str, float] | None:  # noqa: ANN001
    """Measure the real voice path: Sarvam round trip + the core, separately.

    Deliberately NOT averaged into the core number. STT is a network call to a
    third party; folding it in would make the reported figure depend on the
    tester's internet connection and would flatter or damn the system for
    something it does not control.
    """
    import asyncio

    from vrag.config import get_secrets
    from vrag.schemas import AudioInput

    clips = sorted(p for p in audio_dir.glob("*.wav"))
    if not clips:
        console.print(f"[yellow]no .wav files in {audio_dir}; skipping voice path[/]")
        return None
    if not get_secrets().sarvam_api_key:
        console.print("[yellow]VRAG_SARVAM_API_KEY not set; skipping voice path[/]")
        return None

    console.print(f"[bold]Voice path[/] ({len(clips)} clips, real API calls)")
    stt_ms: list[float] = []
    core_ms: list[float] = []
    total_ms: list[float] = []
    failures = 0

    async def run_one(path: Path):  # noqa: ANN202
        return await pipeline.answer_audio(
            AudioInput(audio=path.read_bytes(), filename=path.name, mime_type="audio/wav")
        )

    for clip in clips:
        envelope = asyncio.run(run_one(clip))
        if envelope.refusal_reason and envelope.refusal_reason.value == "STT_UNAVAILABLE":
            failures += 1
            continue
        stt_ms.append(envelope.timings_ms.get("stt", 0.0))
        core_ms.append(envelope.core_latency_ms)
        total_ms.append(envelope.total_latency_ms)

    if not total_ms:
        console.print(f"[red]all {failures} clips failed at STT[/]")
        return None
    if failures:
        console.print(f"[yellow]{failures} clip(s) failed at STT and were excluded[/]")

    def pct(values: list[float], p: int) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))]

    out: dict[str, float] = {"n": float(len(total_ms)), "failures": float(failures)}
    for p in ps:
        out[f"stt_p{p}"] = pct(stt_ms, p)
        out[f"core_p{p}"] = pct(core_ms, p)
        out[f"total_p{p}"] = pct(total_ms, p)
    return out


def _write_report(cfg, out_rel, collector, ps, measured, budget, boot_s,  # noqa: ANN001
                  abstained, degraded, over_budget, voice_stats=None) -> None:
    core = collector.series["__core__"].percentiles(ps)
    total = collector.series["__total__"].percentiles(ps)

    lines = [
        "# Latency",
        "",
        "Generated by `bench/run_latency.py`. Numbers come from the spans the",
        "pipeline emits while serving real requests -- not from a separate",
        "measurement path -- so they describe the code that actually runs.",
        "",
        f"- **{measured} measured requests**, preceded by a discarded warmup pass",
        f"- corpus: {cfg.corpus.queries_per_language * len(cfg.corpus.languages):,} queries",
        f"- cold start (process boot, model load, index mmap): **{boot_s:.1f}s**, paid once",
        "",
        "## The <200 ms contract",
        "",
        "The RAG core is **embed -> retrieve -> fuse -> rerank -> guard -> answer**,",
        f"budgeted at {budget:.0f} ms and enforced by `harness/budget.py`, which skips",
        "optional stages when their measured p90 will not fit in the remaining time.",
        "",
        "| | P50 | P70 | P100 | verdict |",
        "|---|---|---|---|---|",
        f"| **RAG core** | {core['p50']:.1f} ms | {core['p70']:.1f} ms | {core['p100']:.1f} ms | "
        + ("**within budget**" if core["p100"] <= budget else "**OVER BUDGET**")
        + " |",
        "",
        "## Per-stage distribution",
        "",
        "| stage | " + " | ".join(f"P{p}" for p in ps) + " | mean | skipped |",
        "|---|" + "---|" * (len(ps) + 2),
    ]

    for name in STAGE_ORDER:
        series = collector.series.get(name)
        if series is None:
            continue
        stats = series.percentiles(ps)
        lines.append(
            f"| `{name}` | "
            + " | ".join(f"{stats[f'p{p}']:.2f}" for p in ps)
            + f" | {stats['mean']:.2f} | {series.skipped or '-'} |"
        )

    lines += [
        "| | | | | | | | | |",
        "| **RAG core** | "
        + " | ".join(f"**{core[f'p{p}']:.2f}**" for p in ps)
        + f" | **{core['mean']:.2f}** | - |",
        "| end-to-end (text in) | "
        + " | ".join(f"{total[f'p{p}']:.2f}" for p in ps)
        + f" | {total['mean']:.2f} | - |",
        "",
        "## Behaviour under load",
        "",
        f"- abstained: **{abstained}/{measured}** ({abstained / measured:.1%}) -- guardrails declining to answer",
        f"- degraded: **{degraded}/{measured}** ({degraded / measured:.1%}) -- budget manager dropped an optional stage",
        f"- over budget: **{over_budget}/{measured}**",
        "",
        "## On P100",
        "",
        "P100 over a few hundred runs is one sample -- the slowest -- and is dominated",
        "by whatever the OS was doing at that instant. It is reported because the",
        "submission form asks for it; P95 and P99 above are the numbers to engineer",
        "against.",
        "",
        "## What is NOT in the core number",
        "",
        "Speech-to-text is a network round trip to Sarvam and cannot fit a 200 ms",
        "budget. It is measured and reported separately rather than averaged in, and",
        "the LLM polish path runs after the grounded answer has already been returned.",
    ]

    if voice_stats:
        lines += [
            "",
            "## Voice path (real Sarvam calls)",
            "",
            f"Measured over **{int(voice_stats['n'])} recorded clips**. Reported separately",
            "because it depends on a third-party network round trip, not on this system.",
            "",
            "| stage | " + " | ".join(f"P{p}" for p in ps) + " |",
            "|---|" + "---|" * len(ps),
            "| speech-to-text (network) | "
            + " | ".join(f"{voice_stats[f'stt_p{p}']:.0f} ms" for p in ps) + " |",
            "| RAG core (local) | "
            + " | ".join(f"{voice_stats[f'core_p{p}']:.1f} ms" for p in ps) + " |",
            "| **end-to-end** | "
            + " | ".join(f"**{voice_stats[f'total_p{p}']:.0f} ms**" for p in ps) + " |",
            "",
            "The gap between the two rows is the point: the retrieval core is a small",
            "and predictable fraction of what the user waits for, and it is the only",
            "part a 200 ms budget can meaningfully govern.",
        ]
    else:
        lines += [
            "",
            "## Voice path",
            "",
            "Not measured in this run. Re-run with `--audio-dir <dir of .wav clips>` and",
            "`VRAG_SARVAM_API_KEY` set to add measured speech-to-text percentiles.",
        ]

    out = cfg.paths.data_dir.parent / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw = Path(cfg.paths.trace_dir) / "latency.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        json.dumps(
            {name: s.percentiles(ps) | {"skipped": s.skipped}
             for name, s in collector.series.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[dim]{out}\n{raw}[/]")


def _write_chart(cfg, collector, ps) -> None:  # noqa: ANN001
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed; skipping chart[/]")
        return

    names = [n for n in STAGE_ORDER if n in collector.series]
    p50 = [collector.series[n].percentiles(ps)["p50"] for n in names]
    p95 = [collector.series[n].percentiles(ps).get("p95", 0.0) for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = range(len(names))
    ax.barh(list(y), p95, color="#c8d8e8", label="P95")
    ax.barh(list(y), p50, color="#2b6cb0", label="P50")
    ax.set_yticks(list(y), names)
    ax.invert_yaxis()
    ax.set_xlabel("milliseconds")
    ax.set_title(f"Per-stage latency (budget {cfg.budget.core_budget_ms:.0f} ms for the core)")
    ax.legend()
    fig.tight_layout()

    path = cfg.paths.data_dir.parent / "docs" / "latency.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    console.print(f"[dim]{path}[/]")


if __name__ == "__main__":
    main()
