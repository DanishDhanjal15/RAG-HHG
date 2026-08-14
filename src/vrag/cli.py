"""``vrag`` command line.

Every stage of the build is a command so the pipeline can be rebuilt from scratch
reproducibly, and so each gate (ingest -> build -> ablate -> bench) can be run and
verified independently.
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vrag.config import get_config, get_secrets

app = typer.Typer(add_completion=False, help="Voice-enabled multilingual RAG.")
console = Console()


@app.command()
def probe(
    lang: str = typer.Option("hi", help="Language key from configs/default.yaml"),
    rows: int = typer.Option(2, help="How many sample rows to render"),
) -> None:
    """Inspect the real arrow schema of a shard over HTTP, without downloading it.

    The HF dataset viewer for this repo is broken, so this is how we confirm the
    on-disk layout before committing to a 1.4 GB download and a long ingest.
    """
    import pyarrow.parquet as pq

    from vrag.ingest.download import open_remote
    from vrag.ingest.normalize import extract_passages

    cfg = get_config()
    remote = cfg.corpus.languages[lang]
    console.print(f"[bold]Probing[/] {cfg.corpus.repo_id}/{remote}")

    with open_remote(cfg, remote) as fh:
        pf = pq.ParquetFile(fh)
        console.print(f"[bold]rows:[/] {pf.metadata.num_rows:,}   "
                      f"[bold]row groups:[/] {pf.metadata.num_row_groups}")
        console.print("\n[bold]schema[/]")
        console.print(str(pf.schema_arrow))

        batch = next(pf.iter_batches(batch_size=rows))
        for i, row in enumerate(batch.to_pylist()[:rows]):
            console.rule(f"row {i}")
            for key, value in row.items():
                if key == "passages":
                    items = extract_passages(value)
                    console.print(f"[cyan]passages[/]: {len(items)} items")
                    for j, item in enumerate(items[:2]):
                        console.print(
                            f"  [{j}] is_selected={item['is_selected']}\n"
                            f"      translated: {str(item['translated'])[:160]}\n"
                            f"      english   : {str(item['english'])[:160]}"
                        )
                elif key == "meta":
                    console.print(f"[dim]meta[/]: {str(value)[:120]}")
                else:
                    console.print(f"[cyan]{key}[/]: {str(value)[:200]}")


@app.command()
def ingest(
    skip_download: bool = typer.Option(False, help="Reuse shards already in data/raw"),
) -> None:
    """Download shards and normalize them into passages.parquet / queries.parquet."""
    from vrag.ingest.download import download_shards
    from vrag.ingest.normalize import (
        CorpusStats,
        normalize_shard,
        plan_rows,
        write_corpus,
    )

    cfg = get_config()
    cfg.paths.ensure()
    plans = plan_rows(cfg)

    console.print("[bold]Row plan[/]")
    for lang, plan in plans.items():
        console.print(f"  {lang}: {plan.ranges}  ({plan.total:,} queries)")

    if skip_download:
        shards = []
        for lang, remote in cfg.corpus.languages.items():
            local = cfg.paths.raw_dir / remote
            if not local.exists():
                raise typer.BadParameter(f"{local} missing; run without --skip-download")
            shards.append((lang, local))
    else:
        shards = [(s.lang, s.local_path) for s in download_shards(cfg)]

    stats = CorpusStats()
    seen_english: set[str] = set()
    all_passages, all_queries = [], []

    for lang, path in shards:
        t0 = time.perf_counter()
        passages, queries = normalize_shard(
            cfg, lang, Path(path), plans[lang], seen_english, stats
        )
        all_passages.extend(passages)
        all_queries.extend(queries)
        console.print(
            f"  [green]{lang}[/]: {len(queries):,} queries, {len(passages):,} passages "
            f"({time.perf_counter() - t0:.1f}s)"
        )

    stats.queries = len(all_queries)
    stats.passages = len(all_passages)
    ppath, qpath = write_corpus(cfg, all_passages, all_queries)

    table = Table(title="Corpus")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("queries", f"{stats.queries:,}")
    table.add_row("passages", f"{stats.passages:,}")
    table.add_row("  of which English parallel", f"{stats.english_passages:,}")
    for lang, n in stats.per_lang.items():
        table.add_row(f"  {lang}", f"{n:,}")
    table.add_row("skipped (too short)", f"{stats.skipped_short:,}")
    table.add_row("rows without passages", f"{stats.rows_without_passages:,}")
    console.print(table)
    console.print(f"[dim]{ppath}\n{qpath}[/]")


@app.command()
def build(
    force: bool = typer.Option(False, help="Rebuild every stage, ignoring cached markers"),
) -> None:
    """Chunk, embed, and index the corpus.

    Resumable: each stage writes a marker and a rerun skips completed stages.
    On CPU-only hardware the embed stage is measured in hours, and losing it to a
    typo in a later stage is not an acceptable failure mode.
    """
    import json

    from vrag.index.build import build_all

    cfg = get_config()
    console.print(f"[bold]Building[/] {cfg.paths.index_dir}")
    console.print(f"  views: {', '.join(cfg.chunking.views.enabled_names())}")
    console.print(f"  max_queries: {cfg.corpus.max_queries or 'all'}")

    state = {"stage": "", "pct": -1.0}

    def progress(stage: str, pct: float) -> None:
        step = int(pct * 100) // 5
        if stage != state["stage"] or step != state["pct"]:
            state["stage"], state["pct"] = stage, step
            console.print(f"  [dim]{stage:<16}[/] {pct * 100:5.1f}%")

    manifest = build_all(cfg, force=force, progress=progress)

    table = Table(title="Index")
    table.add_column("stage")
    table.add_column("detail")
    for stage in ("chunk", "embed", "dense", "sparse"):
        info = manifest.get(stage)
        if not info:
            continue
        if stage == "chunk":
            views = "  ".join(f"{k}={v:,}" for k, v in sorted(info["per_view"].items()))
            detail = f"{info['chunks']:,} chunks from {info['passages']:,} passages\n{views}"
        elif stage == "embed":
            detail = f"{info['vectors']:,} x {info['dim']}  ({info['bytes'] / 1e6:.0f} MB)"
        elif stage == "dense":
            detail = (f"{info['index_type']}/{info['quantizer']}  "
                      f"{info['ntotal']:,} vectors  ({info['bytes'] / 1e6:.0f} MB)")
        else:
            detail = (f"{info.get('documents', 0):,} docs  ({info.get('bytes', 0) / 1e6:.0f} MB)"
                      if info.get("enabled") else "disabled")
        table.add_row(stage, f"{detail}\n[dim]{info.get('seconds', 0)}s[/]")
    console.print(table)
    console.print(f"[dim]{json.dumps(manifest.get('chunk', {}).get('mean_chars', {}))}[/]")


@app.command()
def ask(
    text: str = typer.Argument(..., help="The question"),
    lang: str = typer.Option("unknown", help="ISO-639-1 hint"),
    show_spans: bool = typer.Option(False, help="Print every stage timing"),
) -> None:
    """Ask a question from the terminal. The fastest way to sanity-check the core."""
    from vrag.harness.pipeline import Pipeline

    cfg = get_config()
    t0 = time.perf_counter()
    pipeline = Pipeline(cfg)
    console.print(f"[dim]boot {time.perf_counter() - t0:.1f}s[/]")

    envelope = pipeline.answer_text(text, lang=lang)

    if envelope.abstained:
        console.print(f"\n[bold yellow]{envelope.refusal_reason}[/]")
        console.print(envelope.refusal_detail)
    else:
        console.print(f"\n[bold]{envelope.answer}[/]")

    if envelope.citations:
        console.print("\n[dim]sources[/]")
        for c in envelope.citations:
            console.print(f"  [cyan]{c.lang}[/] {c.doc_id}  [dim]{c.score:.3f}[/]")
            console.print(f"    {c.quote[:160]}")

    verdict = "[green]within[/]" if envelope.within_budget else "[red]OVER[/]"
    console.print(
        f"\ncore [bold]{envelope.core_latency_ms:.1f}ms[/] "
        f"({verdict} {cfg.budget.core_budget_ms:.0f}ms budget)"
    )
    if show_spans:
        for name, ms in envelope.timings_ms.items():
            console.print(f"  {name:<18} {ms:7.2f}ms")
    for d in envelope.degradations:
        console.print(f"  [yellow]dropped {d.stage}[/]: {d.reason}")

    pipeline.close()


@app.command()
def serve(
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
    reload: bool = typer.Option(False),
) -> None:
    """Run the web app (mic UI + JSON API)."""
    import uvicorn

    cfg = get_config()
    uvicorn.run(
        "vrag.server.app:app",
        host=host or cfg.server.host,
        port=port or cfg.server.port,
        reload=reload,
    )


@app.command()
def propositions(
    limit: int = typer.Option(0, help="Cap passages to extract (0 = config default)"),
    concurrency: int = typer.Option(8),
) -> None:
    """Offline LLM pass that fills the proposition-view cache.

    Run once, before enabling the proposition view. Cached to disk and resumable,
    so a crash costs minutes rather than the whole pass.
    """
    import anyio

    from vrag.chunking.proposition import (
        PropositionCache,
        default_cache_path,
        extract_propositions,
    )
    from vrag.ingest.normalize import load_passages

    cfg = get_config()
    secrets = get_secrets()
    if not secrets.anthropic_api_key:
        raise typer.BadParameter("VRAG_ANTHROPIC_API_KEY is not set")

    cache = PropositionCache(default_cache_path(cfg.paths.model_dir))
    passages = load_passages(cfg)
    # The selected passages are the ones that actually answer a query, so they are
    # where propositionalising pays. Doing all 224k would cost far more for far less.
    selected = [p for p in passages if p.is_selected]
    top = limit or cfg.chunking.views.proposition.top_passages
    selected = selected[:top]

    console.print(f"extracting propositions for {len(selected):,} selected passages "
                  f"({len(cache.data):,} already cached)")
    n = anyio.run(
        extract_propositions, selected, cache, secrets.anthropic_api_key,
        cfg.generation.llm.model, concurrency,
        cfg.chunking.views.proposition.max_props_per_passage,
    )
    console.print(f"[green]extracted {n:,} new[/]  total cached: {len(cache.data):,}")
    console.print("Now set chunking.views.proposition.enabled: true and rerun `vrag build --force`")


@app.command()
def doctor() -> None:
    """Check that the environment can actually run the pipeline."""
    cfg = get_config()
    secrets = get_secrets()

    table = Table(title="Environment")
    table.add_column("check")
    table.add_column("status")

    def row(name: str, ok: bool, detail: str = "") -> None:
        table.add_row(name, ("[green]ok[/] " if ok else "[red]missing[/] ") + detail)

    import importlib

    for mod in ("numpy", "pyarrow", "faiss", "bm25s", "onnxruntime", "transformers", "fastapi"):
        try:
            m = importlib.import_module(mod)
            row(mod, True, getattr(m, "__version__", ""))
        except Exception as exc:  # noqa: BLE001
            row(mod, False, str(exc)[:60])

    row("SARVAM_API_KEY", bool(secrets.sarvam_api_key))
    row("ANTHROPIC_API_KEY (optional)", bool(secrets.anthropic_api_key))
    row("corpus", (cfg.paths.corpus_dir / "passages.parquet").exists(), str(cfg.paths.corpus_dir))
    row("index", (cfg.paths.index_dir / "dense.faiss").exists(), str(cfg.paths.index_dir))
    console.print(table)


if __name__ == "__main__":
    app()
