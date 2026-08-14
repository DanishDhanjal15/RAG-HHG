"""Index construction: passages -> chunks -> vectors -> FAISS + BM25.

Two passes, deliberately:

1. **Chunk and store.** Stream every passage through every enabled view and write
   the chunks straight to the columnar store. Cheap, and it establishes the exact
   chunk count.
2. **Embed and index.** Read the embed texts back in slices and write vectors into
   a preallocated on-disk ``float32`` memmap, then hand that to FAISS.

The alternative -- accumulate vectors in a Python list and ``np.vstack`` at the
end -- peaks at roughly 2x the final array (1.3 GB becomes 2.6 GB for 850k x 384)
and is the most likely way for a build to die on a free-tier container. Writing
into a preallocated memmap keeps resident memory at one batch.

The build is also *resumable*: each stage writes a marker, and a rerun skips
completed stages unless ``--force`` is given. Losing a 50-minute embed to a typo
in the BM25 stage is not an acceptable failure mode this close to a deadline.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vrag.chunking.proposition import PropositionCache, default_cache_path
from vrag.chunking.registry import ChunkRegistry
from vrag.config import Config
from vrag.index.dense import DenseIndex
from vrag.index.embedder import OnnxEmbedder
from vrag.index.sparse import SparseIndex
from vrag.index.store import ChunkStore, ChunkStoreWriter
from vrag.ingest.normalize import load_passages

Progress = Callable[[str, float], None]


def _noop(_stage: str, _pct: float) -> None:
    return


VECTORS_FILE = "vectors.f32"
MANIFEST_FILE = "manifest.json"


def subsample_by_query(cfg: Config, passages: list) -> list:  # noqa: ANN001
    """Shrink the corpus by dropping whole QUERIES, never individual passages.

    This distinction is not cosmetic -- getting it wrong silently invalidates the
    ablation. A query's gold labels are the ``is_selected`` passages *of that
    query*. Sample passages independently and you keep query Q's question while
    dropping the passage that answers it, so Q is scored as an unrecoverable miss
    no matter how good retrieval is. Recall then measures the sampler, and every
    arm looks equally bad.

    Dropping whole queries keeps each surviving query's gold set intact, and the
    dropped queries' passages simply stop existing -- which is exactly what a
    smaller corpus means. The queries that remain also keep every one of their
    non-selected passages, so they keep their hard negatives.
    """
    limit = cfg.corpus.max_queries
    if not limit:
        return passages

    query_ids = sorted({p.query_id for p in passages})
    if len(query_ids) <= limit:
        return passages

    keep = set(random.Random(cfg.corpus.seed).sample(query_ids, limit))
    return [p for p in passages if p.query_id in keep]


# --------------------------------------------------------------------------- #
# Stage 1 -- chunk
# --------------------------------------------------------------------------- #
def build_chunks(cfg: Config, embedder: OnnxEmbedder, progress: Progress = _noop) -> dict:
    passages = load_passages(cfg)

    passages = subsample_by_query(cfg, passages)
    total = len(passages)

    prop_cache = None
    if cfg.chunking.views.proposition.enabled:
        prop_cache = PropositionCache(default_cache_path(cfg.paths.model_dir))

    registry = ChunkRegistry(
        cfg,
        tokenizer=embedder.tokenizer,
        embed_fn=embedder.encode_raw,
        proposition_cache=prop_cache,
    )

    store_path = cfg.paths.index_dir / "chunks"
    t0 = time.perf_counter()
    written = 0

    with ChunkStoreWriter(store_path) as writer:
        for chunk in registry.build(passages):
            writer.add(chunk)
            written += 1
            if written % 20_000 == 0:
                progress("chunk", registry.stats.passages / max(1, total))

    stats = registry.stats.finalize()
    return {
        "passages": total,
        "chunks": written,
        "per_view": stats.per_view,
        "mean_chars": {k: round(v, 1) for k, v in stats.mean_chars.items()},
        "seconds": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------- #
# Stage 2 -- embed
# --------------------------------------------------------------------------- #
def build_vectors(
    cfg: Config, embedder: OnnxEmbedder, progress: Progress = _noop, slice_size: int = 8192
) -> dict:
    """Embed every chunk into an on-disk float32 memmap. **Resumable.**

    Resumability matters more here than anywhere else in the build. This is the
    multi-hour stage, and the file size tells you nothing about progress --
    ``np.memmap(mode="w+")`` preallocates the full array up front, so a
    half-finished run looks byte-identical to a finished one. Without a progress
    marker, any interruption silently costs the entire elapsed time.

    That is not hypothetical: this stage was killed at 40.5% by a process restart
    and had to start over. The marker below records the last fully-written row, so
    a rerun continues from there.
    """
    store = ChunkStore(cfg.paths.index_dir / "chunks")
    n = len(store)
    dim = cfg.embedding.dim

    path = cfg.paths.index_dir / VECTORS_FILE
    marker = cfg.paths.index_dir / ".embed.progress"

    # Resume only if the existing file matches the expected shape exactly. A
    # mismatch means the chunk set changed underneath us, and continuing would
    # interleave vectors from two different corpora.
    resume_from = 0
    expected_bytes = n * dim * 4
    if marker.exists() and path.exists() and path.stat().st_size == expected_bytes:
        try:
            done = int(json.loads(marker.read_text(encoding="utf-8"))["rows"])
            resume_from = max(0, min(done, n))
        except (ValueError, KeyError, json.JSONDecodeError):
            resume_from = 0

    mode = "r+" if resume_from else "w+"
    vectors = np.memmap(path, dtype=np.float32, mode=mode, shape=(n, dim))
    if resume_from:
        progress("embed (resumed)", resume_from / n)

    t0 = time.perf_counter()
    for start in range(resume_from, n, slice_size):
        end = min(start + slice_size, n)
        texts = store.iter_embed_texts(start, end)
        vectors[start:end] = embedder.encode_passages(texts)
        # Flush BEFORE recording progress, so the marker can never claim rows
        # that are not actually on disk.
        vectors.flush()
        marker.write_text(json.dumps({"rows": end, "total": n}), encoding="utf-8")
        progress("embed", end / n)

    vectors.flush()
    del vectors
    marker.unlink(missing_ok=True)

    return {
        "vectors": n,
        "dim": dim,
        "resumed_from": resume_from,
        "seconds": round(time.perf_counter() - t0, 1),
        "bytes": path.stat().st_size,
    }


def open_vectors(cfg: Config) -> np.memmap:
    store = ChunkStore(cfg.paths.index_dir / "chunks")
    return np.memmap(
        cfg.paths.index_dir / VECTORS_FILE,
        dtype=np.float32,
        mode="r",
        shape=(len(store), cfg.embedding.dim),
    )


# --------------------------------------------------------------------------- #
# Stage 3 -- dense index + corpus centroid
# --------------------------------------------------------------------------- #
def build_dense(cfg: Config, progress: Progress = _noop) -> dict:
    vectors = open_vectors(cfg)
    t0 = time.perf_counter()

    index = DenseIndex.build(cfg, np.asarray(vectors))
    path = index.save()
    progress("dense", 1.0)

    # The corpus centroid powers the out-of-domain guardrail: a query far from the
    # manifold has nothing to retrieve, regardless of what its top-1 score says.
    # Computed in slices so it never materializes the full matrix twice.
    acc = np.zeros(cfg.embedding.dim, dtype=np.float64)
    step = 100_000
    for start in range(0, len(vectors), step):
        acc += vectors[start : start + step].sum(axis=0, dtype=np.float64)
    centroid = (acc / len(vectors)).astype(np.float32)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    np.save(cfg.paths.index_dir / "centroid.npy", centroid)

    return {
        "index_type": cfg.dense.index_type,
        "quantizer": cfg.dense.hnsw.quantizer if cfg.dense.index_type == "hnsw" else "pq",
        "ntotal": int(index.size),
        "bytes": path.stat().st_size,
        "seconds": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------- #
# Stage 4 -- sparse index
# --------------------------------------------------------------------------- #
def build_sparse(cfg: Config, progress: Progress = _noop) -> dict:
    if not cfg.sparse.enabled:
        return {"enabled": False}

    store = ChunkStore(cfg.paths.index_dir / "chunks")
    t0 = time.perf_counter()

    texts = store.iter_embed_texts()
    progress("sparse", 0.5)
    index = SparseIndex.build(cfg, texts)
    path = index.save(cfg)
    progress("sparse", 1.0)

    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return {"enabled": True, "documents": len(texts), "bytes": size,
            "seconds": round(time.perf_counter() - t0, 1)}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _marker(cfg: Config, stage: str) -> Path:
    return cfg.paths.index_dir / f".{stage}.done"


def build_all(
    cfg: Config, force: bool = False, progress: Progress = _noop
) -> dict:
    cfg.paths.ensure()
    cfg.paths.index_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cfg.paths.index_dir / MANIFEST_FILE
    manifest: dict = {}
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Build-mode embedder: every core, since indexing is throughput-bound.
    # The serving pipeline constructs its own with the low-thread serve settings.
    embedder = OnnxEmbedder(cfg, build_mode=True)

    stages: list[tuple[str, Callable[[], dict]]] = [
        ("chunk", lambda: build_chunks(cfg, embedder, progress)),
        ("embed", lambda: build_vectors(cfg, embedder, progress)),
        ("dense", lambda: build_dense(cfg, progress)),
        ("sparse", lambda: build_sparse(cfg, progress)),
    ]

    for name, fn in stages:
        marker = _marker(cfg, name)
        if marker.exists() and not force:
            progress(f"{name} (cached)", 1.0)
            continue
        manifest[name] = fn()
        marker.write_text(json.dumps(manifest[name]), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest["config"] = {
        "embedding_model": cfg.embedding.model_id,
        "views": cfg.chunking.views.enabled_names(),
        "dense": asdict_safe(cfg.dense),
        "corpus": asdict_safe(cfg.corpus),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def asdict_safe(model) -> dict:  # noqa: ANN001
    return model.model_dump() if hasattr(model, "model_dump") else asdict(model)
