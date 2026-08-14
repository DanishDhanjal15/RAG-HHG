"""Reciprocal Rank Fusion across chunking views and retrieval modalities.

Why RRF rather than score blending: the five views produce scores on genuinely
different scales. A one-sentence ``sentence_window`` chunk and a 250-token
``fixed_overlap`` chunk, both perfectly relevant, do not receive comparable cosine
similarities -- short texts concentrate, long texts average out. Any weighted sum
of raw scores silently prefers whichever view happens to score higher, and the
same is true across dense (cosine, [-1,1]) and BM25 (unbounded, corpus-dependent).

RRF throws away magnitudes and keeps only *rank*, which is the one thing every
run agrees on the meaning of::

    score(d) = Σ_runs  w_run / (k + rank_run(d))

It also gives us a genuine signal for free: a chunk found by several independent
views accumulates contributions from each, so **multi-view agreement is itself
evidence of relevance**. That is the actual payoff of running five views rather
than one, and it is why fusion happens before dedup rather than after.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from vrag.config import DedupCfg, FusionCfg
from vrag.schemas import ChunkView, ScoredChunk


@dataclass(slots=True)
class Candidate:
    chunk_id: int
    score: float
    view: ChunkView
    views_hit: int = 1
    doc_id: str = ""
    span: tuple[int, int] = (0, 0)


def rrf_fuse(
    runs: dict[str, list[ScoredChunk]],
    cfg: FusionCfg,
    modality_weights: dict[str, float] | None = None,
) -> list[Candidate]:
    """Fuse named ranked runs into one list, best first.

    ``runs`` keys are ``"{modality}:{view}"`` (e.g. ``"dense:atomic"``), so a view's
    dense and sparse evidence are separate runs and both contribute.
    """
    k = cfg.rrf_k
    acc: dict[int, float] = defaultdict(float)
    best_view: dict[int, tuple[float, ChunkView]] = {}
    hits: dict[int, set[str]] = defaultdict(set)

    for run_name, hits_list in runs.items():
        if not hits_list:
            continue
        modality, _, view = run_name.partition(":")
        weight = cfg.view_weights.get(view, 1.0)
        if modality_weights:
            weight *= modality_weights.get(modality, 1.0)
        if weight <= 0:
            continue

        for rank, hit in enumerate(hits_list, start=1):
            acc[hit.chunk_id] += weight / (k + rank)
            hits[hit.chunk_id].add(view)
            # Remember which view ranked this chunk best, for explainability in
            # the UI ("found by: sentence_window").
            prev = best_view.get(hit.chunk_id)
            if prev is None or hit.score > prev[0]:
                best_view[hit.chunk_id] = (hit.score, hit.view)

    out = [
        Candidate(
            chunk_id=chunk_id,
            score=score,
            view=best_view[chunk_id][1],
            views_hit=len(hits[chunk_id]),
        )
        for chunk_id, score in acc.items()
    ]
    out.sort(key=lambda c: (-c.score, c.chunk_id))
    return out


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0, hi - lo)
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def dedup(candidates: list[Candidate], cfg: DedupCfg) -> list[Candidate]:
    """Collapse candidates whose character spans substantially overlap.

    Five views over one passage produce overlapping text *by construction*: a
    sentence-window chunk, the semantic chunk containing it, and the atomic
    passage all cover the same words. Without this, the top-8 handed to the
    generator can be the same paragraph five times, and the reranker burns its
    entire budget scoring duplicates.

    Runs after fusion, so a chunk that several views agreed on has already been
    rewarded for that agreement before its duplicates are dropped.
    """
    if not cfg.enabled:
        return candidates

    kept: list[Candidate] = []
    by_doc: dict[str, list[Candidate]] = defaultdict(list)
    per_doc_count: dict[str, int] = defaultdict(int)

    for cand in candidates:  # already sorted best-first
        siblings = by_doc[cand.doc_id]
        if any(_iou(cand.span, other.span) >= cfg.span_iou_threshold for other in siblings):
            continue
        # Cap chunks per source passage so one verbose document cannot occupy
        # every context slot and starve genuinely different evidence.
        if cfg.max_per_doc and per_doc_count[cand.doc_id] >= cfg.max_per_doc:
            continue

        siblings.append(cand)
        per_doc_count[cand.doc_id] += 1
        kept.append(cand)

    return kept
