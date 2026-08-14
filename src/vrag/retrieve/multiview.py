"""Multi-view retrieval over a single co-indexed store.

The five chunking views live in one FAISS index and one BM25 index, tagged by
``view``. So a query costs **one** dense search and **one** sparse search no
matter how many views are enabled -- and the per-view structure that fusion needs
is recovered by *bucketing* the single result list by view afterwards.

The alternative (an index per view, searched in parallel) costs N searches, N
result merges, and N times the memory. At a 200 ms budget where dense search is
allotted 25 ms, that is not a trade worth making.

The one thing bucketing needs care with: if a single view dominates the global
top-k, the others get empty buckets and contribute nothing to fusion. So the dense
search asks for ``top_k_per_view x n_views`` neighbours rather than ``top_k_fused``,
which costs ~1 ms more on HNSW and keeps every view represented.

Methods are deliberately fine-grained (embed / dense / sparse / fuse) rather than
one ``retrieve()`` call, so the harness can time, budget, and individually skip
each one.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from vrag.config import Config
from vrag.index.dense import DenseIndex
from vrag.index.embedder import OnnxEmbedder
from vrag.index.sparse import SparseIndex
from vrag.index.store import ChunkStore
from vrag.retrieve.fusion import Candidate, dedup, rrf_fuse
from vrag.schemas import ChunkView, Evidence, QueryPlan, ScoredChunk

MAX_DENSE_K = 240


class MultiViewRetriever:
    def __init__(
        self,
        cfg: Config,
        embedder: OnnxEmbedder,
        dense: DenseIndex,
        store: ChunkStore,
        sparse: SparseIndex | None = None,
    ) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.dense = dense
        self.store = store
        self.sparse = sparse
        self.centroid = self._load_centroid()
        self._lang_mask_cache: dict[tuple, np.ndarray] = {}
        self._warm_filters()

    def _warm_filters(self) -> None:
        """Precompute every language-filter id array at boot.

        Building one costs a full pass over the chunk metadata plus a flatnonzero
        over ~1M rows. Doing that lazily means the FIRST request in each language
        pays it -- which is exactly the request a demo is most likely to show.
        Paying it once at startup keeps it out of the served latency entirely.
        """
        lf = self.cfg.retrieval.language_filter
        for lang in self.store.languages:
            langs = [lang]
            if lf.always_include_english and lang != "en":
                langs.append("en")
            plan = QueryPlan(raw_query="", normalized_query="", embed_text="",
                             lang=lang, lang_filter=langs, views=self.enabled_views)
            self._allowed_ids(plan)

    def _load_centroid(self) -> np.ndarray | None:
        path = self.cfg.paths.index_dir / "centroid.npy"
        return np.load(path) if path.exists() else None

    # -- stage: embed -------------------------------------------------------- #
    def embed_query(self, plan: QueryPlan) -> np.ndarray:
        return self.embedder.encode_query(plan.normalized_query)

    # -- stage: dense -------------------------------------------------------- #
    def dense_search(
        self, vector: np.ndarray, plan: QueryPlan
    ) -> dict[str, list[ScoredChunk]]:
        n_views = max(1, len(plan.views))
        k = min(self.cfg.retrieval.top_k_per_view * n_views, MAX_DENSE_K)

        allowed = self._allowed_ids(plan)
        ids, scores = self.dense.search(vector, k, allowed_ids=allowed)

        # Vectors are L2-normalized and the index uses inner product, so these
        # scores ARE cosine similarities. Capture the best one now: it is the
        # out-of-domain guard's primary signal and this is the only place it is
        # available for free. (Recomputing it later via reconstruct() is not an
        # option -- the deployed SQ8 index cannot reconstruct.)
        valid = scores[ids >= 0]
        plan.top_dense_score = float(valid.max()) if valid.size else 0.0

        return self._bucket(ids, scores, "dense", plan)

    def _allowed_ids(self, plan: QueryPlan) -> np.ndarray | None:
        """ID selector combining the language filter and any view restriction.

        The language filter is only applied when ASR was confident: filtering on a
        low-confidence language guess is worse than not filtering, because it
        removes the correct passages and leaves the model answering from whatever
        survived.

        The view restriction is normally a no-op at serve time (all views are
        searched). It exists for the chunking ablation, where each single-view arm
        must get its own full top-k from the index rather than the leftovers of a
        mixed result list.

        Selectors are cached per key -- materialising a 1M-element id array on
        every request would cost more than the search it constrains.
        """
        views = set(plan.views or [])
        all_views = set(self.enabled_views)
        view_restricted = bool(views) and views != all_views

        if not plan.lang_filter and not view_restricted:
            return None

        key = (
            tuple(sorted(plan.lang_filter)) if plan.lang_filter else (),
            tuple(sorted(v.value for v in views)) if view_restricted else (),
        )
        cached = self._lang_mask_cache.get(key)
        if cached is None:
            mask = np.ones(len(self.store), dtype=bool)
            if plan.lang_filter:
                mask &= self.store.lang_mask(list(plan.lang_filter))
            if view_restricted:
                mask &= self.store.view_mask(list(views))
            cached = np.flatnonzero(mask).astype(np.int64)
            self._lang_mask_cache[key] = cached
        return cached

    # -- stage: sparse ------------------------------------------------------- #
    def sparse_search(self, plan: QueryPlan) -> dict[str, list[ScoredChunk]]:
        if self.sparse is None:
            return {}
        k = min(self.cfg.retrieval.top_k_per_view * max(1, len(plan.views)), MAX_DENSE_K)
        ids, scores = self.sparse.search(plan.normalized_query, k)
        if len(ids) == 0:
            return {}
        return self._bucket(ids, scores, "sparse", plan)

    # -- bucketing ----------------------------------------------------------- #
    def _bucket(
        self,
        ids: np.ndarray,
        scores: np.ndarray,
        modality: str,
        plan: QueryPlan,
    ) -> dict[str, list[ScoredChunk]]:
        wanted = set(plan.views) if plan.views else None
        runs: dict[str, list[ScoredChunk]] = defaultdict(list)

        for chunk_id, score in zip(ids, scores, strict=True):
            if chunk_id < 0:
                continue
            cid = int(chunk_id)
            view = self.store.view(cid)
            if wanted is not None and view not in wanted:
                continue
            run = runs[f"{modality}:{view.value}"]
            run.append(
                ScoredChunk(
                    chunk_id=cid,
                    score=float(score),
                    view=view,
                    source=modality,  # type: ignore[arg-type]
                    rank=len(run) + 1,
                )
            )

        return dict(runs)

    # -- stage: fuse --------------------------------------------------------- #
    def fuse(self, runs: dict[str, list[ScoredChunk]]) -> list[Candidate]:
        rcfg = self.cfg.retrieval
        candidates = rrf_fuse(
            runs,
            rcfg.fusion,
            modality_weights={
                "dense": rcfg.fusion.dense_weight,
                "sparse": rcfg.fusion.sparse_weight,
            },
        )

        # Hydrate the metadata dedup needs. Only for the head of the list --
        # dedupping candidates that will never reach the context window is
        # budget spent for nothing.
        head = candidates[: rcfg.top_k_fused * 3]
        for cand in head:
            cand.doc_id = self.store.doc_id(cand.chunk_id)
            cand.span = (
                int(self.store.char_start[cand.chunk_id]),
                int(self.store.char_end[cand.chunk_id]),
            )

        return dedup(head, rcfg.dedup)[: rcfg.top_k_fused]

    # -- hydration ----------------------------------------------------------- #
    def hydrate(self, candidates: list[Candidate], limit: int | None = None) -> list[Evidence]:
        limit = limit or self.cfg.retrieval.top_k_final
        out: list[Evidence] = []
        for cand in candidates[:limit]:
            payload = self.store.payload(cand.chunk_id)
            out.append(
                Evidence(
                    chunk_id=payload.chunk_id,
                    doc_id=payload.doc_id,
                    lang=payload.lang,
                    view=payload.view,
                    text=payload.text,
                    score=cand.score,
                    is_selected=payload.is_selected,
                )
            )
        return out

    # -- guardrail signals --------------------------------------------------- #
    def cosine_top1(self, plan: QueryPlan) -> float:
        """Absolute cosine similarity of the best dense hit.

        The out-of-domain guard needs an *absolute* similarity, and RRF scores
        cannot provide one: they encode rank, so a query with no good match still
        produces a top score near 1/(k+1) -- indistinguishable from a query with a
        perfect match. Hence the raw cosine.

        It is captured during dense search rather than recomputed afterwards. The
        alternative -- reconstruct the winning vector and dot it -- does work on
        this FAISS build (SQ8 supports reconstruct), but it costs an extra
        dequantize plus a dot product on the critical path to recover a number the
        search already computed, and it silently returns a *dequantized* vector, so
        the value would differ slightly from the score that actually ranked the
        result. Reading it off the search is free and exact.
        """
        return plan.top_dense_score

    def centroid_distance(self, vector: np.ndarray) -> float:
        """Cosine distance from the corpus centroid.

        Second, independent out-of-domain signal. A query can score a decent
        similarity against one lucky chunk while sitting nowhere near the corpus
        as a whole; requiring both signals to pass catches that.
        """
        if self.centroid is None:
            return 0.0
        return float(1.0 - np.dot(vector, self.centroid))

    def neighbours(self, chunk_id: int) -> list[Evidence]:
        """Widen context without a second retrieval round. Backs the harness's
        ``fetch_neighbours`` tool."""
        out = []
        for nid in self.store.neighbour_ids(chunk_id):
            payload = self.store.payload(nid)
            out.append(
                Evidence(
                    chunk_id=payload.chunk_id,
                    doc_id=payload.doc_id,
                    lang=payload.lang,
                    view=payload.view,
                    text=payload.text,
                    score=0.0,
                    is_selected=payload.is_selected,
                )
            )
        return out

    @property
    def enabled_views(self) -> list[ChunkView]:
        return [ChunkView(v) for v in self.cfg.chunking.views.enabled_names()]
