"""Multi-view chunk orchestration.

Runs every enabled view over the same corpus and emits one flat stream of
``Chunk`` records tagged with the view that produced them. They all land in a
single FAISS index and a single BM25 index -- *not* one index per view -- so a
query costs one dense search plus one sparse search regardless of how many views
are enabled. Multi-view retrieval that pays a separate index round trip per view
does not fit in a 200 ms budget; multi-view retrieval over one co-indexed store does.

Views are separated at *scoring* time instead, via the ``view`` field: fusion
groups hits by view, applies per-view weights, and reciprocal-rank-fuses them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vrag.chunking.atomic import AtomicChunker
from vrag.chunking.base import RawChunk, contextualize
from vrag.chunking.fixed_overlap import FixedOverlapChunker
from vrag.chunking.proposition import PropositionCache, PropositionChunker
from vrag.chunking.semantic import SemanticChunker
from vrag.chunking.sentence_window import SentenceWindowChunker
from vrag.config import Config
from vrag.schemas import Chunk, ChunkView, Passage

EmbedFn = Callable[[list[str]], np.ndarray]


@dataclass
class ChunkStats:
    per_view: dict[str, int] = field(default_factory=dict)
    passages: int = 0
    total: int = 0
    mean_chars: dict[str, float] = field(default_factory=dict)
    _chars: dict[str, int] = field(default_factory=dict, repr=False)

    def record(self, view: str, n_chars: int) -> None:
        self.per_view[view] = self.per_view.get(view, 0) + 1
        self._chars[view] = self._chars.get(view, 0) + n_chars
        self.total += 1

    def finalize(self) -> ChunkStats:
        self.mean_chars = {
            v: self._chars[v] / n for v, n in self.per_view.items() if n
        }
        return self


class ChunkRegistry:
    """Builds every enabled view and assigns global chunk ids.

    The chunk id is the FAISS row position, so it is assigned here and nowhere
    else -- it is the join key between the dense index, the sparse index, the
    payload store, and every citation the system ever emits.
    """

    def __init__(
        self,
        cfg: Config,
        tokenizer: Any | None = None,
        embed_fn: EmbedFn | None = None,
        proposition_cache: PropositionCache | None = None,
    ) -> None:
        self.cfg = cfg
        views = cfg.chunking.views
        self.chunkers: dict[ChunkView, Any] = {}

        if views.atomic.enabled:
            self.chunkers[ChunkView.ATOMIC] = AtomicChunker()

        if views.sentence_window.enabled:
            self.chunkers[ChunkView.SENTENCE_WINDOW] = SentenceWindowChunker(
                views.sentence_window
            )

        if views.fixed_overlap.enabled:
            if tokenizer is None:
                raise ValueError(
                    "fixed_overlap view needs a tokenizer -- it windows on tokens, "
                    "not characters, so that boundaries never split a grapheme cluster"
                )
            self.chunkers[ChunkView.FIXED_OVERLAP] = FixedOverlapChunker(
                views.fixed_overlap, tokenizer
            )

        if views.semantic.enabled:
            if embed_fn is None:
                raise ValueError("semantic view needs an embed_fn to find breakpoints")
            self.chunkers[ChunkView.SEMANTIC] = SemanticChunker(views.semantic, embed_fn)

        if views.proposition.enabled:
            if proposition_cache is None:
                raise ValueError(
                    "proposition view needs a populated cache -- run `vrag propositions` first"
                )
            self.chunkers[ChunkView.PROPOSITION] = PropositionChunker(
                views.proposition, proposition_cache
            )

        self._next_id = 0
        self.stats = ChunkStats()

    # -- building ------------------------------------------------------------ #
    def build(self, passages: Iterable[Passage], batch_size: int = 256) -> Iterator[Chunk]:
        batch: list[Passage] = []
        for passage in passages:
            batch.append(passage)
            if len(batch) >= batch_size:
                yield from self._build_batch(batch)
                batch = []
        if batch:
            yield from self._build_batch(batch)

    def _build_batch(self, passages: list[Passage]) -> Iterator[Chunk]:
        # The semantic view embeds every sentence, so it runs once per batch
        # rather than once per passage -- that single change is the difference
        # between a 30-minute and a 6-hour index build on CPU.
        semantic_out: dict[str, list[RawChunk]] = {}
        semantic = self.chunkers.get(ChunkView.SEMANTIC)
        if semantic is not None:
            results = semantic.chunk_many(passages)
            semantic_out = {p.doc_id: r for p, r in zip(passages, results, strict=True)}

        for passage in passages:
            self.stats.passages += 1
            for view, chunker in self.chunkers.items():
                raws = (
                    semantic_out.get(passage.doc_id, [])
                    if view is ChunkView.SEMANTIC
                    else chunker.chunk(passage)
                )
                yield from self._materialize(passage, view, raws)

    def _materialize(
        self, passage: Passage, view: ChunkView, raws: list[RawChunk]
    ) -> Iterator[Chunk]:
        if not raws:
            return

        start_id = self._next_id
        ids = list(range(start_id, start_id + len(raws)))
        self._next_id += len(raws)

        for pos, (chunk_id, raw) in enumerate(zip(ids, raws, strict=True)):
            embed_text = (
                contextualize(passage, raw) if self.cfg.chunking.contextual_prefix else raw.text
            )
            # Neighbours are same-passage, same-view siblings. The harness's
            # `fetch_neighbours` tool uses these to widen context without a
            # second retrieval round.
            neighbours = [i for i in (ids[pos - 1] if pos > 0 else None,
                                      ids[pos + 1] if pos + 1 < len(ids) else None)
                          if i is not None]

            self.stats.record(view.value, len(raw.text))

            yield Chunk(
                chunk_id=chunk_id,
                doc_id=passage.doc_id,
                query_id=passage.query_id,
                lang=passage.lang,
                view=view,
                text=embed_text,
                context_text=raw.context_text,
                char_start=raw.char_start,
                char_end=raw.char_end,
                is_selected=passage.is_selected,
                parallel_en_id=passage.parallel_en_id,
                neighbour_ids=neighbours,
            )
