"""View 4 -- semantic breakpoint chunking.

Embed every sentence, walk the sequence, and cut wherever the similarity between
adjacent sentences falls into the bottom ``breakpoint_percentile`` of that
passage's own gap distribution. The result is topic-coherent chunks whose length
adapts to the text instead of to a constant we guessed.

The percentile is computed *per passage*, not globally. A globally-fixed
similarity threshold behaves very differently on a tight factual passage (all
gaps high) than on a list-like one (all gaps low), and would either never cut the
first or shred the second. A per-passage percentile is scale-free.

Embedding cost is paid entirely at build time; at query time this view is just
more vectors in the same index.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from vrag.chunking.base import RawChunk, Sentence, split_sentences
from vrag.config import SemanticCfg
from vrag.schemas import ChunkView, Passage

EmbedFn = Callable[[list[str]], np.ndarray]


class SemanticChunker:
    view = ChunkView.SEMANTIC

    def __init__(self, cfg: SemanticCfg, embed_fn: EmbedFn) -> None:
        self.cfg = cfg
        self.embed_fn = embed_fn

    # -- batch entry point --------------------------------------------------- #
    def chunk_many(self, passages: list[Passage]) -> list[list[RawChunk]]:
        """Chunk a batch in one embedding call.

        Sentence embedding is the expensive part of the build; batching every
        passage's sentences into a single forward pass is what keeps this view
        from dominating index construction time.
        """
        per_passage: list[list[Sentence]] = []
        flat: list[str] = []

        for passage in passages:
            sentences = split_sentences(passage.text, min_chars=25, merge_short=True)
            per_passage.append(sentences)
            flat.extend(s.text for s in sentences)

        if not flat:
            return [[] for _ in passages]

        vectors = self.embed_fn(flat)

        out: list[list[RawChunk]] = []
        cursor = 0
        for passage, sentences in zip(passages, per_passage, strict=True):
            n = len(sentences)
            chunk_vecs = vectors[cursor : cursor + n]
            cursor += n
            out.append(self._segment(passage, sentences, chunk_vecs))
        return out

    def chunk(self, passage: Passage) -> list[RawChunk]:
        return self.chunk_many([passage])[0]

    # -- segmentation -------------------------------------------------------- #
    def _segment(
        self, passage: Passage, sentences: list[Sentence], vectors: np.ndarray
    ) -> list[RawChunk]:
        n = len(sentences)
        # Fewer than three sentences cannot form a meaningful breakpoint
        # distribution, and the result would just duplicate the atomic view.
        if n < 3:
            return []

        # Vectors are L2-normalized upstream, so a dot product is the cosine.
        sims = np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = float(np.percentile(sims, self.cfg.breakpoint_percentile))

        boundaries: list[int] = []
        run = 1
        for i, sim in enumerate(sims):
            at_max = run >= self.cfg.max_chunk_sentences
            if (sim <= threshold and run >= self.cfg.min_chunk_sentences) or at_max:
                boundaries.append(i + 1)
                run = 1
            else:
                run += 1

        starts = [0, *boundaries]
        ends = [*boundaries, n]

        out: list[RawChunk] = []
        for idx, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
            if lo >= hi:
                continue
            group = sentences[lo:hi]
            body = " ".join(s.text for s in group).strip()
            if not body:
                continue
            out.append(
                RawChunk(
                    text=body,
                    context_text=body,
                    char_start=group[0].start,
                    char_end=group[-1].end,
                    local_idx=idx,
                    extra={"n_sentences": str(hi - lo), "threshold": f"{threshold:.4f}"},
                )
            )

        # If segmentation produced a single chunk covering everything, the view
        # adds nothing over atomic -- drop it rather than pay for a duplicate vector.
        if len(out) <= 1:
            return []

        return out
