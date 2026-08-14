"""View 1 -- atomic: the whole passage is one chunk.

This is the control arm of the ablation. MS MARCO passages are already
retrieval-sized (median ~60 words), so any fancier view has to *beat this* to
justify the vectors it costs. Several published chunking comparisons never
include the do-nothing baseline, which is how obviously-worse strategies get
shipped.
"""

from __future__ import annotations

from vrag.chunking.base import RawChunk
from vrag.schemas import ChunkView, Passage


class AtomicChunker:
    view = ChunkView.ATOMIC

    def chunk(self, passage: Passage) -> list[RawChunk]:
        text = passage.text.strip()
        if not text:
            return []
        return [
            RawChunk(
                text=text,
                context_text=text,
                char_start=0,
                char_end=len(passage.text),
                local_idx=0,
            )
        ]
