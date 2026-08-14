"""View 2 -- sentence window: embed narrow, return wide.

Each sentence becomes its own vector, which makes matching precise: a query about
one specific fact scores against a vector that contains only that fact, instead of
being diluted by the other five sentences in the passage. But a lone sentence is
often too thin to *answer* from, so the chunk carries its neighbours (+/- ``window``)
as ``context_text``.

This decoupling -- small embedded unit, large returned unit -- is the single
highest-value idea in the chunking design, and it is why ``text`` and
``context_text`` are separate fields on ``RawChunk``.
"""

from __future__ import annotations

from vrag.chunking.base import RawChunk, split_sentences
from vrag.config import SentenceWindowCfg
from vrag.schemas import ChunkView, Passage


class SentenceWindowChunker:
    view = ChunkView.SENTENCE_WINDOW

    def __init__(self, cfg: SentenceWindowCfg) -> None:
        self.cfg = cfg

    def chunk(self, passage: Passage) -> list[RawChunk]:
        sentences = split_sentences(
            passage.text,
            min_chars=self.cfg.min_sentence_chars,
            merge_short=self.cfg.merge_short_into_next,
        )
        if not sentences:
            return []

        # A single-sentence passage is exactly the atomic view; emitting it here
        # would double-index identical text for zero recall gain.
        if len(sentences) == 1:
            return []

        window = self.cfg.window
        out: list[RawChunk] = []

        for i, sent in enumerate(sentences):
            lo = max(0, i - window)
            hi = min(len(sentences), i + window + 1)
            neighbours = sentences[lo:hi]
            context = " ".join(s.text for s in neighbours)

            out.append(
                RawChunk(
                    text=sent.text,
                    context_text=context,
                    char_start=sent.start,
                    char_end=sent.end,
                    local_idx=i,
                    # Recorded so the harness can widen context on demand
                    # (the `fetch_neighbours` tool) without re-chunking.
                    extra={"window_start": str(neighbours[0].start),
                           "window_end": str(neighbours[-1].end)},
                )
            )

        return out
