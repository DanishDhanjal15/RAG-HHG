"""Fusion and dedup tests.

RRF is four lines of arithmetic, which is exactly why it deserves tests: a sign
error or an off-by-one in the rank produces plausible-looking rankings that are
quietly wrong, and no downstream metric will point at fusion as the cause.
"""

from __future__ import annotations

from vrag.config import DedupCfg, FusionCfg
from vrag.retrieve.fusion import Candidate, dedup, rrf_fuse
from vrag.schemas import ChunkView, ScoredChunk


def hits(*pairs: tuple[int, float], view: ChunkView = ChunkView.ATOMIC) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk_id=cid, score=score, view=view, rank=i + 1)
        for i, (cid, score) in enumerate(pairs)
    ]


class TestRrfFuse:
    def test_rank_not_magnitude_decides(self):
        """The whole reason for RRF: a run with tiny scores still contributes.

        BM25 scores are unbounded and cosines live in [-1, 1]. If magnitudes
        leaked into the fusion, the sparse run below would be ignored entirely.
        """
        runs = {
            "dense:atomic": hits((1, 0.99), (2, 0.98)),
            "sparse:atomic": hits((2, 0.001), (1, 0.0005)),
        }
        fused = rrf_fuse(runs, FusionCfg(rrf_k=60))
        # Chunk 2 is rank 2 then rank 1; chunk 1 is rank 1 then rank 2. Symmetric,
        # so both get identical scores despite wildly different magnitudes.
        assert len(fused) == 2
        assert abs(fused[0].score - fused[1].score) < 1e-12

    def test_multi_view_agreement_is_rewarded(self):
        """A chunk found by several views outranks one found by a single view.

        This is the actual payoff of running five views instead of one.
        """
        runs = {
            "dense:atomic": hits((1, 0.9), (2, 0.8)),
            "dense:semantic": hits((1, 0.7)),
            "dense:sentence_window": hits((1, 0.6)),
        }
        fused = rrf_fuse(runs, FusionCfg(rrf_k=60))
        assert fused[0].chunk_id == 1
        assert fused[0].views_hit == 3
        assert fused[0].score > fused[1].score

    def test_views_hit_counts_distinct_views_not_runs(self):
        runs = {
            "dense:atomic": hits((1, 0.9)),
            "sparse:atomic": hits((1, 0.5)),
        }
        fused = rrf_fuse(runs, FusionCfg(rrf_k=60))
        assert fused[0].views_hit == 1  # same view, two modalities

    def test_view_weights_apply(self):
        runs = {
            "dense:atomic": hits((1, 0.9)),
            "dense:semantic": hits((2, 0.9), view=ChunkView.SEMANTIC),
        }
        cfg = FusionCfg(rrf_k=60, view_weights={"atomic": 2.0, "semantic": 1.0})
        fused = rrf_fuse(runs, cfg)
        assert fused[0].chunk_id == 1

    def test_zero_weight_excludes_a_view(self):
        runs = {
            "dense:atomic": hits((1, 0.9)),
            "dense:semantic": hits((2, 0.9), view=ChunkView.SEMANTIC),
        }
        cfg = FusionCfg(rrf_k=60, view_weights={"atomic": 1.0, "semantic": 0.0})
        fused = rrf_fuse(runs, cfg)
        assert [c.chunk_id for c in fused] == [1]

    def test_modality_weights_apply(self):
        runs = {
            "dense:atomic": hits((1, 0.9)),
            "sparse:atomic": hits((2, 0.9)),
        }
        fused = rrf_fuse(
            runs, FusionCfg(rrf_k=60), modality_weights={"dense": 1.0, "sparse": 0.5}
        )
        assert fused[0].chunk_id == 1

    def test_smaller_k_sharpens_top_rank_advantage(self):
        runs = {"dense:atomic": hits((1, 0.9), (2, 0.8))}
        sharp = rrf_fuse(runs, FusionCfg(rrf_k=1))
        flat = rrf_fuse(runs, FusionCfg(rrf_k=1000))
        assert (sharp[0].score / sharp[1].score) > (flat[0].score / flat[1].score)

    def test_empty_and_missing_runs(self):
        assert rrf_fuse({}, FusionCfg()) == []
        assert rrf_fuse({"dense:atomic": []}, FusionCfg()) == []

    def test_output_is_sorted_descending(self):
        runs = {"dense:atomic": hits((5, 0.1), (3, 0.9), (7, 0.5))}
        fused = rrf_fuse(runs, FusionCfg())
        assert [c.score for c in fused] == sorted((c.score for c in fused), reverse=True)


class TestDedup:
    @staticmethod
    def candidate(cid: int, doc: str, span: tuple[int, int], score: float) -> Candidate:
        return Candidate(chunk_id=cid, score=score, view=ChunkView.ATOMIC,
                         doc_id=doc, span=span)

    def test_overlapping_spans_collapse_keeping_best(self):
        cands = [
            self.candidate(1, "d1", (0, 100), 0.9),
            self.candidate(2, "d1", (5, 100), 0.5),   # IoU ~0.95
        ]
        kept = dedup(cands, DedupCfg(span_iou_threshold=0.6, max_per_doc=5))
        assert [c.chunk_id for c in kept] == [1]

    def test_distinct_spans_both_survive(self):
        cands = [
            self.candidate(1, "d1", (0, 50), 0.9),
            self.candidate(2, "d1", (60, 110), 0.8),  # no overlap
        ]
        kept = dedup(cands, DedupCfg(span_iou_threshold=0.6, max_per_doc=5))
        assert len(kept) == 2

    def test_same_span_different_docs_both_survive(self):
        cands = [
            self.candidate(1, "d1", (0, 100), 0.9),
            self.candidate(2, "d2", (0, 100), 0.8),
        ]
        kept = dedup(cands, DedupCfg(span_iou_threshold=0.6, max_per_doc=5))
        assert len(kept) == 2

    def test_max_per_doc_prevents_one_document_monopolising_context(self):
        cands = [
            self.candidate(i, "d1", (i * 200, i * 200 + 50), 1.0 - i * 0.1)
            for i in range(5)
        ]
        kept = dedup(cands, DedupCfg(span_iou_threshold=0.6, max_per_doc=2))
        assert len(kept) == 2
        assert [c.chunk_id for c in kept] == [0, 1]  # best-scoring survive

    def test_disabled_is_a_passthrough(self):
        cands = [
            self.candidate(1, "d1", (0, 100), 0.9),
            self.candidate(2, "d1", (0, 100), 0.5),
        ]
        assert len(dedup(cands, DedupCfg(enabled=False))) == 2

    def test_atomic_does_not_swallow_a_single_sentence(self):
        """A sentence chunk inside a long passage has low IoU with the whole
        passage, so both survive -- which is what makes the sentence-window view
        contribute anything at all."""
        cands = [
            self.candidate(1, "d1", (0, 1000), 0.9),   # atomic, whole passage
            self.candidate(2, "d1", (100, 180), 0.85),  # one sentence: IoU 0.08
        ]
        kept = dedup(cands, DedupCfg(span_iou_threshold=0.6, max_per_doc=5))
        assert len(kept) == 2
