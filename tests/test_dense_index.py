"""Dense index tests, with emphasis on the FAISS memory contract.

``IDSelectorBatch`` stores a **raw pointer** into a numpy array and does not copy
it. Get the lifetime wrong and you get a use-after-free: it survives most calls
and segfaults whenever the allocator happens to reuse the page, which means it
passes a normal test run and then kills the server in front of an audience.

That is not hypothetical -- it happened here. The original code wrote
``swig_ptr(allowed_ids.astype(np.int64))``; ``astype`` copies, so the pointer
referred to a temporary freed on the next line. Sixteen warm-up searches passed,
then the second real request took the process down with no Python traceback.

The stress test below is the one that reproduces it: many filtered searches with
allocation churn in between, so the freed page actually gets reused.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vrag.config import load_config
from vrag.index.dense import DenseIndex


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    rng = np.random.default_rng(11)
    v = rng.normal(size=(4000, 384)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(scope="module", params=["sq8", "flat"])
def index(request, vectors) -> DenseIndex:
    cfg = load_config()
    cfg.dense.hnsw.quantizer = request.param
    return DenseIndex.build(cfg, vectors)


class TestSearch:
    def test_self_match(self, index, vectors):
        ids, scores = index.search(vectors[7], 5)
        assert ids[0] == 7
        assert scores[0] == pytest.approx(1.0, abs=1e-2)

    def test_scores_are_cosines_in_range(self, index, vectors):
        _, scores = index.search(vectors[100], 10)
        assert np.all(scores <= 1.0001)
        assert np.all(scores >= -1.0001)

    def test_scores_descend(self, index, vectors):
        _, scores = index.search(vectors[50], 10)
        valid = scores[scores > -1e30]
        assert np.all(np.diff(valid) <= 1e-6)

    def test_k_larger_than_corpus_is_safe(self, index, vectors):
        ids, _ = index.search(vectors[0], 10_000)
        assert len(ids) == 10_000  # padded with -1, not an error


class TestIdFiltering:
    def test_results_stay_inside_the_allowed_set(self, index, vectors):
        allowed = np.ascontiguousarray(np.arange(100, 300), dtype=np.int64)
        ids, _ = index.search(vectors[7], 10, allowed_ids=allowed)
        returned = [int(i) for i in ids if i >= 0]
        assert returned
        assert all(i in set(allowed.tolist()) for i in returned)

    def test_excluded_self_is_actually_excluded(self, index, vectors):
        """The strongest filter check: the perfect match must NOT come back."""
        allowed = np.ascontiguousarray(np.arange(1000, 1200), dtype=np.int64)
        ids, _ = index.search(vectors[7], 10, allowed_ids=allowed)
        assert 7 not in [int(i) for i in ids]

    def test_empty_filter_is_treated_as_no_filter(self, index, vectors):
        ids, _ = index.search(vectors[7], 5, allowed_ids=np.empty(0, dtype=np.int64))
        assert ids[0] == 7

    def test_repeated_filtered_search_does_not_corrupt_memory(self, index, vectors):
        """Regression: the FAISS id-selector use-after-free.

        Runs many filtered searches with deliberate allocation churn and garbage
        collection between them, so a dangling pointer refers to a page the
        allocator has genuinely handed out again. Under the old code this
        crashed the interpreter; a crash here is the failure, so simply
        completing is the assertion.
        """
        rng = np.random.default_rng(3)
        allowed = np.ascontiguousarray(np.arange(0, 2000, 2), dtype=np.int64)
        allowed_set = set(allowed.tolist())

        for i in range(300):
            ids, _ = index.search(vectors[i % len(vectors)], 8, allowed_ids=allowed)
            assert all(int(x) in allowed_set for x in ids if x >= 0)

            # Churn the allocator so a freed buffer is genuinely reused.
            _ = [rng.normal(size=4096) for _ in range(4)]
            if i % 25 == 0:
                gc.collect()

    def test_filter_arrays_are_not_copied_when_already_int64(self):
        """The fix depends on ascontiguousarray being a no-op for a cached array.

        If this ever starts copying, the pointer would again refer to a temporary
        and the use-after-free would come back silently.
        """
        cached = np.ascontiguousarray(np.arange(50), dtype=np.int64)
        assert np.ascontiguousarray(cached, dtype=np.int64) is cached


class TestPersistence:
    def test_roundtrip(self, index, vectors, tmp_path):
        path = index.save(tmp_path / "x.faiss")
        assert path.stat().st_size > 0

        import faiss

        reloaded = DenseIndex(faiss.read_index(str(path)), index.cfg)
        assert reloaded.size == index.size
        ids_a, _ = index.search(vectors[7], 5)
        ids_b, _ = reloaded.search(vectors[7], 5)
        assert list(ids_a) == list(ids_b)

    def test_sq8_is_substantially_smaller_than_flat(self, vectors, tmp_path):
        cfg = load_config()
        sizes = {}
        for quant in ("sq8", "flat"):
            cfg.dense.hnsw.quantizer = quant
            idx = DenseIndex.build(cfg, vectors)
            sizes[quant] = idx.save(tmp_path / f"{quant}.faiss").stat().st_size
        # The whole reason SQ8 is the default: it is what makes the index fit
        # free-tier hosting.
        assert sizes["sq8"] < sizes["flat"] * 0.6
