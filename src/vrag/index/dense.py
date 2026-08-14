"""Dense vector index (FAISS), in-process.

In-process is the point. A vector database reached over HTTP or gRPC spends
1-5 ms on serialization and loopback before it does any work -- affordable at a
1 s SLA, not at 200 ms where the whole retrieval stage is budgeted at 25 ms.
FAISS in the same address space searches 850k vectors in single-digit
milliseconds with no IPC at all.

Index choice:

* ``hnsw`` + ``sq8`` (default) -- graph search over 8-bit scalar-quantized
  vectors. 850k x 384 goes from 1.3 GB (fp32) to ~326 MB, which is the difference
  between fitting a free-tier container and not. Recall loss versus flat fp32 is
  under a point at ``ef_search=64``.
* ``ivfpq`` -- available for a much larger corpus, where even SQ8 is too big.

All vectors are L2-normalized, so inner product *is* cosine similarity and scores
land in [-1, 1] -- which the domain guardrail depends on, since its threshold is
expressed as a cosine.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from vrag.config import Config


class DenseIndex:
    def __init__(self, index: faiss.Index, cfg: Config) -> None:
        self.index = index
        self.cfg = cfg
        self._apply_search_params()

    # -- construction -------------------------------------------------------- #
    @classmethod
    def build(cls, cfg: Config, vectors: np.ndarray) -> DenseIndex:
        dim = vectors.shape[1]
        dcfg = cfg.dense

        if dcfg.index_type == "hnsw":
            h = dcfg.hnsw
            if h.quantizer == "sq8":
                index = faiss.IndexHNSWSQ(
                    dim, faiss.ScalarQuantizer.QT_8bit, h.m, faiss.METRIC_INNER_PRODUCT
                )
            else:
                index = faiss.IndexHNSWFlat(dim, h.m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = h.ef_construction
        else:
            i = dcfg.ivfpq
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(
                quantizer, dim, i.nlist, i.m, i.nbits, faiss.METRIC_INNER_PRODUCT
            )

        if not index.is_trained:
            # SQ and PQ both need to see the data distribution. A sample is enough
            # and avoids a long single-threaded training pass over everything.
            sample = vectors
            cap = 262_144
            if len(vectors) > cap:
                rng = np.random.default_rng(cfg.corpus.seed)
                sample = vectors[rng.choice(len(vectors), cap, replace=False)]
            index.train(sample)

        index.add(vectors)
        return cls(index, cfg)

    @classmethod
    def load(cls, cfg: Config) -> DenseIndex:
        path = cfg.paths.index_dir / "dense.faiss"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `vrag build` first")
        return cls(faiss.read_index(str(path)), cfg)

    def save(self, path: Path | None = None) -> Path:
        path = path or (self.cfg.paths.index_dir / "dense.faiss")
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        return path

    def _apply_search_params(self) -> None:
        dcfg = self.cfg.dense
        if dcfg.index_type == "hnsw":
            try:
                self.index.hnsw.efSearch = dcfg.hnsw.ef_search
            except AttributeError:
                pass
        else:
            try:
                self.index.nprobe = dcfg.ivfpq.nprobe
            except AttributeError:
                pass

    # -- search -------------------------------------------------------------- #
    def search(
        self, query: np.ndarray, k: int, allowed_ids: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(ids, scores)``, both shape ``(k,)``, ids ``-1`` where empty.

        ``allowed_ids`` filters *inside* the search via an ID selector rather than
        afterwards. Post-filtering would require inflating k enough that the
        surviving hits still number k, which is unpredictable and slow; selector
        filtering keeps k meaning what it says.
        """
        q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))

        params = None
        keepalive: tuple | None = None

        if allowed_ids is not None and len(allowed_ids):
            # `IDSelectorBatch` stores a RAW POINTER into this array and does not
            # copy it, so the array must outlive the search call.
            #
            # The subtle version of getting this wrong -- which this code had, and
            # which crashed the server intermittently -- is
            # `swig_ptr(allowed_ids.astype(np.int64))`. `astype` COPIES by default,
            # so the pointer is taken into a temporary that is freed on the next
            # line: a use-after-free that survives most calls and segfaults
            # whenever the allocator happens to reuse the page. `ascontiguousarray`
            # returns the input unchanged when it is already int64 and contiguous
            # (which the cached filter arrays are), so no temporary is created.
            ids64 = np.ascontiguousarray(allowed_ids, dtype=np.int64)
            selector = faiss.IDSelectorBatch(len(ids64), faiss.swig_ptr(ids64))

            if self.cfg.dense.index_type == "hnsw":
                params = faiss.SearchParametersHNSW()
                params.efSearch = self.cfg.dense.hnsw.ef_search
            else:
                params = faiss.SearchParametersIVF()
                params.nprobe = self.cfg.dense.ivfpq.nprobe
            params.sel = selector

            # Hold the array AND the selector on a local for the duration of the
            # call. A local is enough and is thread-safe; the previous version
            # stashed only the selector on `self`, which both missed the array and
            # would have raced across concurrent requests.
            keepalive = (ids64, selector)

        scores, ids = (
            self.index.search(q, k, params=params)
            if params is not None
            else self.index.search(q, k)
        )
        del keepalive
        return ids[0], scores[0]

    def search_batch(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries.astype(np.float32))
        scores, ids = self.index.search(q, k)
        return ids, scores

    @property
    def size(self) -> int:
        return self.index.ntotal

    def reconstruct(self, chunk_id: int) -> np.ndarray:
        return self.index.reconstruct(int(chunk_id))
