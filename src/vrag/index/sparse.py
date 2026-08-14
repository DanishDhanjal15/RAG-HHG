"""Sparse lexical index (BM25).

Dense retrieval on a multilingual model has a specific, reproducible failure mode:
rare surface forms. Proper nouns, product codes, years, transliterated names --
tokens the encoder has barely seen -- get embedded near their neighbourhood rather
than themselves. BM25 does not care how rare a token is; rarity is precisely what
it rewards. Fusing the two covers each other's blind spot, and on MS MARCO the
hybrid reliably beats either alone.

``bm25s`` rather than ``rank_bm25``: ``rank_bm25`` scores in a Python loop over
the corpus, which is ~2 s per query at 850k documents -- ten times the entire
latency budget. ``bm25s`` precomputes a scipy sparse score matrix and answers in
single-digit milliseconds.

Tokenization is deliberately script-agnostic: no stemmer and no stopword list,
because a stemmer trained on English would mangle Devanagari, Tamil and Bengali
tokens, and the corpus is majority non-English.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from vrag.config import Config

# Unicode-aware token pattern: keep letters, marks (combining vowel signs matter
# enormously in Indic scripts) and digits. \w under `re.UNICODE` already covers
# these; we exclude underscore and split on everything else.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def tokenize_corpus(texts: list[str]) -> list[list[str]]:
    return [_TOKEN_RE.findall(t.lower()) for t in texts]


class SparseIndex:
    def __init__(self, retriever) -> None:  # noqa: ANN001 -- bm25s.BM25
        self.retriever = retriever

    @classmethod
    def build(cls, cfg: Config, texts: list[str]) -> SparseIndex:
        import bm25s

        retriever = bm25s.BM25(k1=cfg.sparse.k1, b=cfg.sparse.b, method=cfg.sparse.method)
        retriever.index(tokenize_corpus(texts))
        return cls(retriever)

    @classmethod
    def load(cls, cfg: Config) -> SparseIndex:
        import bm25s

        path = cfg.paths.index_dir / "sparse"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `vrag build` first")
        return cls(bm25s.BM25.load(str(path), mmap=True))

    def save(self, cfg: Config, path: Path | None = None) -> Path:
        path = path or (cfg.paths.index_dir / "sparse")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(path))
        return path

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(ids, scores)``. Empty arrays when the query has no indexable
        tokens (pure punctuation, or a script the tokenizer produced nothing for)."""
        tokens = tokenize(query)
        if not tokens:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        ids, scores = self.retriever.retrieve([tokens], k=k, show_progress=False)
        return ids[0].astype(np.int64), scores[0].astype(np.float32)
