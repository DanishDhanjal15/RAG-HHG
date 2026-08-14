"""Chunk payload store -- memory-mapped, O(1) lookup, zero parsing.

Retrieval returns chunk ids; something has to turn ~40 of those into text and
metadata inside a few milliseconds, on every single request. A row-oriented
format (JSONL, SQLite, pickle-per-record) spends that budget on deserialization.

So the store is columnar and mmapped:

* ``texts.bin``   -- every chunk's *display* text concatenated as UTF-8
* ``offsets.npy`` -- ``int64[n+1]``; text *i* is ``bytes[offsets[i]:offsets[i+1]]``
* ``embed.bin`` / ``embed_offsets.npy`` -- the *embedded* text, which differs from
  the display text for the sentence-window view (embed one sentence, display three)
  and whenever contextual prefixing is on. BM25 indexes this one too: indexing the
  window instead would make every sibling chunk lexically identical.
* ``meta.npz``    -- fixed-width columns (query_id, lang code, view code, spans, ...)

``doc_id`` is not stored: it is ``f"{lang}:{query_id}:{passage_idx}"`` and
reconstructing it from three small integer columns costs less than storing and
loading 850k strings.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vrag.schemas import Chunk, ChunkView

VIEW_ORDER = [
    ChunkView.ATOMIC,
    ChunkView.SENTENCE_WINDOW,
    ChunkView.FIXED_OVERLAP,
    ChunkView.SEMANTIC,
    ChunkView.PROPOSITION,
]
VIEW_TO_CODE = {v: i for i, v in enumerate(VIEW_ORDER)}


@dataclass(slots=True)
class ChunkPayload:
    chunk_id: int
    doc_id: str
    query_id: int
    lang: str
    view: ChunkView
    text: str
    char_start: int
    char_end: int
    is_selected: bool
    passage_idx: int


class ChunkStoreWriter:
    """Streaming writer -- never holds the whole corpus in memory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self._fh = (path / "texts.bin").open("wb")
        self._offsets: list[int] = [0]
        self._cursor = 0

        self._efh = (path / "embed.bin").open("wb")
        self._eoffsets: list[int] = [0]
        self._ecursor = 0

        self._query_id: list[int] = []
        self._lang: list[int] = []
        self._view: list[int] = []
        self._char_start: list[int] = []
        self._char_end: list[int] = []
        self._is_selected: list[bool] = []
        self._passage_idx: list[int] = []
        self._neighbours: list[tuple[int, int]] = []

        self._lang_vocab: dict[str, int] = {}

    def _lang_code(self, lang: str) -> int:
        if lang not in self._lang_vocab:
            self._lang_vocab[lang] = len(self._lang_vocab)
        return self._lang_vocab[lang]

    def add(self, chunk: Chunk) -> None:
        blob = chunk.context_text.encode("utf-8")
        self._fh.write(blob)
        self._cursor += len(blob)
        self._offsets.append(self._cursor)

        eblob = chunk.text.encode("utf-8")
        self._efh.write(eblob)
        self._ecursor += len(eblob)
        self._eoffsets.append(self._ecursor)

        self._query_id.append(chunk.query_id)
        self._lang.append(self._lang_code(chunk.lang))
        self._view.append(VIEW_TO_CODE[chunk.view])
        self._char_start.append(chunk.char_start)
        self._char_end.append(chunk.char_end)
        self._is_selected.append(chunk.is_selected)
        self._passage_idx.append(int(chunk.doc_id.rsplit(":", 1)[-1]))

        left = chunk.neighbour_ids[0] if len(chunk.neighbour_ids) > 0 else -1
        right = chunk.neighbour_ids[1] if len(chunk.neighbour_ids) > 1 else -1
        self._neighbours.append((left, right))

    def close(self) -> None:
        self._fh.close()
        self._efh.close()
        np.save(self.path / "offsets.npy", np.asarray(self._offsets, dtype=np.int64))
        np.save(self.path / "embed_offsets.npy", np.asarray(self._eoffsets, dtype=np.int64))
        np.savez(
            self.path / "meta.npz",
            query_id=np.asarray(self._query_id, dtype=np.int64),
            lang=np.asarray(self._lang, dtype=np.int8),
            view=np.asarray(self._view, dtype=np.int8),
            char_start=np.asarray(self._char_start, dtype=np.int32),
            char_end=np.asarray(self._char_end, dtype=np.int32),
            is_selected=np.asarray(self._is_selected, dtype=bool),
            passage_idx=np.asarray(self._passage_idx, dtype=np.int16),
            neighbours=np.asarray(self._neighbours, dtype=np.int32),
        )
        (self.path / "vocab.json").write_text(
            json.dumps({"lang": self._lang_vocab}, ensure_ascii=False), encoding="utf-8"
        )

    def __enter__(self) -> ChunkStoreWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ChunkStore:
    """Read side. Everything is mmapped, so opening is instant and the OS page
    cache -- not the Python heap -- holds the working set."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._texts = np.memmap(path / "texts.bin", dtype=np.uint8, mode="r")
        self.offsets = np.load(path / "offsets.npy", mmap_mode="r")

        self._embed_texts = np.memmap(path / "embed.bin", dtype=np.uint8, mode="r")
        self.embed_offsets = np.load(path / "embed_offsets.npy", mmap_mode="r")

        meta = np.load(path / "meta.npz", mmap_mode="r")
        self.query_id = meta["query_id"]
        self.lang_code = meta["lang"]
        self.view_code = meta["view"]
        self.char_start = meta["char_start"]
        self.char_end = meta["char_end"]
        self.is_selected = meta["is_selected"]
        self.passage_idx = meta["passage_idx"]
        self.neighbours = meta["neighbours"]

        vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        self._lang_names = [""] * len(vocab["lang"])
        for name, code in vocab["lang"].items():
            self._lang_names[code] = name

        self.size = len(self.offsets) - 1

    def __len__(self) -> int:
        return self.size

    def text(self, chunk_id: int) -> str:
        lo, hi = int(self.offsets[chunk_id]), int(self.offsets[chunk_id + 1])
        return bytes(self._texts[lo:hi]).decode("utf-8", errors="replace")

    def texts(self, chunk_ids: Iterable[int]) -> list[str]:
        return [self.text(i) for i in chunk_ids]

    def embed_text(self, chunk_id: int) -> str:
        lo, hi = int(self.embed_offsets[chunk_id]), int(self.embed_offsets[chunk_id + 1])
        return bytes(self._embed_texts[lo:hi]).decode("utf-8", errors="replace")

    def embed_texts(self, chunk_ids: Iterable[int]) -> list[str]:
        return [self.embed_text(i) for i in chunk_ids]

    def iter_embed_texts(self, start: int = 0, end: int | None = None) -> list[str]:
        """Bulk slice for index building. Decoding one large byte range and
        splitting is markedly faster than 850k individual slice-and-decode calls."""
        end = self.size if end is None else min(end, self.size)
        if start >= end:
            return []
        lo, hi = int(self.embed_offsets[start]), int(self.embed_offsets[end])
        blob = bytes(self._embed_texts[lo:hi])
        bounds = np.asarray(self.embed_offsets[start : end + 1]) - lo
        return [
            blob[bounds[i] : bounds[i + 1]].decode("utf-8", errors="replace")
            for i in range(end - start)
        ]

    def lang(self, chunk_id: int) -> str:
        return self._lang_names[int(self.lang_code[chunk_id])]

    def view(self, chunk_id: int) -> ChunkView:
        return VIEW_ORDER[int(self.view_code[chunk_id])]

    def doc_id(self, chunk_id: int) -> str:
        return (
            f"{self._lang_names[int(self.lang_code[chunk_id])]}"
            f":{int(self.query_id[chunk_id])}"
            f":{int(self.passage_idx[chunk_id])}"
        )

    def payload(self, chunk_id: int) -> ChunkPayload:
        return ChunkPayload(
            chunk_id=chunk_id,
            doc_id=self.doc_id(chunk_id),
            query_id=int(self.query_id[chunk_id]),
            lang=self.lang(chunk_id),
            view=self.view(chunk_id),
            text=self.text(chunk_id),
            char_start=int(self.char_start[chunk_id]),
            char_end=int(self.char_end[chunk_id]),
            is_selected=bool(self.is_selected[chunk_id]),
            passage_idx=int(self.passage_idx[chunk_id]),
        )

    def neighbour_ids(self, chunk_id: int) -> list[int]:
        return [int(i) for i in self.neighbours[chunk_id] if i >= 0]

    def view_mask(self, views: list[ChunkView]) -> np.ndarray:
        """Boolean mask over all chunks for a view restriction.

        Used by the ablation so a single-view arm gets its own full top-k from the
        index rather than whatever survives a post-hoc filter of a mixed result
        list -- otherwise a view that ranks 5th everywhere would look empty and
        the comparison would measure the filter, not the view.
        """
        codes = [VIEW_TO_CODE[v] for v in views if v in VIEW_TO_CODE]
        if not codes or len(codes) == len(VIEW_ORDER):
            return np.ones(self.size, dtype=bool)
        mask = np.zeros(self.size, dtype=bool)
        for code in codes:
            mask |= self.view_code == code
        return mask

    @property
    def languages(self) -> list[str]:
        return list(self._lang_names)

    def lang_mask(self, langs: list[str]) -> np.ndarray:
        """Boolean mask over all chunks for a language filter.

        Used to build a FAISS ``IDSelector`` so the language restriction happens
        inside the search rather than as a post-filter that would need a much
        larger k to survive.
        """
        codes = [
            self._lang_names.index(lang) for lang in langs if lang in self._lang_names
        ]
        if not codes:
            return np.ones(self.size, dtype=bool)
        mask = np.zeros(self.size, dtype=bool)
        for code in codes:
            mask |= self.lang_code == code
        return mask
