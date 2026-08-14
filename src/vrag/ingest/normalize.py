"""Turn raw MSMARCO-XI parquet rows into a flat corpus of passages + queries.

Output (both parquet, in ``paths.corpus_dir``):

* ``passages.parquet`` -- one row per retrievable passage, in one language.
* ``queries.parquet``  -- one row per query, with gold ``doc_id``s taken from the
  dataset's ``is_selected`` flags. These gold labels are what make the chunking
  ablation and threshold calibration real measurements rather than vibes.

Row selection: each language takes a *shared head block* (identical query ids
across languages, so one spoken question can be asked in three languages and hit
parallel content) plus a *disjoint tail block* (so the corpus stays topically
diverse instead of being the same 8.5k questions three times).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from vrag.config import Config
from vrag.schemas import Passage, QueryRecord

# Columns we actually need. Reading a subset keeps a 460 MB shard from
# materializing translation-metadata blobs we never look at.
WANTED_COLUMNS = [
    "query_id",
    "query",
    "Answer",
    "query_type",
    "Eng_Query",
    "Eng_Answer",
    "source_lang",
    "target_lang",
    "passages",
]

_WS = re.compile(r"\s+")


def _clean(text: Any) -> str:
    if text is None:
        return ""
    return _WS.sub(" ", str(text)).strip()


# --------------------------------------------------------------------------- #
# Row-range planning
# --------------------------------------------------------------------------- #
@dataclass
class RowPlan:
    """Which row indices of a shard this language should take."""

    lang: str
    ranges: list[tuple[int, int]] = field(default_factory=list)  # [start, end)

    def contains(self, idx: int) -> bool:
        return any(start <= idx < end for start, end in self.ranges)

    @property
    def max_row(self) -> int:
        return max((end for _, end in self.ranges), default=0)

    @property
    def total(self) -> int:
        return sum(end - start for start, end in self.ranges)


def plan_rows(cfg: Config) -> dict[str, RowPlan]:
    shared = max(0, cfg.corpus.shared_queries)
    per_lang = cfg.corpus.queries_per_language
    disjoint = max(0, per_lang - shared)

    plans: dict[str, RowPlan] = {}
    for i, lang in enumerate(cfg.corpus.languages):
        ranges: list[tuple[int, int]] = []
        if shared:
            ranges.append((0, shared))
        if disjoint:
            start = shared + i * disjoint
            ranges.append((start, start + disjoint))
        plans[lang] = RowPlan(lang=lang, ranges=ranges)
    return plans


# --------------------------------------------------------------------------- #
# Parquet reading
# --------------------------------------------------------------------------- #
def _available_columns(pf: pq.ParquetFile) -> list[str]:
    present = set(pf.schema_arrow.names)
    return [c for c in WANTED_COLUMNS if c in present]


def iter_selected_rows(
    path: Path, plan: RowPlan, batch_size: int = 512
) -> Iterator[dict[str, Any]]:
    """Yield the rows this language's plan asks for, in shard order.

    Stops as soon as the highest planned row index is passed, so we never read a
    whole 460 MB shard to collect the first 8.5k rows.
    """
    pf = pq.ParquetFile(path)
    columns = _available_columns(pf)
    offset = 0
    stop_at = plan.max_row

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        n = batch.num_rows
        if offset >= stop_at:
            break
        # Skip whole batches that fall outside every planned range.
        if not any(plan.contains(i) for i in range(offset, offset + n)):
            offset += n
            continue
        rows = batch.to_pylist()
        for i, row in enumerate(rows):
            if plan.contains(offset + i):
                yield row
        offset += n


# --------------------------------------------------------------------------- #
# Passage extraction
# --------------------------------------------------------------------------- #
def extract_passages(raw: Any) -> list[dict[str, Any]]:
    """Normalize the ``passages`` column into a list of per-passage dicts.

    The column is documented as a struct-of-lists
    (``{is_selected: [...], English_passages: [...], Translated_passages: [...]}``)
    but some HF exports flip this to a list-of-structs. Both are handled, because
    the dataset viewer is broken and we cannot confirm the on-disk layout without
    reading it -- and a shape assumption that fails at row 40,000 of an ingest is
    an expensive way to find out.
    """
    if raw is None:
        return []

    # struct-of-lists
    if isinstance(raw, dict):
        translated = raw.get("Translated_passages") or raw.get("translated_passages") or []
        english = raw.get("English_passages") or raw.get("english_passages") or []
        selected = raw.get("is_selected") or []
        n = max(len(translated), len(english))
        out = []
        for i in range(n):
            out.append(
                {
                    "translated": translated[i] if i < len(translated) else None,
                    "english": english[i] if i < len(english) else None,
                    "is_selected": bool(selected[i]) if i < len(selected) else False,
                }
            )
        return out

    # list-of-structs
    if isinstance(raw, list):
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "translated": item.get("Translated_passages")
                    or item.get("translated_passages")
                    or item.get("passage_text"),
                    "english": item.get("English_passages") or item.get("english_passages"),
                    "is_selected": bool(item.get("is_selected") or 0),
                }
            )
        return out

    return []


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
@dataclass
class CorpusStats:
    queries: int = 0
    passages: int = 0
    english_passages: int = 0
    per_lang: dict[str, int] = field(default_factory=dict)
    skipped_short: int = 0
    rows_without_passages: int = 0


def normalize_shard(
    cfg: Config,
    lang: str,
    path: Path,
    plan: RowPlan,
    seen_english: set[str],
    stats: CorpusStats,
) -> tuple[list[Passage], list[QueryRecord]]:
    passages: list[Passage] = []
    queries: list[QueryRecord] = []
    min_chars = cfg.corpus.min_passage_chars
    max_per_query = cfg.corpus.max_passages_per_query

    for row in iter_selected_rows(path, plan):
        query_id = int(row.get("query_id") or 0)
        query_text = _clean(row.get("query"))
        if not query_text:
            continue

        items = extract_passages(row.get("passages"))
        if not items:
            stats.rows_without_passages += 1
            continue

        gold: list[str] = []
        kept = 0

        for idx, item in enumerate(items):
            if kept >= max_per_query:
                break

            translated = _clean(item["translated"])
            english = _clean(item["english"])
            is_sel = item["is_selected"]

            en_doc_id = f"en:{query_id}:{idx}"
            has_en = bool(english) and len(english) >= min_chars

            if translated and len(translated) >= min_chars:
                doc_id = f"{lang}:{query_id}:{idx}"
                passages.append(
                    Passage(
                        doc_id=doc_id,
                        query_id=query_id,
                        passage_idx=idx,
                        lang=lang,
                        text=translated,
                        is_selected=is_sel,
                        parallel_en_id=en_doc_id if has_en else None,
                    )
                )
                if is_sel:
                    gold.append(doc_id)
                kept += 1
            elif translated:
                stats.skipped_short += 1

            # English parallel passages are shared across language shards, so
            # dedup globally -- otherwise the same English text is indexed 3x and
            # RRF quietly triple-counts it.
            if cfg.corpus.include_english_parallel and has_en and en_doc_id not in seen_english:
                seen_english.add(en_doc_id)
                passages.append(
                    Passage(
                        doc_id=en_doc_id,
                        query_id=query_id,
                        passage_idx=idx,
                        lang="en",
                        text=english,
                        is_selected=is_sel,
                        parallel_en_id=None,
                    )
                )
                stats.english_passages += 1
                if is_sel:
                    gold.append(en_doc_id)

        if kept == 0:
            continue

        queries.append(
            QueryRecord(
                query_id=query_id,
                lang=lang,
                query=query_text,
                answer=_clean(row.get("Answer")),
                eng_query=_clean(row.get("Eng_Query")),
                eng_answer=_clean(row.get("Eng_Answer")),
                query_type=_clean(row.get("query_type")),
                gold_doc_ids=gold,
            )
        )

    stats.per_lang[lang] = len(passages)
    return passages, queries


PASSAGE_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("query_id", pa.int64()),
        ("passage_idx", pa.int32()),
        ("lang", pa.string()),
        ("text", pa.string()),
        ("is_selected", pa.bool_()),
        ("parallel_en_id", pa.string()),
    ]
)

QUERY_SCHEMA = pa.schema(
    [
        ("query_id", pa.int64()),
        ("lang", pa.string()),
        ("query", pa.string()),
        ("answer", pa.string()),
        ("eng_query", pa.string()),
        ("eng_answer", pa.string()),
        ("query_type", pa.string()),
        ("gold_doc_ids", pa.list_(pa.string())),
    ]
)


def write_corpus(
    cfg: Config, passages: list[Passage], queries: list[QueryRecord]
) -> tuple[Path, Path]:
    cfg.paths.corpus_dir.mkdir(parents=True, exist_ok=True)

    ptable = pa.Table.from_pylist([p.model_dump() for p in passages], schema=PASSAGE_SCHEMA)
    qtable = pa.Table.from_pylist([q.model_dump() for q in queries], schema=QUERY_SCHEMA)

    ppath = cfg.paths.corpus_dir / "passages.parquet"
    qpath = cfg.paths.corpus_dir / "queries.parquet"
    pq.write_table(ptable, ppath, compression="zstd")
    pq.write_table(qtable, qpath, compression="zstd")
    return ppath, qpath


def load_passages(cfg: Config) -> list[Passage]:
    table = pq.read_table(cfg.paths.corpus_dir / "passages.parquet")
    return [Passage(**row) for row in table.to_pylist()]


def load_queries(cfg: Config) -> list[QueryRecord]:
    table = pq.read_table(cfg.paths.corpus_dir / "queries.parquet")
    return [QueryRecord(**row) for row in table.to_pylist()]
