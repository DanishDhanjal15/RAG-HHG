"""View 5 -- proposition chunking: atomic, self-contained facts.

A passage that says "It was founded in 1911 and moved to Detroit in 1925" carries
two facts, neither of which is retrievable on its own and both of which are
pronoun-bound to a subject stated earlier. Proposition chunking rewrites text into
standalone declarative statements ("Chevrolet was founded in 1911.", "Chevrolet
moved to Detroit in 1925."), each of which embeds cleanly.

This is the most precise view and the most expensive to build, so it is:

* **offline only** -- an LLM pass run once by ``vrag propositions``, never at
  query time;
* **cached to disk** keyed by passage hash, so re-running skips finished work and
  a crash costs minutes rather than the whole pass;
* **restricted** to the highest-value passages (those the dataset marks
  ``is_selected``, i.e. the ones that actually answer a query), because
  propositionalising 224k passages buys far less than propositionalising the 25k
  that carry the answers.

Disabled by default in ``configs/default.yaml``; the ablation reports the view
with and without it so its cost is visible rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from vrag.chunking.base import RawChunk
from vrag.config import PropositionCfg
from vrag.schemas import ChunkView, Passage

SYSTEM_PROMPT = """\
You decompose a passage into atomic propositions for a retrieval index.

Rules:
- Each proposition states exactly one fact.
- Each proposition must be understandable with NO other context: resolve every
  pronoun and every elliptical reference to the explicit entity named in the passage.
- Use only information present in the passage. Never add, infer, or generalise.
- Write each proposition in the SAME language as the passage.
- Omit propositions that carry no retrievable content (greetings, navigation text).

Return JSON only: {"propositions": ["...", "..."]}"""


def _cache_key(passage: Passage) -> str:
    return hashlib.sha1(f"{passage.doc_id}|{passage.text}".encode()).hexdigest()[:16]


class PropositionCache:
    """Append-only JSONL cache, loaded into memory once.

    JSONL rather than a database because the write pattern is append-only and the
    read pattern is load-everything -- and because a half-written JSONL loses one
    line, not the file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, list[str]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.data[row["key"]] = row["propositions"]

    def get(self, key: str) -> list[str] | None:
        return self.data.get(key)

    def put(self, key: str, propositions: list[str]) -> None:
        self.data[key] = propositions
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "propositions": propositions},
                                ensure_ascii=False) + "\n")


class PropositionChunker:
    """Serves propositions from cache. Extraction is a separate offline command."""

    view = ChunkView.PROPOSITION

    def __init__(self, cfg: PropositionCfg, cache: PropositionCache) -> None:
        self.cfg = cfg
        self.cache = cache

    def chunk(self, passage: Passage) -> list[RawChunk]:
        propositions = self.cache.get(_cache_key(passage))
        if not propositions:
            return []

        out: list[RawChunk] = []
        for idx, prop in enumerate(propositions[: self.cfg.max_props_per_passage]):
            prop = prop.strip()
            if len(prop) < 15:
                continue
            # Propositions are rewrites, so they have no faithful char span in the
            # source. They point at the whole passage: dedup then treats them as
            # covering it, which is correct -- a proposition IS the passage's content.
            out.append(
                RawChunk(
                    text=prop,
                    context_text=passage.text,
                    char_start=0,
                    char_end=len(passage.text),
                    local_idx=idx,
                    extra={"synthetic": "1"},
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Offline extraction
# --------------------------------------------------------------------------- #
async def extract_propositions(
    passages: list[Passage],
    cache: PropositionCache,
    api_key: str,
    model: str = "claude-haiku-4-5",
    concurrency: int = 8,
    max_props: int = 6,
) -> int:
    """Fill the cache for ``passages``. Returns how many were newly extracted.

    Failures are swallowed per passage: one bad response must not abort a pass
    over tens of thousands of documents. The view degrades to "fewer propositions",
    which the ablation will show, rather than to a crash.
    """
    import anyio
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key)
    todo = [p for p in passages if cache.get(_cache_key(p)) is None]
    if not todo:
        return 0

    limiter = anyio.CapacityLimiter(concurrency)
    extracted = 0

    async def one(passage: Passage) -> None:
        nonlocal extracted
        async with limiter:
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=600,
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": passage.text}],
                )
                text = "".join(b.text for b in resp.content if b.type == "text")
                start, end = text.find("{"), text.rfind("}")
                if start < 0 or end < 0:
                    return
                props = json.loads(text[start : end + 1]).get("propositions", [])
                props = [str(p) for p in props][:max_props]
                cache.put(_cache_key(passage), props)
                extracted += 1
            except Exception:  # noqa: BLE001 -- per-item failure is expected and tolerable
                return

    async with anyio.create_task_group() as tg:
        for passage in todo:
            tg.start_soon(one, passage)

    return extracted


def default_cache_path(model_dir: Path) -> Path:
    return Path(os.environ.get("VRAG_PROPOSITION_CACHE", model_dir / "propositions.jsonl"))
