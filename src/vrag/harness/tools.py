"""Tool registry for the LLM generation path.

The optional polish path does not receive a prompt with passages pasted into it.
It receives *tools* and decides what to look at. That distinction is the whole
point of the harness requirement, and it buys three concrete things:

* **The model can recover from bad retrieval.** If the first shortlist is thin, it
  can search again with different terms rather than writing an answer from
  whatever it was handed.
* **Context stays small.** It pulls the neighbours of one chunk instead of being
  given every chunk's neighbours pre-emptively -- fewer tokens, lower latency,
  less distraction.
* **Every action is observable.** Tool calls are spans. When an answer is wrong,
  the trace shows what the model looked at before writing it.

The loop is bounded on both axes -- ``max_tool_rounds`` and a hard wall-clock
timeout -- because an unbounded agent loop behind a voice interface is a hang.
Tool results are typed and validated; a malformed call gets one structured repair
attempt and then the request falls back to the extractive answer, which is already
computed and already grounded.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vrag.retrieve.expand import normalize_query, plan_text


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Builds the tool set bound to a live pipeline and dispatches calls."""

    def __init__(self, pipeline) -> None:  # noqa: ANN001 -- avoids a circular import
        self.pipeline = pipeline
        self._tools: dict[str, Tool] = {}
        self._register_all()

    # -- registration -------------------------------------------------------- #
    def _register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def _register_all(self) -> None:
        self._register(
            Tool(
                name="search_corpus",
                description=(
                    "Search the indexed MSMARCO-XI corpus (Hindi, Tamil, Bengali and "
                    "English passages). Use this when the passages you already have "
                    "do not contain the answer, or to check a specific claim with "
                    "different search terms. Returns chunks with ids you can cite."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search terms. Rephrasing helps when the first attempt was thin.",
                        },
                        "lang": {
                            "type": "string",
                            "enum": ["hi", "ta", "bn", "en", "any"],
                            "description": "Restrict to one language, or 'any'.",
                        },
                        "k": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
                handler=self._search_corpus,
            )
        )

        self._register(
            Tool(
                name="fetch_neighbours",
                description=(
                    "Get the passage text immediately before and after a chunk. Use "
                    "this when a chunk looks relevant but is cut off, or when it "
                    "starts with a pronoun whose referent you cannot see."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"chunk_id": {"type": "integer"}},
                    "required": ["chunk_id"],
                },
                handler=self._fetch_neighbours,
            )
        )

        self._register(
            Tool(
                name="normalize_query",
                description=(
                    "Clean a spoken transcript: strip disfluencies and leading "
                    "politeness. Useful when the user's question came through ASR "
                    "with noise in it."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                handler=self._normalize_query,
            )
        )

    # -- handlers ------------------------------------------------------------ #
    def _search_corpus(self, query: str, lang: str = "any", k: int = 5) -> dict[str, Any]:
        cfg = self.pipeline.cfg
        plan = plan_text(cfg, query, lang="unknown", confidence=1.0)
        if lang != "any":
            plan.lang_filter = [lang] if lang == "en" else [lang, "en"]

        vector = self.pipeline.retriever.embed_query(plan)
        runs = self.pipeline.retriever.dense_search(vector, plan)
        if self.pipeline.sparse is not None:
            runs.update(self.pipeline.retriever.sparse_search(plan))
        candidates = self.pipeline.retriever.fuse(runs)
        evidence = self.pipeline.retriever.hydrate(candidates, limit=min(k, 10))

        return {
            "results": [
                {
                    "chunk_id": e.chunk_id,
                    "doc_id": e.doc_id,
                    "lang": e.lang,
                    "view": e.view.value,
                    "text": e.text[:600],
                    "score": round(e.score, 4),
                }
                for e in evidence
            ]
        }

    def _fetch_neighbours(self, chunk_id: int) -> dict[str, Any]:
        try:
            neighbours = self.pipeline.retriever.neighbours(int(chunk_id))
        except (IndexError, ValueError):
            return {"error": f"chunk_id {chunk_id} is not in the index", "results": []}
        return {
            "results": [
                {"chunk_id": e.chunk_id, "doc_id": e.doc_id, "lang": e.lang, "text": e.text[:600]}
                for e in neighbours
            ]
        }

    @staticmethod
    def _normalize_query(text: str) -> dict[str, str]:
        return {"normalized": normalize_query(text)}

    # -- dispatch ------------------------------------------------------------ #
    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool call and return a JSON string for the model.

        Errors are returned *to the model* as structured results rather than
        raised. A model that called a tool wrongly can correct itself if it is
        told how it was wrong; an exception here would abort a request that was
        one retry away from succeeding.
        """
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"unknown tool {name!r}",
                               "available": list(self._tools)})
        try:
            return json.dumps(tool.handler(**arguments), ensure_ascii=False)
        except TypeError as exc:
            return json.dumps({"error": f"bad arguments for {name}: {exc}",
                               "expected_schema": tool.input_schema})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
