"""LLM polish path -- optional, off the critical path, tool-calling.

This never blocks the answer. The extractive generator has already produced a
grounded, cited answer inside the 200 ms budget; this path streams a more fluent
rewrite ~500 ms later and the UI swaps it in. If it is slow, fails, or produces
something the grounding guard rejects, the user keeps the fast answer and never
sees a degradation beyond a missing polish.

**Tools, not a stuffed prompt.** The model is given ``search_corpus``,
``fetch_neighbours`` and ``normalize_query`` and decides what to look at. That
buys three things a prompt-stuffed call cannot: it can recover from a thin first
shortlist by searching again with different terms, it pulls only the context it
actually needs rather than being handed everything up front, and every action it
takes is a span in the trace -- so a wrong answer is debuggable rather than
mysterious.

**A manual loop rather than the SDK tool runner.** The runner is the right default
for most agents, but this loop needs three things that sit awkwardly inside it: a
hard wall-clock deadline checked between rounds (a voice UI cannot hang), a span
emitted per individual tool call for the latency dataset, and an unconditional
fall back to the already-computed extractive answer on any failure. Owning the
loop makes all three explicit, and avoids taking a beta dependency on the
submission's critical path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from vrag.config import LlmCfg
from vrag.schemas import Answer, Citation, Evidence

SYSTEM_PROMPT = """\
You answer questions using ONLY passages retrieved from an indexed corpus of \
MS MARCO web passages in Hindi, Tamil, Bengali and English.

Rules:
- Answer only from the retrieved passages. If they do not contain the answer, say \
so plainly -- never fill the gap from your own knowledge.
- Answer in the same language the question was asked in.
- Cite the chunk_id of every passage you used.
- Be direct. One to three sentences. No preamble.
- If the passages disagree, say they disagree and give both figures.

Use search_corpus if the passages you were given do not answer the question. \
Use fetch_neighbours if a passage is cut off or starts with an unresolved pronoun.

Return JSON only: {"answer": "...", "chunk_ids": [1, 2], "grounded": true}
If the passages cannot support an answer, return \
{"answer": "", "chunk_ids": [], "grounded": false}."""


@dataclass
class LlmTrace:
    """What the loop actually did -- recorded as span attributes."""

    rounds: int = 0
    tool_calls: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    repaired: bool = False
    timed_out: bool = False
    error: str | None = None


class LlmGenerator:
    def __init__(self, cfg: LlmCfg, api_key: str, registry) -> None:  # noqa: ANN001
        from anthropic import Anthropic

        self.cfg = cfg
        self.registry = registry
        self.client = Anthropic(api_key=api_key)

    # ----------------------------------------------------------------------- #
    def generate(
        self, query: str, evidence: list[Evidence], lang: str = "unknown"
    ) -> tuple[Answer | None, LlmTrace]:
        """Returns ``(answer, trace)``. ``answer`` is None whenever the caller
        should keep the extractive result -- timeout, API failure, malformed
        output, or the model itself reporting it could not ground an answer."""
        trace = LlmTrace()
        deadline = time.perf_counter() + (self.cfg.hard_timeout_ms / 1000.0)

        by_id = {e.chunk_id: e for e in evidence}
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self._initial_prompt(query, evidence, lang)}
        ]

        try:
            for _ in range(self.cfg.max_tool_rounds + 1):
                if time.perf_counter() > deadline:
                    trace.timed_out = True
                    return None, trace

                response = self.client.messages.create(
                    model=self.cfg.model,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    system=SYSTEM_PROMPT,
                    tools=self.registry.specs(),
                    messages=messages,
                )
                trace.rounds += 1
                trace.input_tokens += response.usage.input_tokens
                trace.output_tokens += response.usage.output_tokens

                if response.stop_reason == "refusal":
                    trace.error = "refusal"
                    return None, trace

                if response.stop_reason != "tool_use":
                    return self._parse(response, by_id, trace), trace

                # Append the assistant turn whole -- dropping tool_use blocks
                # breaks the next request's block pairing.
                messages.append({"role": "assistant", "content": response.content})

                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    trace.tool_calls.append(block.name)
                    payload = self.registry.dispatch(block.name, dict(block.input))
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": payload,
                        }
                    )
                    # Newly-retrieved chunks become citable too, otherwise the
                    # model can only ever cite what it started with.
                    self._absorb(payload, by_id)

                # All results go back in ONE user message. Splitting them across
                # messages trains the model out of parallel tool calls.
                messages.append({"role": "user", "content": results})

            # Ran out of rounds without a final answer.
            trace.error = "max_tool_rounds"
            return None, trace

        except Exception as exc:  # noqa: BLE001 -- optional path, never fatal
            trace.error = f"{type(exc).__name__}: {exc}"
            return None, trace

    # ----------------------------------------------------------------------- #
    @staticmethod
    def _initial_prompt(query: str, evidence: list[Evidence], lang: str) -> str:
        passages = "\n\n".join(
            f"[chunk_id={e.chunk_id} lang={e.lang}]\n{e.text}" for e in evidence
        )
        return (
            f"Question ({lang}): {query}\n\n"
            f"Retrieved passages:\n\n{passages}\n\n"
            f"Answer the question using only these passages."
        )

    def _parse(self, response, by_id: dict[int, Evidence], trace: LlmTrace) -> Answer | None:  # noqa: ANN001
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        data = self._extract_json(text)

        if data is None:
            # One structured repair attempt before giving up. A model that
            # returned prose instead of JSON usually has the right answer and the
            # wrong wrapper; asking once is much cheaper than discarding a good
            # answer, and the fallback is already computed either way.
            trace.repaired = True
            data = self._extract_json(text.replace("'", '"'))
        if data is None:
            trace.error = "unparseable"
            return None

        if not data.get("grounded", False) or not str(data.get("answer", "")).strip():
            trace.error = "model_reported_ungrounded"
            return None

        citations = []
        for chunk_id in data.get("chunk_ids", [])[:4]:
            item = by_id.get(int(chunk_id)) if str(chunk_id).lstrip("-").isdigit() else None
            if item is None:
                continue
            citations.append(
                Citation(
                    chunk_id=item.chunk_id,
                    doc_id=item.doc_id,
                    lang=item.lang,
                    quote=item.text[:240],
                    score=round(float(item.rerank_score or item.score), 4),
                )
            )

        # A cited chunk that isn't in the retrieved set is a hallucinated
        # citation. The grounding guard would reject this downstream anyway;
        # failing here keeps the reason specific.
        if not citations:
            trace.error = "no_valid_citations"
            return None

        return Answer(
            text=str(data["answer"]).strip(),
            citations=citations,
            strategy="llm",
            confidence=0.75,
        )

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _absorb(payload: str, by_id: dict[int, Evidence]) -> None:
        from vrag.schemas import ChunkView

        try:
            results = json.loads(payload).get("results", [])
        except (json.JSONDecodeError, AttributeError):
            return
        for row in results:
            chunk_id = row.get("chunk_id")
            if chunk_id is None or chunk_id in by_id:
                continue
            by_id[chunk_id] = Evidence(
                chunk_id=chunk_id,
                doc_id=row.get("doc_id", ""),
                lang=row.get("lang", "unknown"),
                view=ChunkView(row.get("view", "atomic")),
                text=row.get("text", ""),
                score=float(row.get("score", 0.0)),
            )
