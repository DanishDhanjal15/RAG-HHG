"""Tool registry and LLM polish-path tests.

This path is optional and off the critical path, which makes it exactly the code
that rots unnoticed: it needs an API key to run, so it never executes in CI or in
a normal dev loop, and a regression sits there until a demo.

Both halves are testable without a key:

* ``ToolRegistry`` is pure dispatch over a retriever -- a fake retriever exercises
  it completely.
* ``LlmGenerator``'s loop, JSON parsing, citation validation and failure handling
  are driven by a stub client, so "what happens when the model returns prose
  instead of JSON" is a test rather than a hope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from vrag.config import LlmCfg, load_config
from vrag.generate.llm import LlmGenerator
from vrag.harness.tools import ToolRegistry
from vrag.schemas import ChunkView, Evidence


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def ev(text: str, chunk_id: int, lang: str = "en") -> Evidence:
    return Evidence(
        chunk_id=chunk_id, doc_id=f"{lang}:1:{chunk_id}", lang=lang,
        view=ChunkView.ATOMIC, text=text, score=0.9,
    )


class FakeRetriever:
    def __init__(self, results: list[Evidence]) -> None:
        self._results = results
        self.searches: list[str] = []
        self.cfg = load_config()

    def embed_query(self, plan):  # noqa: ANN001, ARG002
        return None

    def dense_search(self, vector, plan):  # noqa: ANN001, ARG002
        self.searches.append(plan.normalized_query)
        return {}

    def sparse_search(self, plan):  # noqa: ANN001, ARG002
        return {}

    def fuse(self, runs):  # noqa: ANN001, ARG002
        return []

    def hydrate(self, candidates, limit=None):  # noqa: ANN001, ARG002
        return self._results[: (limit or len(self._results))]

    def neighbours(self, chunk_id: int) -> list[Evidence]:
        if chunk_id == 999:
            raise IndexError("no such chunk")
        return [ev("neighbouring context sentence", chunk_id + 1)]


@dataclass
class FakePipeline:
    retriever: FakeRetriever
    sparse: Any = None
    cfg: Any = field(default_factory=load_config)


# --------------------------------------------------------------------------- #
class TestToolRegistry:
    @pytest.fixture
    def registry(self) -> ToolRegistry:
        retriever = FakeRetriever([ev("A corporation is a company.", 1),
                                   ev("It is recognised in law.", 2)])
        return ToolRegistry(FakePipeline(retriever=retriever, cfg=retriever.cfg))

    def test_exposes_tool_specs(self, registry):
        names = {spec["name"] for spec in registry.specs()}
        assert names == {"search_corpus", "fetch_neighbours", "normalize_query"}

    def test_every_spec_has_a_schema_and_description(self, registry):
        for spec in registry.specs():
            assert spec["description"].strip()
            assert spec["input_schema"]["type"] == "object"
            assert spec["input_schema"].get("required")

    def test_search_returns_citable_chunk_ids(self, registry):
        payload = json.loads(registry.dispatch("search_corpus", {"query": "corporation"}))
        assert payload["results"]
        assert payload["results"][0]["chunk_id"] == 1

    def test_search_respects_k(self, registry):
        payload = json.loads(
            registry.dispatch("search_corpus", {"query": "corporation", "k": 1})
        )
        assert len(payload["results"]) == 1

    def test_fetch_neighbours(self, registry):
        payload = json.loads(registry.dispatch("fetch_neighbours", {"chunk_id": 1}))
        assert payload["results"][0]["chunk_id"] == 2

    def test_normalize_query_strips_disfluency(self, registry):
        payload = json.loads(
            registry.dispatch("normalize_query", {"text": "um what is a corporation"})
        )
        assert "um" not in payload["normalized"].split()

    # -- errors are returned TO THE MODEL, never raised ---------------------- #
    def test_unknown_tool_returns_a_structured_error(self, registry):
        payload = json.loads(registry.dispatch("no_such_tool", {}))
        assert "error" in payload
        assert "available" in payload

    def test_bad_arguments_return_the_schema(self, registry):
        """A model that called a tool wrongly can correct itself if told how."""
        payload = json.loads(registry.dispatch("search_corpus", {"wrong_kwarg": 1}))
        assert "error" in payload
        assert "expected_schema" in payload

    def test_handler_exception_is_captured_not_raised(self, registry):
        payload = json.loads(registry.dispatch("fetch_neighbours", {"chunk_id": 999}))
        assert "error" in payload
        assert payload["results"] == []


# --------------------------------------------------------------------------- #
# Stub Anthropic client
# --------------------------------------------------------------------------- #
class Block:
    def __init__(self, type_: str, **kw: Any) -> None:
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class Response:
    def __init__(self, content: list[Block], stop_reason: str = "end_turn") -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 20})()


class StubMessages:
    def __init__(self, responses: list[Response]) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    def create(self, **kw: Any) -> Response:
        self.calls.append(kw)
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


class StubClient:
    def __init__(self, responses: list[Response]) -> None:
        self.messages = StubMessages(responses)


def make_generator(responses: list[Response]) -> LlmGenerator:
    retriever = FakeRetriever([ev("A corporation is a company.", 1)])
    registry = ToolRegistry(FakePipeline(retriever=retriever, cfg=retriever.cfg))
    gen = LlmGenerator.__new__(LlmGenerator)   # skip the real Anthropic constructor
    gen.cfg = LlmCfg(max_tool_rounds=2, hard_timeout_ms=5000)
    gen.registry = registry
    gen.client = StubClient(responses)
    return gen


def text_response(payload: dict) -> Response:
    return Response([Block("text", text=json.dumps(payload, ensure_ascii=False))])


EVIDENCE = [ev("A corporation is a company authorized to act as one entity.", 1)]


# --------------------------------------------------------------------------- #
class TestLlmGenerator:
    def test_grounded_answer_is_returned_with_citations(self):
        gen = make_generator([text_response(
            {"answer": "A corporation acts as a single legal entity.",
             "chunk_ids": [1], "grounded": True}
        )])
        answer, trace = gen.generate("what is a corporation", EVIDENCE)
        assert answer is not None
        assert answer.strategy == "llm"
        assert answer.citations[0].chunk_id == 1
        assert trace.rounds == 1

    def test_model_reporting_ungrounded_falls_back(self):
        """The model saying it cannot answer must NOT become an empty answer."""
        gen = make_generator([text_response(
            {"answer": "", "chunk_ids": [], "grounded": False}
        )])
        answer, trace = gen.generate("q", EVIDENCE)
        assert answer is None
        assert trace.error == "model_reported_ungrounded"

    def test_hallucinated_citation_is_rejected(self):
        """A chunk id that was never retrieved cannot be cited."""
        gen = make_generator([text_response(
            {"answer": "Something plausible.", "chunk_ids": [4242], "grounded": True}
        )])
        answer, trace = gen.generate("q", EVIDENCE)
        assert answer is None
        assert trace.error == "no_valid_citations"

    def test_prose_instead_of_json_falls_back_cleanly(self):
        gen = make_generator([Response([Block("text", text="I think it is a company.")])])
        answer, trace = gen.generate("q", EVIDENCE)
        assert answer is None
        assert trace.error == "unparseable"

    def test_json_embedded_in_prose_is_extracted(self):
        gen = make_generator([Response([Block(
            "text",
            text='Sure! {"answer": "A corporation is a company.", '
                 '"chunk_ids": [1], "grounded": true} hope that helps',
        )])])
        answer, _ = gen.generate("q", EVIDENCE)
        assert answer is not None
        assert "corporation" in answer.text

    def test_refusal_is_handled(self):
        gen = make_generator([Response([], stop_reason="refusal")])
        answer, trace = gen.generate("q", EVIDENCE)
        assert answer is None
        assert trace.error == "refusal"

    def test_api_exception_never_escapes(self):
        """This path is optional. It must degrade, never fail the request."""
        gen = make_generator([])

        def boom(**kw: Any):  # noqa: ANN202, ARG001
            raise RuntimeError("connection reset")

        gen.client.messages.create = boom
        answer, trace = gen.generate("q", EVIDENCE)
        assert answer is None
        assert "RuntimeError" in (trace.error or "")

    def test_usage_is_accumulated(self):
        gen = make_generator([text_response(
            {"answer": "A corporation is a company.", "chunk_ids": [1], "grounded": True}
        )])
        _, trace = gen.generate("q", EVIDENCE)
        assert trace.input_tokens == 10
        assert trace.output_tokens == 20


class TestToolLoop:
    @staticmethod
    def tool_call_response() -> Response:
        return Response(
            [Block("tool_use", id="tu_1", name="search_corpus",
                   input={"query": "corporation"})],
            stop_reason="tool_use",
        )

    def test_tool_call_is_executed_and_fed_back(self):
        gen = make_generator([
            self.tool_call_response(),
            text_response({"answer": "A corporation is a company.",
                           "chunk_ids": [1], "grounded": True}),
        ])
        answer, trace = gen.generate("what is a corporation", EVIDENCE)
        assert answer is not None
        assert trace.tool_calls == ["search_corpus"]
        assert trace.rounds == 2

        # Results must go back in ONE user message -- splitting them across
        # messages trains the model out of parallel tool calls.
        second = gen.client.messages.calls[1]
        tool_results = [
            m for m in second["messages"]
            if m["role"] == "user" and isinstance(m["content"], list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_results) == 1

    def test_assistant_turn_is_appended_whole(self):
        """Dropping tool_use blocks breaks block pairing on the next request."""
        gen = make_generator([
            self.tool_call_response(),
            text_response({"answer": "A corporation is a company.",
                           "chunk_ids": [1], "grounded": True}),
        ])
        gen.generate("q", EVIDENCE)
        second = gen.client.messages.calls[1]
        assistant = [m for m in second["messages"] if m["role"] == "assistant"]
        assert assistant
        assert any(getattr(b, "type", None) == "tool_use" for b in assistant[0]["content"])

    def test_round_limit_is_enforced(self):
        """An unbounded agent loop behind a voice interface is a hang."""
        gen = make_generator([self.tool_call_response()])   # never stops calling tools
        answer, trace = gen.generate("q", EVIDENCE)
        assert answer is None
        assert trace.error == "max_tool_rounds"
        assert trace.rounds <= gen.cfg.max_tool_rounds + 1

    def test_newly_retrieved_chunks_become_citable(self):
        """Otherwise the model can only ever cite what it started with, which
        defeats the point of giving it a search tool."""
        gen = make_generator([
            self.tool_call_response(),
            text_response({"answer": "A corporation is a company.",
                           "chunk_ids": [1], "grounded": True}),
        ])
        answer, _ = gen.generate("q", [])   # started with NO evidence
        assert answer is not None
        assert answer.citations
