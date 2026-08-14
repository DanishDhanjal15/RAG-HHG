"""Extractive generator tests.

This is the component on the critical path, and the reason the system can claim
it cannot hallucinate: it selects spans from retrieved text rather than writing
new text. That claim deserves tests that actually try to break it -- not just a
happy-path check.

Runs without an embedder (``semantic_scoring=False``), so these are fast and have
no model dependency. The semantic refinement path is exercised separately with a
stub.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrag.config import ExtractiveCfg
from vrag.generate.extractive import ExtractiveGenerator
from vrag.schemas import ChunkView, Evidence


def ev(text: str, chunk_id: int = 1, doc: str | None = None, lang: str = "en",
       score: float = 0.9) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        doc_id=doc or f"{lang}:1:{chunk_id}",
        lang=lang,
        view=ChunkView.ATOMIC,
        text=text,
        score=score,
    )


@pytest.fixture
def gen() -> ExtractiveGenerator:
    return ExtractiveGenerator(ExtractiveCfg())


def answer_of(gen, query, evidence, **kw):  # noqa: ANN001, ANN201
    ans, _ms = gen.generate(query, evidence, semantic_scoring=False, **kw)
    return ans


# --------------------------------------------------------------------------- #
class TestGrounding:
    """The central property: output is always a substring-level selection of
    retrieved text, never new prose."""

    def test_answer_text_comes_from_the_evidence(self, gen):
        source = (
            "A corporation is a company authorized to act as a single entity. "
            "It is recognized as such in law. Shareholders own it jointly."
        )
        ans = answer_of(gen, "what is a corporation", [ev(source)])
        assert ans.text
        # Every sentence emitted must appear verbatim in the source.
        for part in ans.text.split(". "):
            cleaned = part.strip().rstrip(".")
            if cleaned:
                assert cleaned in source, f"invented text: {cleaned!r}"

    def test_never_stitches_two_documents_into_one_claim(self, gen):
        a = "The tower is 324 metres tall. It stands in Paris and opened in 1889."
        b = "The bridge is 500 metres long. It stands in London and opened in 1894."
        ans = answer_of(gen, "how tall is the tower in Paris",
                        [ev(a, 1, doc="en:1:0"), ev(b, 2, doc="en:2:0")])
        if ans.text:
            # Composition is restricted to a single source document, so a claim
            # can never be assembled from two unrelated passages.
            assert not (("324" in ans.text) and ("500" in ans.text))

    def test_every_answer_carries_a_citation(self, gen):
        ans = answer_of(gen, "what is a corporation",
                        [ev("A corporation is a company authorized to act as one entity.")])
        assert ans.text
        assert ans.citations
        assert ans.citations[0].chunk_id == 1

    def test_citation_quote_is_from_the_cited_chunk(self, gen):
        source = "A corporation is a company authorized to act as a single entity."
        ans = answer_of(gen, "what is a corporation", [ev(source)])
        assert ans.citations[0].quote.rstrip(" …") in source


class TestSelection:
    def test_picks_the_sentence_that_answers_the_question(self, gen):
        source = (
            "Many businesses exist in various forms today. "
            "A corporation is a company authorized to act as a single legal entity. "
            "Weather patterns vary considerably by season."
        )
        ans = answer_of(gen, "what is a corporation", [ev(source)])
        assert "authorized to act" in ans.text

    def test_question_words_do_not_dominate_scoring(self, gen):
        """Without stripping them, every sentence containing "what"/"is" scores
        alike and selection becomes arbitrary."""
        source = (
            "What is happening here is unclear and it is what it is. "
            "Photosynthesis converts light energy into chemical energy in plants."
        )
        ans = answer_of(gen, "what is photosynthesis", [ev(source)])
        assert "Photosynthesis converts" in ans.text

    def test_prefers_evidence_in_the_query_language(self, gen):
        hindi = ev("निगम एक कंपनी है जो एकल इकाई के रूप में कार्य करती है।",
                   chunk_id=1, lang="hi")
        english = ev("A corporation is a company that acts as a single entity.",
                     chunk_id=2, lang="en")
        ans = answer_of(gen, "निगम क्या है", [hindi, english], query_lang="hi")
        assert ans.text
        assert ans.citations[0].lang == "hi"

    def test_extends_a_too_short_leading_span(self, gen):
        """A bare "1911." is a correct span and a useless answer."""
        source = "1911. The company was founded in Detroit by two engineers that year."
        ans = answer_of(gen, "when was the company founded", [ev(source)])
        assert len(ans.text) > 20

    def test_respects_the_sentence_cap(self):
        gen = ExtractiveGenerator(ExtractiveCfg(max_sentences=1))
        source = ". ".join(
            f"Corporation fact number {i} about corporations and companies" for i in range(6)
        ) + "."
        ans = answer_of(gen, "corporation facts", [ev(source)])
        assert ans.text.count(".") <= 2

    def test_respects_the_character_cap(self):
        gen = ExtractiveGenerator(ExtractiveCfg(max_answer_chars=60, max_sentences=5))
        source = " ".join(["A corporation is a company authorized to act."] * 12)
        ans = answer_of(gen, "what is a corporation", [ev(source)])
        assert len(ans.text) <= 64  # cap plus the ellipsis marker


class TestDegenerateInput:
    def test_no_evidence_returns_nothing(self, gen):
        ans = answer_of(gen, "anything", [])
        assert ans.text == ""
        assert ans.strategy == "none"

    def test_evidence_with_no_query_overlap_returns_nothing(self, gen):
        ans = answer_of(gen, "quantum chromodynamics lagrangian",
                        [ev("Penguins are flightless birds found in Antarctica.")])
        assert ans.text == "" or ans.confidence < 0.5

    def test_empty_query(self, gen):
        ans = answer_of(gen, "", [ev("A corporation is a company.")])
        assert isinstance(ans.text, str)

    def test_whitespace_only_evidence(self, gen):
        ans = answer_of(gen, "what is a corporation", [ev("   ")])
        assert ans.text == ""

    def test_evidence_without_sentence_terminators(self, gen):
        ans = answer_of(gen, "corporation",
                        [ev("a corporation is a company with no terminator")])
        assert isinstance(ans.text, str)

    def test_indic_text_without_latin_punctuation(self, gen):
        source = "निगम एक कंपनी है। यह कानून द्वारा मान्यता प्राप्त है। यह एक इकाई है।"
        ans = answer_of(gen, "निगम क्या है", [ev(source, lang="hi")], query_lang="hi")
        assert ans.text
        assert ans.text in source or any(s.strip() in source for s in ans.text.split("।"))


class TestSemanticRefinement:
    """The optional, expensive half -- exercised with a stub so the test stays
    fast and deterministic."""

    class StubEmbedder:
        """Scores by keyword presence, so the 'semantic' winner is predictable."""

        def __init__(self, keyword: str) -> None:
            self.keyword = keyword
            self.calls = 0

        def encode(self, texts, prefix=""):  # noqa: ANN001, ARG002
            self.calls += 1
            return np.array(
                [[1.0, 0.0] if self.keyword in t else [0.0, 1.0] for t in texts],
                dtype=np.float32,
            )

        def encode_query(self, text):  # noqa: ANN001, ARG002
            return np.array([1.0, 0.0], dtype=np.float32)

    def test_semantic_scoring_can_override_lexical_choice(self):
        stub = self.StubEmbedder("photosynthesis")
        gen = ExtractiveGenerator(ExtractiveCfg(), embedder=stub)
        source = (
            "The process is complex and the process has many steps in the process. "
            "Photosynthesis converts light into chemical energy."
        )
        ans, ms = gen.generate("what is the process", [ev(source)], semantic_scoring=True)
        assert stub.calls > 0
        assert ms >= 0.0
        assert "Photosynthesis" in ans.text

    def test_reports_its_own_cost_separately(self):
        """The harness budgets this as its own stage, so the generator has to
        hand back how long the semantic half took."""
        stub = self.StubEmbedder("corporation")
        gen = ExtractiveGenerator(ExtractiveCfg(), embedder=stub)
        source = "A corporation is a company. It acts as one entity. Law recognises it."

        _, with_semantic = gen.generate("what is a corporation", [ev(source)],
                                        semantic_scoring=True)
        _, without = gen.generate("what is a corporation", [ev(source)],
                                  semantic_scoring=False)
        assert without == 0.0
        assert with_semantic > 0.0

    def test_disabled_semantic_never_calls_the_embedder(self):
        stub = self.StubEmbedder("x")
        gen = ExtractiveGenerator(ExtractiveCfg(), embedder=stub)
        gen.generate("q", [ev("A corporation is a company that acts as an entity.")],
                     semantic_scoring=False)
        assert stub.calls == 0
