"""Layer 3 -- groundedness, run after generation and before the answer is returned.

The question this layer answers is: *is every claim in this answer actually
supported by a chunk we retrieved?* An answer that fails is discarded and replaced
with an abstention, because a fluent unsupported answer is worse than no answer --
it is wrong in a way the user cannot detect.

Three checks:

1. **Citation required.** An answer with no citation cannot be returned at all.
   This is what makes the extractive path structurally safe: it can only emit
   spans that exist in retrieved text, so it always has something to cite.
2. **Lexical attribution.** What fraction of the answer's content tokens appear in
   the cited chunk. Catches the case where a generative model has drifted from
   its source. Cheap (~0.3 ms) and precise for extractive output.
3. **Semantic attribution.** Embedding similarity between answer and cited chunk.
   Catches faithful paraphrase, which lexical overlap wrongly punishes -- and
   paraphrase is exactly what the LLM polish path produces.

The two attribution checks are combined with **or**, not **and**: an answer that
is a verbatim span passes check 2 and may fail check 3 on a long chunk; an answer
that is a faithful rewrite passes check 3 and fails check 2. Requiring both would
reject correct answers of both kinds.

An optional NLI tier (entailment rather than similarity) is the stronger check and
is wired as a budget-gated upgrade -- similarity can be high for a sentence that
*contradicts* its source, which is precisely the case NLI catches and similarity
cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from vrag.chunking.base import split_sentences
from vrag.config import GroundingCfg
from vrag.schemas import Answer, Evidence, GuardVerdict, RefusalReason

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Function words carry no attribution signal -- an answer and an unrelated chunk
# will always share "the", "है", "এবং". Counting them inflates overlap toward a
# floor of ~0.4 regardless of grounding.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "with", "as", "by", "from",
    "है", "हैं", "का", "की", "के", "को", "में", "से", "और", "एक", "यह", "वह", "पर",
    "ஆகும்", "ஒரு", "இந்த", "அந்த", "மற்றும்", "என்று",
    "এবং", "একটি", "এই", "সেই", "হয়", "করে", "থেকে",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


@dataclass
class GroundingGuard:
    cfg: GroundingCfg
    embedder: object | None = None

    def check(
        self,
        answer: Answer,
        evidence: list[Evidence],
        semantic: bool = True,
    ) -> GuardVerdict:
        signals: dict[str, float] = {}

        if not answer.text.strip():
            return GuardVerdict(
                allowed=False,
                reason=RefusalReason.NO_GROUNDING,
                detail="I found related passages but none of them answer that question.",
                signals=signals,
            )

        if self.cfg.require_citation and not answer.citations:
            return GuardVerdict(
                allowed=False,
                reason=RefusalReason.NO_GROUNDING,
                detail="I could not attribute an answer to any retrieved passage.",
                signals=signals,
            )

        by_id = {e.chunk_id: e for e in evidence}
        cited = [by_id[c.chunk_id] for c in answer.citations if c.chunk_id in by_id]
        if not cited:
            return GuardVerdict(
                allowed=False,
                reason=RefusalReason.NO_GROUNDING,
                detail="Cited passages are not in the retrieved set.",
                signals=signals,
            )

        support = " ".join(e.text for e in cited)
        lexical = self._lexical_overlap(answer.text, support)
        signals["lexical_overlap"] = round(lexical, 4)

        semantic_score = 0.0
        if semantic and self.embedder is not None and lexical < self.cfg.min_lexical_overlap:
            # Only pay for embedding when lexical already failed. A verbatim
            # extractive answer scores ~1.0 lexically and never reaches this.
            semantic_score = self._semantic_similarity(answer.text, support)
            signals["semantic_similarity"] = round(semantic_score, 4)

        grounded = (
            lexical >= self.cfg.min_lexical_overlap
            or semantic_score >= self.cfg.min_embedding_similarity
        )

        if not grounded:
            return GuardVerdict(
                allowed=False,
                reason=RefusalReason.NO_GROUNDING,
                detail="I could not verify that answer against the retrieved passages.",
                signals=signals,
            )

        signals["grounded"] = 1.0
        return GuardVerdict(allowed=True, signals=signals)

    # -- checks -------------------------------------------------------------- #
    @staticmethod
    def _lexical_overlap(answer: str, support: str) -> float:
        answer_tokens = _content_tokens(answer)
        if not answer_tokens:
            return 0.0
        support_tokens = _content_tokens(support)
        return len(answer_tokens & support_tokens) / len(answer_tokens)

    def _semantic_similarity(self, answer: str, support: str) -> float:
        vectors = self.embedder.encode([answer, support], prefix="passage: ")  # type: ignore[union-attr]
        return float(np.dot(vectors[0], vectors[1]))

    # -- per-sentence attribution -------------------------------------------- #
    def unsupported_sentences(self, answer: Answer, evidence: list[Evidence]) -> list[str]:
        """Sentence-level attribution, for the UI and the guardrail benchmark.

        The pass/fail decision above is whole-answer; this shows *which* sentence
        is the weak one, which is what makes a refusal explainable rather than
        merely correct.
        """
        support_tokens = _content_tokens(" ".join(e.text for e in evidence))
        weak: list[str] = []
        for sentence in split_sentences(answer.text, min_chars=15):
            tokens = _content_tokens(sentence.text)
            if not tokens:
                continue
            if len(tokens & support_tokens) / len(tokens) < self.cfg.min_lexical_overlap:
                weak.append(sentence.text)
        return weak


@dataclass
class ConflictDetector:
    """Flags top evidence that disagrees rather than silently picking a winner.

    Two retrieved passages can give different numbers for the same fact -- MS MARCO
    is scraped from the web and contains genuinely contradictory pages. Choosing
    one at random and presenting it as fact is a quiet failure; surfacing both is
    the honest behaviour.

    The heuristic is deliberately narrow, and two constraints do most of the work
    of keeping it that way:

    1. **Only across different documents.** Two chunks of the *same* passage
       naturally carry different subsets of its numbers -- an overlapping window is
       often a strict prefix of its neighbour -- so comparing them finds
       "disagreement" on every numeric passage in the corpus. This was a real
       false positive: two chunks of one company-registration passage were flagged
       as contradictory sources and blocked a perfectly good answer.
    2. **Only between near-duplicate passages.** Two documents about the same
       subject routinely mention different numbers without contradicting anything
       (a date in one, a price in the other). A genuine contradiction looks like
       two passages that say almost the same words and differ on the figure, so the
       topical-overlap bar is set high.

    General semantic contradiction detection needs an NLI model and a latency
    budget this stage does not have. This catches the narrow, high-confidence case
    and stays quiet otherwise -- a conflict guard that misfires on ordinary queries
    is worse than no conflict guard, because it blocks correct answers.
    """

    min_disagreement: float = 0.45
    min_topical_overlap: float = 0.55
    enabled: bool = True

    _NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

    def check(self, evidence: list[Evidence], top_n: int = 4) -> tuple[bool, dict[str, float]]:
        if not self.enabled or len(evidence) < 2:
            return False, {}

        head = evidence[:top_n]
        numbers = [set(self._NUMBER_RE.findall(e.text)) for e in head]
        topics = [_content_tokens(e.text) for e in head]

        for i in range(len(head)):
            for j in range(i + 1, len(head)):
                # Constraint 1: different source documents only.
                if head[i].doc_id == head[j].doc_id:
                    continue
                if not numbers[i] or not numbers[j]:
                    continue

                # Constraint 2: near-duplicate passages only.
                union = topics[i] | topics[j]
                if not union:
                    continue
                topical = len(topics[i] & topics[j]) / len(union)
                if topical < self.min_topical_overlap:
                    continue

                shared = numbers[i] & numbers[j]
                disagreement = 1.0 - (len(shared) / max(len(numbers[i] | numbers[j]), 1))
                if disagreement >= self.min_disagreement:
                    return True, {
                        "topical_overlap": round(topical, 4),
                        "numeric_disagreement": round(disagreement, 4),
                    }

        return False, {}
