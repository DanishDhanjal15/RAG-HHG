"""Extractive answer generation -- the fast path, and the reason the budget closes.

Generation is where voice RAG systems normally lose the latency argument: a cloud
LLM call is 500 ms+ before it emits a first token. This generator produces the
answer locally in single-digit milliseconds by *selecting* text rather than
writing it.

That is not a compromise on this dataset -- it is the right method for it. MS MARCO
answers are, by construction, short spans drawn from a passage the annotator
marked relevant; the gold ``Answer`` field is frequently near-verbatim from the
selected passage. Selecting the best-supported sentence is what the task actually
is.

It also has a property no generative model has: **it cannot hallucinate**. Every
character it emits came from a retrieved chunk, so groundedness is structural
rather than something we check for afterwards and hope holds. The grounding
guardrail still runs -- but on this path it is verifying an invariant, not
patrolling for fabrication.

Scoring is two-tier so it can degrade under budget pressure:

1. **Lexical** (~0.2 ms) -- IDF-weighted query-term coverage, where the IDF is
   estimated from the retrieved evidence set itself. A local IDF is the right one
   here: within these ~8 passages, a term appearing in all of them genuinely does
   not discriminate between them.
2. **Semantic** (~10-20 ms, optional) -- re-score the lexical top few by embedding
   similarity to the query. Catches paraphrase, which is exactly what lexical
   scoring misses and exactly what a translated corpus is full of.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from vrag.chunking.base import split_sentences
from vrag.config import ExtractiveCfg
from vrag.schemas import Answer, Citation, Evidence

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Candidate sentences are truncated to this before semantic scoring. The batch
# pads to its longest member, so one long sentence sets the cost for the whole
# batch; a query-relevance signal does not need the tail of a 400-character
# sentence to be accurate.
_SEMANTIC_SCORING_MAX_CHARS = 220

# Cross-script question words. Removing them from the query before scoring stops
# every sentence containing "what"/"क्या" from scoring alike -- they are the most
# frequent tokens in a question set and carry no discriminative content.
_QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "the", "a", "an", "of",
    "in", "on", "for", "to", "and", "or",
    "क्या", "कौन", "कब", "कहाँ", "कहां", "क्यों", "कैसे", "कितना", "कितने", "है", "हैं", "का", "की", "के",
    "என்ன", "எது", "யார்", "எப்போது", "எங்கே", "ஏன்", "எப்படி",
    "কি", "কী", "কে", "কখন", "কোথায়", "কেন", "কীভাবে", "কত",
}


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(slots=True)
class SentenceCandidate:
    text: str
    evidence_index: int
    sentence_index: int
    lexical: float = 0.0
    semantic: float = 0.0
    score: float = 0.0


class ExtractiveGenerator:
    def __init__(self, cfg: ExtractiveCfg, embedder=None) -> None:  # noqa: ANN001
        self.cfg = cfg
        self.embedder = embedder

    # -- scoring ------------------------------------------------------------- #
    @staticmethod
    def _local_idf(docs: list[list[str]]) -> dict[str, float]:
        n = len(docs)
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(doc))
        return {term: math.log(1.0 + n / (1.0 + count)) for term, count in df.items()}

    def _score_lexical(
        self, query_terms: set[str], idf: dict[str, float], sentence_terms: list[str]
    ) -> float:
        if not sentence_terms or not query_terms:
            return 0.0
        present = query_terms.intersection(sentence_terms)
        if not present:
            return 0.0
        covered = sum(idf.get(term, 1.0) for term in present)
        total = sum(idf.get(term, 1.0) for term in query_terms) or 1.0
        # Normalising by sqrt(length) rather than length keeps the scorer from
        # collapsing onto three-word fragments that happen to contain one query term.
        return (covered / total) / math.sqrt(len(sentence_terms))

    # -- main ---------------------------------------------------------------- #
    def generate(
        self,
        query: str,
        evidence: list[Evidence],
        query_lang: str = "unknown",
        semantic_scoring: bool = True,
        query_vector: np.ndarray | None = None,
    ) -> tuple[Answer, float]:
        """Returns ``(answer, semantic_ms)``.

        The semantic re-scoring cost is reported separately so the harness can
        budget it as its own stage. It is by far the more expensive half -- on a
        CPU-only box the embedding pass costs an order of magnitude more than the
        lexical selection it refines -- so lumping them into one ``generate``
        number would leave the budget manager unable to drop the expensive part
        while keeping the cheap one.
        """
        if not evidence:
            return Answer(text="", strategy="none", confidence=0.0), 0.0

        candidates: list[SentenceCandidate] = []
        tokenized: list[list[str]] = []

        for e_idx, item in enumerate(evidence):
            sentences = split_sentences(item.text, min_chars=20, merge_short=True)
            if not sentences:
                sentences = split_sentences(item.text)
            for s_idx, sentence in enumerate(sentences):
                candidates.append(
                    SentenceCandidate(
                        text=sentence.text, evidence_index=e_idx, sentence_index=s_idx
                    )
                )
                tokenized.append(_tokens(sentence.text))

        if not candidates:
            return Answer(text="", strategy="none", confidence=0.0), 0.0

        idf = self._local_idf(tokenized)
        query_terms = {t for t in _tokens(query) if t not in _QUESTION_WORDS}
        if not query_terms:
            query_terms = set(_tokens(query))

        for cand, terms in zip(candidates, tokenized, strict=True):
            cand.lexical = self._score_lexical(query_terms, idf, terms)
            item = evidence[cand.evidence_index]

            # Retrieval rank prior: the reranker already decided which chunk is
            # best, and that judgement is stronger than any sentence-level lexical
            # signal. Decay by position rather than ignoring it.
            rank_prior = 1.0 / (1.0 + 0.6 * cand.evidence_index)
            # Lead-sentence prior: MS MARCO passages state their topic up front.
            lead_prior = 1.0 if cand.sentence_index == 0 else 0.85
            # Answer in the language that was asked, where possible. Cross-lingual
            # evidence still supports the answer as a citation.
            lang_prior = 1.15 if (query_lang not in ("", "unknown") and item.lang == query_lang) else 1.0

            cand.score = cand.lexical * rank_prior * lead_prior * lang_prior

        candidates.sort(key=lambda c: -c.score)

        semantic_ms = 0.0
        if semantic_scoring and self.embedder is not None:
            t0 = time.perf_counter()
            candidates = self._refine_semantic(query, candidates, query_vector)
            semantic_ms = (time.perf_counter() - t0) * 1000.0

        return self._compose(candidates, evidence), semantic_ms

    def _refine_semantic(
        self,
        query: str,
        candidates: list[SentenceCandidate],
        query_vector: np.ndarray | None,
        top_k: int = 4,
    ) -> list[SentenceCandidate]:
        """Re-score the lexical shortlist by embedding similarity.

        Only the top few, in one batched forward pass. Scoring every sentence
        would cost far more than the whole generation budget.

        ``top_k`` is 4, not 6, and candidates are truncated before embedding: the
        batch pads to its longest member, so one long sentence sets the cost for
        all of them. Both were measured -- on this CPU the six-candidate,
        untruncated version cost 65-127 ms, several times the entire budget for
        this stage.
        """
        head = candidates[:top_k]
        if len(head) < 2:
            return candidates

        qv = query_vector if query_vector is not None else self.embedder.encode_query(query)
        texts = [c.text[:_SEMANTIC_SCORING_MAX_CHARS] for c in head]
        vectors = self.embedder.encode(texts, prefix="passage: ")
        sims = vectors @ qv

        for cand, sim in zip(head, sims, strict=True):
            cand.semantic = float(sim)
            # Blend rather than replace: lexical carries exact-match evidence
            # (numbers, names) that embeddings smooth away, and semantic carries
            # paraphrase that lexical cannot see. Both failure modes are real.
            cand.score = 0.45 * cand.score + 0.55 * max(0.0, cand.semantic)

        head.sort(key=lambda c: -c.score)
        return head + candidates[top_k:]

    def _compose(
        self, candidates: list[SentenceCandidate], evidence: list[Evidence]
    ) -> Answer:
        best = candidates[0]
        if best.score <= 0:
            return Answer(text="", strategy="none", confidence=0.0)

        source = evidence[best.evidence_index]
        parts = [best.text]
        used = {(best.evidence_index, best.sentence_index)}

        # Extend with the following sentence when the best one is too short to
        # stand alone -- a bare "1911." is a correct span and a useless answer.
        if len(best.text) < 80 and self.cfg.lead_context_sentences > 0:
            for cand in candidates[1:]:
                if cand.evidence_index != best.evidence_index:
                    continue
                if cand.sentence_index != best.sentence_index + 1:
                    continue
                parts.append(cand.text)
                used.add((cand.evidence_index, cand.sentence_index))
                break

        # Add further high-scoring sentences up to the sentence cap, but only
        # from the same source, so the answer never stitches two documents into a
        # claim neither of them makes.
        for cand in candidates[1:]:
            if len(parts) >= self.cfg.max_sentences:
                break
            key = (cand.evidence_index, cand.sentence_index)
            if key in used or cand.evidence_index != best.evidence_index:
                continue
            if cand.score < best.score * 0.6:
                break
            parts.append(cand.text)
            used.add(key)

        text = " ".join(parts).strip()
        if len(text) > self.cfg.max_answer_chars:
            text = text[: self.cfg.max_answer_chars].rsplit(" ", 1)[0] + " …"

        if len(text) < self.cfg.min_answer_chars:
            return Answer(text="", strategy="none", confidence=0.0)

        citations = [
            Citation(
                chunk_id=source.chunk_id,
                doc_id=source.doc_id,
                lang=source.lang,
                quote=best.text[:240],
                score=round(float(source.rerank_score or source.score), 4),
            )
        ]
        # Cite corroborating chunks: distinct passages whose text also supports
        # the answer. Shown in the UI as supporting evidence.
        for cand in candidates[1:4]:
            other = evidence[cand.evidence_index]
            if other.doc_id == source.doc_id or cand.score < best.score * 0.5:
                continue
            citations.append(
                Citation(
                    chunk_id=other.chunk_id,
                    doc_id=other.doc_id,
                    lang=other.lang,
                    quote=cand.text[:240],
                    score=round(float(cand.score), 4),
                )
            )

        confidence = float(min(1.0, max(0.0, best.score if best.semantic else best.score * 4)))
        return Answer(
            text=text, citations=citations, strategy="extractive", confidence=round(confidence, 4)
        )
