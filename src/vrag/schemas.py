"""Typed boundaries between every pipeline stage.

Nothing crosses a stage boundary untyped. This is the backbone of the harness
requirement: each stage declares exactly what it consumes and produces, so a
malformed intermediate fails at the boundary that produced it rather than three
stages later inside a string-formatting call.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Corpus / chunk primitives
# --------------------------------------------------------------------------- #
class ChunkView(StrEnum):
    ATOMIC = "atomic"
    SENTENCE_WINDOW = "sentence_window"
    FIXED_OVERLAP = "fixed_overlap"
    SEMANTIC = "semantic"
    PROPOSITION = "proposition"


class Passage(BaseModel):
    """One MS MARCO passage, in one language."""

    doc_id: str                 # f"{lang}:{query_id}:{passage_idx}"
    query_id: int
    passage_idx: int
    lang: str                   # "hi" | "ta" | "bn" | "en"
    text: str
    is_selected: bool = False   # gold relevance label from the dataset
    parallel_en_id: str | None = None   # link to the English parallel passage


class QueryRecord(BaseModel):
    """A dataset query plus its gold answer -- used for eval, never at serve time."""

    query_id: int
    lang: str
    query: str
    answer: str
    eng_query: str = ""
    eng_answer: str = ""
    query_type: str = ""
    gold_doc_ids: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """A retrievable unit. Carries enough metadata to filter, dedup, and cite."""

    chunk_id: int               # dense index position; the primary key everywhere
    doc_id: str
    query_id: int
    lang: str
    view: ChunkView
    text: str                   # the text that gets embedded
    context_text: str           # what gets shown/returned (window-expanded)
    char_start: int
    char_end: int
    is_selected: bool = False
    parallel_en_id: str | None = None
    neighbour_ids: list[int] = Field(default_factory=list)

    @property
    def span(self) -> tuple[int, int]:
        return (self.char_start, self.char_end)


# --------------------------------------------------------------------------- #
# Stage I/O
# --------------------------------------------------------------------------- #
class AudioInput(BaseModel):
    audio: bytes
    filename: str = "audio.wav"
    mime_type: str = "audio/wav"
    lang_hint: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class Transcript(BaseModel):
    text: str
    lang: str = "unknown"
    confidence: float = 0.0     # provider language_probability, reused as an ASR gate
    provider: str = ""
    request_id: str = ""
    audio_duration_s: float | None = None


class RefusalReason(StrEnum):
    """Why we declined to answer. Every abstention carries exactly one of these.

    A typed enum rather than a free-text message so the guardrail benchmark can
    score decisions, and so the UI can render an appropriate recovery action.
    """

    LOW_CONFIDENCE_ASR = "LOW_CONFIDENCE_ASR"
    EMPTY_INPUT = "EMPTY_INPUT"
    INPUT_TOO_LONG = "INPUT_TOO_LONG"
    UNSAFE_INPUT = "UNSAFE_INPUT"
    INJECTION_ATTEMPT = "INJECTION_ATTEMPT"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    NO_GROUNDING = "NO_GROUNDING"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    STT_UNAVAILABLE = "STT_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class GuardVerdict(BaseModel):
    allowed: bool = True
    reason: RefusalReason | None = None
    detail: str = ""
    signals: dict[str, float] = Field(default_factory=dict)
    redacted_text: str | None = None    # PII-scrubbed variant, safe to log


class QueryPlan(BaseModel):
    """What the retriever is actually going to execute."""

    raw_query: str
    normalized_query: str
    embed_text: str                     # normalized + e5 "query: " prefix
    lang: str = "unknown"
    lang_confident: bool = False
    lang_filter: list[str] | None = None
    views: list[ChunkView] = Field(default_factory=list)
    expansions: list[str] = Field(default_factory=list)
    # Best raw cosine seen during dense search -- the out-of-domain guard's primary
    # signal. Captured here because the search already computed it; see
    # MultiViewRetriever.cosine_top1.
    top_dense_score: float = 0.0


class ScoredChunk(BaseModel):
    chunk_id: int
    score: float
    view: ChunkView
    source: Literal["dense", "sparse", "fused", "rerank"] = "dense"
    rank: int = 0


class RetrievalResult(BaseModel):
    """Per-view, per-modality hits before fusion. Kept separate so the ablation
    can attribute recall to a specific view."""

    per_view: dict[str, list[ScoredChunk]] = Field(default_factory=dict)
    sparse: list[ScoredChunk] = Field(default_factory=list)
    total_candidates: int = 0


class Evidence(BaseModel):
    """A retrieved chunk hydrated with its text, ready to cite."""

    chunk_id: int
    doc_id: str
    lang: str
    view: ChunkView
    text: str
    score: float
    rerank_score: float | None = None
    is_selected: bool = False


class RankedContext(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    top1_score: float = 0.0
    centroid_distance: float = 1.0
    reranked: bool = False


class Citation(BaseModel):
    chunk_id: int
    doc_id: str
    lang: str
    quote: str
    score: float


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    strategy: Literal["extractive", "llm", "none"] = "extractive"
    confidence: float = 0.0


# --------------------------------------------------------------------------- #
# Observability
# --------------------------------------------------------------------------- #
class Span(BaseModel):
    name: str
    duration_ms: float
    ok: bool = True
    skipped: bool = False
    error: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class Degradation(BaseModel):
    """A stage the budget manager dropped to stay inside the SLA.

    Surfaced in the response so a fast answer is never silently a worse answer.
    """

    stage: str
    reason: str
    remaining_budget_ms: float
    expected_cost_ms: float


class AnswerEnvelope(BaseModel):
    """The single response type. Every path -- success, refusal, crash -- returns
    one of these. There is no other shape the API can emit."""

    request_id: str
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    refusal_reason: RefusalReason | None = None
    refusal_detail: str = ""

    transcript: str = ""
    detected_lang: str = "unknown"
    asr_confidence: float = 0.0

    strategy: Literal["extractive", "llm", "none"] = "none"
    confidence: float = 0.0

    timings_ms: dict[str, float] = Field(default_factory=dict)
    core_latency_ms: float = 0.0        # the <200ms contract
    total_latency_ms: float = 0.0
    within_budget: bool = True
    degradations: list[Degradation] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)

    def refused(self, reason: RefusalReason, detail: str = "") -> AnswerEnvelope:
        self.abstained = True
        self.refusal_reason = reason
        self.refusal_detail = detail
        self.strategy = "none"
        return self
