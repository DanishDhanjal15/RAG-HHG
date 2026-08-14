"""Query normalization and planning.

Turns a raw transcript into a ``QueryPlan``: the exact text to embed, which views
to search, and whether to trust the detected language enough to filter on it.

Deliberately *not* here: LLM query rewriting and HyDE. Both are well-established
recall wins and both require a generation round trip before retrieval can even
start -- several hundred milliseconds spent before the 200 ms budget's first
stage. They belong to a system with a different SLA.

What we do instead is free: strip ASR artifacts, and lean on the multilingual
encoder for cross-lingual matching rather than translating the query. ``e5``
places a Hindi question and its English answer passage in the same region of the
space, so the English parallel passages in the index are reachable from an Indic
query without any translation step at all.
"""

from __future__ import annotations

import re

from vrag.config import Config
from vrag.schemas import ChunkView, QueryPlan, Transcript

_WS = re.compile(r"\s+")

# Disfluencies and transcription noise, per script. ASR emits these verbatim and
# they carry no retrieval signal -- but they do occupy tokens and shift the mean
# pooled vector, so removing them measurably tightens short queries.
_FILLERS = [
    r"\bum+\b", r"\buh+\b", r"\berm\b", r"\bhmm+\b", r"\byou know\b", r"\bi mean\b",
    r"\bमतलब\b", r"\bयानी\b", r"\bअरे\b",
    r"\bஅதாவது\b",
    r"\bমানে\b",
]
_FILLER_RE = re.compile("|".join(_FILLERS), re.IGNORECASE | re.UNICODE)

# Leading politeness that ASR captures and that dilutes a short question.
_LEAD_RE = re.compile(
    r"^\s*(please|kindly|hey|hello|hi|ok(?:ay)?|so)\s*[,:]?\s+",
    re.IGNORECASE,
)


def normalize_query(text: str) -> str:
    text = _WS.sub(" ", text).strip()
    text = _FILLER_RE.sub(" ", text)
    text = _LEAD_RE.sub("", text)
    text = _WS.sub(" ", text).strip()
    # Strip a trailing run of punctuation but keep a single question mark: it is
    # a genuine cue that the text is a question, in every script here.
    return re.sub(r"[\s,;:.\-]+$", "", text)


def plan_query(cfg: Config, transcript: Transcript) -> QueryPlan:
    normalized = normalize_query(transcript.text)
    views = [ChunkView(v) for v in cfg.chunking.views.enabled_names()]

    lf = cfg.retrieval.language_filter
    confident = transcript.confidence >= lf.min_asr_confidence and transcript.lang not in (
        "",
        "unknown",
    )

    lang_filter: list[str] | None = None
    if confident:
        # Filter to the detected language, but keep English in scope: the English
        # parallel passages are often the better evidence, and excluding them
        # would throw away the cross-lingual capability the index was built for.
        lang_filter = [transcript.lang]
        if lf.always_include_english and transcript.lang != "en":
            lang_filter.append("en")

    return QueryPlan(
        raw_query=transcript.text,
        normalized_query=normalized,
        embed_text=f"{cfg.embedding.query_prefix}{normalized}",
        lang=transcript.lang,
        lang_confident=confident,
        lang_filter=lang_filter,
        views=views,
    )


def plan_text(cfg: Config, text: str, lang: str = "unknown", confidence: float = 1.0) -> QueryPlan:
    """Planning entry point for typed input and for the benchmarks, which run
    thousands of queries without any audio."""
    return plan_query(
        cfg, Transcript(text=text, lang=lang, confidence=confidence, provider="text")
    )
