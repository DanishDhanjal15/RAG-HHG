"""Layer 1 -- input guardrails, run before any retrieval happens.

Six checks, cheapest first, short-circuiting on the first failure. Total cost is
~0.2 ms, which is what lets them sit on the critical path rather than being
sampled or run asynchronously.

On method: these are lexicon and pattern based, not a neural classifier. That is
a deliberate trade and it is stated plainly rather than dressed up. A transformer
safety classifier would be more robust to paraphrase, and would also add a model
download, ~20 ms of latency, and a second thing that can fail to load. Because the
decision is *abstain*, the cost of a false positive is a refused answer rather
than a harmful one -- so the honest thing is to measure how often that happens.
``bench/run_guardrail_eval.py`` reports precision **and** recall against a suite
that deliberately includes benign queries containing alarming words, and the
false-positive rate is published rather than hidden.

On the ASR gate specifically -- what it does and does not do, measured rather than
assumed. Across 16 recorded clips in four languages, Sarvam returned
``language_probability: 1.00`` on every one, *including* the clips it
mis-transcribed ("chia seeds" -> "chair seeds"). A pure 440 Hz tone returned
``0.00``.

So this signal identifies **language**, not transcription quality. The gate
reliably rejects non-speech -- silence, an accidental mic tap, a recording that
captured nothing -- and it does not catch confident mis-hearing. The threshold is
set low (0.45) to match what it actually measures, and the defence against
mis-transcription lives in the UI instead: the transcript is shown above every
answer, so a user can see they were misheard and retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vrag.config import InputGuardCfg
from vrag.schemas import GuardVerdict, RefusalReason, Transcript

# --------------------------------------------------------------------------- #
# Prompt injection
# --------------------------------------------------------------------------- #
# Spoken injections are rarer than typed ones but not absent, and the transcript
# feeds an LLM on the polish path. Patterns cover English, Devanagari, Tamil and
# Bengali, plus romanised Hindi, which is how most people actually type/say it.
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+(?:instruction|prompt|rule|direction)",
    r"disregard\s+(?:all\s+)?(?:previous|prior|the\s+above)",
    r"forget\s+(?:everything|all\s+(?:previous|prior))",
    r"you\s+are\s+now\s+(?:a|an|in)\b",
    r"(?:reveal|show|print|repeat|output)\s+(?:me\s+)?(?:your\s+)?(?:system\s+prompt|instructions|initial\s+prompt)",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
    r"act\s+as\s+(?:if\s+you\s+(?:are|were)|a\s+)",
    r"pretend\s+(?:you\s+are|to\s+be)",
    r"</?(?:system|assistant|user)>",
    r"\bDAN\s+mode\b",
    r"पिछले?\s+(?:सभी\s+)?निर्देश(?:ों)?\s+(?:को\s+)?(?:अनदेखा|नज़रअंदाज़|भूल)",
    r"अपना\s+सिस्टम\s+प्रॉम्प्ट",
    r"முந்தைய\s+வழிமுறைகளை\s+புறக்கணி",
    r"পূর্ববর্তী\s+নির্দেশ(?:না)?\s+উপেক্ষা",
    r"pichle\s+(?:sabhi\s+)?nirdesh\s+(?:ko\s+)?(?:ignore|bhool)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.UNICODE)

# --------------------------------------------------------------------------- #
# Unsafe content
# --------------------------------------------------------------------------- #
# Scoped to *actionable harm* -- instructions for producing weapons, drugs,
# malware, or self-harm. Deliberately NOT a profanity list: swearing at a search
# system is rude, not dangerous, and refusing it is the kind of over-blocking the
# benchmark is designed to catch.

# A shared "how do I / how to / how can I / steps to" opener, because English
# question phrasing varies far more than the harmful verb does. Writing the
# opener once and reusing it is what stops the list from having holes -- the
# first version of this matched "how to synthesize" but not "how do I
# synthesize", which the unit tests caught.
_HOW = r"(?:how\s+(?:to|do\s+i|can\s+i|would\s+i|d(?:o|oes)\s+one)|steps?\s+to|guide\s+to|ways?\s+to)"

_UNSAFE_PATTERNS = [
    rf"{_HOW}\s+(?:make|build|synthesi[sz]e|manufacture|construct|produce|cook)\s+.{{0,30}}\b"
    r"(?:bomb|explosive|ied|nerve\s+agent|sarin|ricin|meth(?:amphetamine)?|napalm|thermite|"
    r"fentanyl|heroin|cocaine)\b",
    r"\b(?:make|build|synthesi[sz]e|manufacture)\s+(?:a\s+|some\s+)?(?:pipe\s+)?"
    r"(?:bomb|explosive|meth(?:amphetamine)?|nerve\s+agent|sarin|ricin)\b",
    rf"{_HOW}\s+(?:kill|murder|poison|stab|shoot)\s+"
    r"(?:someone|somebody|a\s+person|people|my|him|her|them)\b",
    rf"{_HOW}\s+(?:hack|ddos|breach|break\s+into)\s+.{{0,30}}\b"
    r"(?:account|bank|server|network|database|wifi|phone)\b",
    r"(?:write|create|generate|build)\s+(?:me\s+)?(?:a\s+)?"
    r"(?:ransomware|keylogger|malware|virus|trojan|botnet|rootkit)\b",
    rf"{_HOW}\s+(?:kill|hurt|harm)\s+myself\b",
    r"(?:best|easiest|painless)\s+way\s+to\s+(?:commit\s+suicide|kill\s+myself|end\s+my\s+life)\b",
    r"\bcommit\s+suicide\s+(?:painlessly|without\s+pain)\b",
    rf"{_HOW}\s+(?:buy|obtain|get|access)\s+.{{0,20}}\b(?:child\s+porn|csam)\b",
    # Devanagari
    r"बम\s+(?:कैसे\s+)?बनाने?\s+(?:का\s+तरीका|कैसे)",
    r"बम\s+कैसे\s+बना",
    r"किसी\s+को\s+(?:कैसे\s+)?(?:मारने?|जहर\s+देने)\s+क[ैा]",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE | re.UNICODE)

# --------------------------------------------------------------------------- #
# PII
# --------------------------------------------------------------------------- #
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("PHONE_IN", re.compile(r"(?:\+?91[\s-]?)?\b[6-9]\d{9}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("CARD", re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

_MEANINGFUL_RE = re.compile(r"[^\W_]", re.UNICODE)

# --------------------------------------------------------------------------- #
# Word boundaries in Indic scripts
# --------------------------------------------------------------------------- #
# DO NOT use a trailing ``\b`` after a Devanagari, Tamil or Bengali word.
#
# Python's ``re`` treats a character as a word character if ``str.isalnum()`` is
# true, and combining marks (Unicode category Mn) are not alphanumeric. Indic
# words routinely END in one: the anusvara in ``में``, the vowel sign in ``खाते``,
# ``है``, ``बनाने``. After such a character there is no word boundary at all, so a
# pattern ending in ``\b`` silently never matches -- no error, just a guard that
# quietly does nothing in one script while working fine in another.
#
# This cost a real bug: a Hindi private-state pattern that looked correct and
# matched nothing. Use ``_WORD_END`` instead, which accepts whitespace, end of
# string, or any of the terminators these scripts actually use.
_WORD_END = r"(?=[\s।॥?!.,;:\"')\]]|$)"

# --------------------------------------------------------------------------- #
# Unanswerable by construction
# --------------------------------------------------------------------------- #
# Questions that need PRIVATE state ("my bank balance", "my inbox") or SYSTEM
# state ("your system prompt", "your embedding model"). These are unanswerable
# from any static document corpus -- but the post-retrieval similarity guard
# cannot see that, because banking and email are perfectly ordinary MS MARCO
# topics and such a query scores a high cosine against real passages about them.
#
# Measured: these were 6 of the 7 out-of-domain cases the similarity guard missed.
# They are a closed, enumerable class, which is what makes a pattern the right
# tool here rather than a lazy one.
#
# Deliberately narrow. "what is a bank account" and "how do I open an email
# account" are ordinary corpus questions and must NOT match, so every pattern
# requires a possessive plus a state-bearing noun, not just the noun.
_PRIVATE_STATE_PATTERNS = [
    # The bounded gap matters: "what is my *current gps* location" has modifiers
    # between the possessive and the noun, and without it the pattern misses.
    r"\b(?:what|how\s+much|how\s+many|when|where)\s+(?:is|are|was|were)\s+(?:my|our)\s+"
    r"(?:\w+\s+){0,3}"
    r"(?:bank|account|balance|salary|password|address|location|order|booking|"
    r"subscription|appointment|calendar|inbox|email|phone|card)\b",
    # A bounded gap, because the object of the verb sits between: "read me
    # *the last email in* my inbox". Unbounded would over-match across clauses.
    r"\b(?:read|show|check|open|tell)\s+(?:me\s+)?(?:\w+\s+){0,5}(?:my|our)\s+"
    r"(?:inbox|email|messages?|calendar|balance|orders?|notifications?)\b",
    r"\b(?:what|when|where|who)\s+did\s+i\b",
    r"\bam\s+i\s+(?:logged\s+in|subscribed|registered)\b",
    # Devanagari: no trailing \b -- see the _WORD_END note above.
    r"मेरे?\s+(?:बैंक\s+)?(?:खाते|अकाउंट)\s+में" + _WORD_END,
    r"मेरा\s+(?:पासवर्ड|पता|बैलेंस)" + _WORD_END,
]
_SYSTEM_STATE_PATTERNS = [
    r"\byour\s+(?:system\s+prompt|initial\s+prompt|instructions|source\s+code|"
    r"training\s+data|weights|api\s+key)\b",
    r"\b(?:which|what)\s+(?:embedding\s+)?model\s+(?:are|do)\s+you\s+(?:running|using|use)\b",
    r"\bhow\s+many\s+(?:chunks|vectors|documents|passages)\s+(?:are\s+)?in\s+your\b",
    r"\byour\s+(?:rerank|retrieval|latency)\s+(?:budget|config|threshold)\b",
    r"\bwho\s+(?:wrote|built|trained)\s+your\s+(?:source\s+)?code\b",
]
_UNANSWERABLE_RE = re.compile(
    "|".join(_PRIVATE_STATE_PATTERNS + _SYSTEM_STATE_PATTERNS),
    re.IGNORECASE | re.UNICODE,
)


@dataclass
class InputGuard:
    cfg: InputGuardCfg

    def check(self, transcript: Transcript) -> GuardVerdict:
        text = transcript.text.strip()
        signals: dict[str, float] = {
            "asr_confidence": round(transcript.confidence, 4),
            "length": float(len(text)),
        }

        # 1. Empty / degenerate. Cheapest, and catches the common case of a user
        #    tapping the mic and saying nothing.
        if len(text) < self.cfg.min_chars or not _MEANINGFUL_RE.search(text):
            return GuardVerdict(
                allowed=False,
                reason=RefusalReason.EMPTY_INPUT,
                detail="No speech detected. Please hold the mic and ask your question.",
                signals=signals,
            )

        if len(text) > self.cfg.max_chars:
            return GuardVerdict(
                allowed=False,
                reason=RefusalReason.INPUT_TOO_LONG,
                detail=f"Question is {len(text)} characters; the limit is {self.cfg.max_chars}.",
                signals=signals,
            )

        # 2. ASR confidence. See the module docstring -- a fluent mis-transcription
        #    is the failure mode with no downstream defence.
        if transcript.provider != "text" and transcript.confidence < self.cfg.min_asr_confidence:
            return GuardVerdict(
                allowed=False,
                reason=RefusalReason.LOW_CONFIDENCE_ASR,
                detail=(
                    f"I heard “{text}” but I am only "
                    f"{transcript.confidence:.0%} confident. Could you repeat that?"
                ),
                signals=signals,
            )

        # 3. Unsafe content.
        if self.cfg.block_unsafe:
            match = _UNSAFE_RE.search(text)
            if match:
                signals["unsafe_match"] = 1.0
                return GuardVerdict(
                    allowed=False,
                    reason=RefusalReason.UNSAFE_INPUT,
                    detail="I can't help with that request.",
                    signals=signals,
                )

        # 4. Prompt injection.
        if self.cfg.block_injection:
            match = _INJECTION_RE.search(text)
            if match:
                signals["injection_match"] = 1.0
                return GuardVerdict(
                    allowed=False,
                    reason=RefusalReason.INJECTION_ATTEMPT,
                    detail="I only answer questions about the indexed corpus.",
                    signals=signals,
                )

        # 5. Unanswerable by construction: needs private or system state, not
        #    documents. Checked here rather than after retrieval because the
        #    similarity guard provably cannot see it -- "what is my bank balance"
        #    scores a high cosine against genuine passages about banking.
        match = _UNANSWERABLE_RE.search(text)
        if match:
            signals["unanswerable_match"] = 1.0
            return GuardVerdict(
                allowed=False,
                reason=RefusalReason.OUT_OF_DOMAIN,
                detail=(
                    "I can only answer from an indexed corpus of web passages. "
                    "I have no access to your personal accounts, and I can't report "
                    "on my own configuration."
                ),
                signals=signals,
            )

        # 6. PII redaction. Not a refusal -- the question is still answered, but
        #    the logged and traced copy is scrubbed, so telemetry never becomes a
        #    place personal data accumulates.
        redacted = self.redact(text) if self.cfg.redact_pii else None
        if redacted is not None and redacted != text:
            signals["pii_redacted"] = 1.0

        return GuardVerdict(allowed=True, signals=signals, redacted_text=redacted)

    @staticmethod
    def redact(text: str) -> str:
        for label, pattern in _PII_PATTERNS:
            text = pattern.sub(f"[{label}]", text)
        return text
