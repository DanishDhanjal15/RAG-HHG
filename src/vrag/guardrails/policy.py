"""Refusal policy -- what each abstention means and what the user should do next.

Guardrails decide *whether* to answer. This module owns what happens when the
answer is no: the user-facing message, the recovery action the UI should offer,
and the category the metrics count it under.

Keeping this separate matters because the three refusal categories deserve very
different treatment, and collapsing them into one "I can't help with that" is a
bad product and bad telemetry:

* ``RECOVERABLE`` -- the user can fix it by acting (speak again, ask something in
  scope). The UI should offer that action directly.
* ``REFUSED`` -- we decline on purpose. No recovery is offered, and none should be.
* ``DEGRADED`` -- *we* failed, not the user. Says so plainly, and counts against
  the system in the metrics rather than against the query.

Over-refusal shows up as a rising ``RECOVERABLE`` rate on in-domain traffic, which
is why the category is recorded on every abstention.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vrag.schemas import AnswerEnvelope, GuardVerdict, RefusalReason


class RefusalClass(StrEnum):
    RECOVERABLE = "recoverable"
    REFUSED = "refused"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class RefusalSpec:
    klass: RefusalClass
    message: str
    action: str | None = None


POLICY: dict[RefusalReason, RefusalSpec] = {
    RefusalReason.EMPTY_INPUT: RefusalSpec(
        RefusalClass.RECOVERABLE,
        "I didn't catch a question.",
        "retry_recording",
    ),
    RefusalReason.LOW_CONFIDENCE_ASR: RefusalSpec(
        RefusalClass.RECOVERABLE,
        "I'm not confident I heard that correctly.",
        "retry_recording",
    ),
    RefusalReason.INPUT_TOO_LONG: RefusalSpec(
        RefusalClass.RECOVERABLE,
        "That question is too long. Try asking one thing at a time.",
        "shorten_query",
    ),
    RefusalReason.OUT_OF_DOMAIN: RefusalSpec(
        RefusalClass.RECOVERABLE,
        "That isn't covered by this corpus.",
        "show_corpus_scope",
    ),
    RefusalReason.NO_GROUNDING: RefusalSpec(
        RefusalClass.RECOVERABLE,
        "I found related passages but can't support an answer from them.",
        "show_retrieved",
    ),
    RefusalReason.CONFLICTING_EVIDENCE: RefusalSpec(
        RefusalClass.RECOVERABLE,
        "The sources disagree, so I'm showing both rather than picking one.",
        "show_retrieved",
    ),
    RefusalReason.UNSAFE_INPUT: RefusalSpec(
        RefusalClass.REFUSED,
        "I can't help with that request.",
        None,
    ),
    RefusalReason.INJECTION_ATTEMPT: RefusalSpec(
        RefusalClass.REFUSED,
        "I only answer questions about the indexed corpus.",
        None,
    ),
    RefusalReason.STT_UNAVAILABLE: RefusalSpec(
        RefusalClass.DEGRADED,
        "Speech recognition is unavailable right now. You can type your question instead.",
        "use_text_input",
    ),
    RefusalReason.INTERNAL_ERROR: RefusalSpec(
        RefusalClass.DEGRADED,
        "Something went wrong on our side.",
        "retry",
    ),
}


def spec_for(reason: RefusalReason) -> RefusalSpec:
    return POLICY.get(
        reason, RefusalSpec(RefusalClass.DEGRADED, "I can't answer that right now.", "retry")
    )


def apply_refusal(
    envelope: AnswerEnvelope, verdict: GuardVerdict, fallback: RefusalReason
) -> AnswerEnvelope:
    """Write a guard verdict into the envelope as a fully-formed refusal.

    Uses the guard's specific ``detail`` when it has one -- "I heard 'X' but I'm
    only 40% confident" is far more useful than the generic category message --
    and falls back to the policy text otherwise.
    """
    reason = verdict.reason or fallback
    spec = spec_for(reason)

    envelope.refused(reason, verdict.detail or spec.message)
    envelope.answer = verdict.detail or spec.message
    envelope.confidence = 0.0
    return envelope


def refusal_metadata(reason: RefusalReason) -> dict[str, str | None]:
    spec = spec_for(reason)
    return {"class": spec.klass.value, "action": spec.action}
