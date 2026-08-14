"""Layer 2 -- out-of-domain detection, run after retrieval and before generation.

A retriever always returns something. Ask a corpus of MS MARCO web passages "what
is the capital of Mars" and it will hand back its ten least-bad chunks with a
straight face; a generator handed those chunks will write a fluent answer built
from irrelevant text. Knowing when the corpus simply does not contain the answer
is a separate problem from retrieving well, and it needs its own signal.

We use **two independent signals and require both to pass**, because each has a
failure mode the other covers:

* **Top-1 cosine similarity.** Directly measures "is the best chunk actually
  similar to the question". Fails on a query that happens to share vocabulary
  with one unrelated passage -- a single lucky match scores high.
* **Distance from the corpus centroid.** Measures "is this question even in the
  neighbourhood of what this corpus is about". Fails on an in-domain question
  phrased unusually, which drifts from the centroid while still having a genuine
  answer.

Requiring both means a lucky lexical collision (high top-1, far from centroid) and
an off-topic question near the centroid by coincidence (low top-1) are each
caught. Requiring *either* would abstain far too often.

Thresholds are **calibrated, not chosen**: ``vrag calibrate`` sweeps them against
real in-domain queries and a labelled out-of-domain set, then writes the values
that maximise F1 into the config. A hand-picked cosine threshold is a guess about
a distribution you have not looked at.
"""

from __future__ import annotations

from dataclasses import dataclass

from vrag.config import DomainGuardCfg
from vrag.schemas import GuardVerdict, RankedContext, RefusalReason


@dataclass
class DomainGuard:
    cfg: DomainGuardCfg

    def check(self, context: RankedContext) -> GuardVerdict:
        signals = {
            "top1_score": round(context.top1_score, 4),
            "centroid_distance": round(context.centroid_distance, 4),
            "n_evidence": float(len(context.evidence)),
        }

        if len(context.evidence) < self.cfg.min_supporting_chunks:
            return GuardVerdict(
                allowed=False,
                reason=RefusalReason.OUT_OF_DOMAIN,
                detail="I could not find anything about that in this corpus.",
                signals=signals,
            )

        weak_match = context.top1_score < self.cfg.min_top1_score
        # When the centroid signal is disabled it must not act as a second
        # condition -- an always-true conjunct would silently make the guard
        # cosine-only anyway, but with a threshold nobody calibrated.
        far_from_corpus = (
            context.centroid_distance > self.cfg.max_centroid_distance
            if self.cfg.use_centroid
            else True
        )

        if weak_match and far_from_corpus:
            return GuardVerdict(
                allowed=False,
                reason=RefusalReason.OUT_OF_DOMAIN,
                detail=(
                    "That question doesn't appear to be covered by this corpus. "
                    "It indexes MS MARCO web passages in Hindi, Tamil, Bengali and English."
                ),
                signals=signals,
            )

        # Allowed, but flag the ones that only just cleared the bar. The penalty
        # propagates into the answer envelope, so a marginal answer is visibly
        # marginal rather than silently returned with full confidence.
        #
        # What counts as marginal depends on how many signals are live. With the
        # centroid signal disabled there is no "one of two failed" case, so
        # "borderline" means the cosine is only just above the threshold.
        if self.cfg.use_centroid:
            borderline = weak_match or far_from_corpus
        else:
            headroom = context.top1_score - self.cfg.min_top1_score
            borderline = headroom < self.cfg.borderline_margin

        if borderline:
            signals["borderline"] = 1.0

        return GuardVerdict(allowed=True, signals=signals)

    def confidence_penalty(self, verdict: GuardVerdict) -> float:
        return 0.6 if verdict.signals.get("borderline") else 1.0
