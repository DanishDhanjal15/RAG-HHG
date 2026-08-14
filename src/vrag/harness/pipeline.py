"""The orchestrator -- an explicit, timed, budgeted stage graph.

This is deliberately not a LangChain ``Chain``. The requirement is to demonstrate
structured orchestration, and a chain abstraction hides exactly the seams that
matter here: where the clock is, which stages are droppable, what happens when one
throws, and how a partial result still becomes a valid response. Those seams *are*
the harness.

Properties the runner enforces, none of which are optional:

* **One response type.** Every path -- success, refusal, timeout, crash, missing
  dependency -- returns an ``AnswerEnvelope``. There is no other shape the API can
  emit and no unhandled exception can escape ``answer()``.
* **Every stage is timed and traced.** The span log is the latency dataset; the
  benchmark reads it back rather than measuring separately.
* **Optional stages are budget-gated.** Before each one the manager checks its
  measured rolling p90 against the remaining budget and skips it if it will not
  fit, recording a ``Degradation``.
* **Guardrails short-circuit.** A failed guard returns immediately. Retrieval for
  an unsafe query is work we should not do, not work we should do and discard.
* **The core clock excludes STT.** ``core_latency_ms`` covers embed → answer, which
  is the <200 ms contract; ``total_latency_ms`` includes the network round trip to
  Sarvam. Both are reported, always, so the number under the target is never
  mistaken for the number the user experiences.
"""

from __future__ import annotations

import time
import uuid

from vrag.config import Config, Secrets, get_config, get_secrets
from vrag.generate.extractive import ExtractiveGenerator
from vrag.guardrails.domain_guard import DomainGuard
from vrag.guardrails.grounding import ConflictDetector, GroundingGuard
from vrag.guardrails.input_guard import InputGuard
from vrag.guardrails.policy import apply_refusal
from vrag.harness.budget import GLOBAL_REGISTRY, LatencyBudget
from vrag.harness.resilience import SttUnavailable
from vrag.harness.tracing import RequestTrace, Tracer
from vrag.index.dense import DenseIndex
from vrag.index.embedder import OnnxEmbedder
from vrag.index.sparse import SparseIndex
from vrag.index.store import ChunkStore
from vrag.retrieve.expand import plan_query
from vrag.retrieve.multiview import MultiViewRetriever
from vrag.retrieve.rerank import CrossEncoderReranker
from vrag.schemas import (
    AnswerEnvelope,
    AudioInput,
    RankedContext,
    RefusalReason,
    Span,
    Transcript,
)


class Pipeline:
    """Loads every component once and serves requests.

    Construction is expensive (ONNX sessions, a FAISS index, a BM25 index, an
    mmapped store) and happens exactly once at boot. Nothing on the request path
    loads, allocates, or opens a file.
    """

    def __init__(self, cfg: Config | None = None, secrets: Secrets | None = None) -> None:
        self.cfg = cfg or get_config()
        self.secrets = secrets or get_secrets()

        self.embedder = OnnxEmbedder(self.cfg)
        self.store = ChunkStore(self.cfg.paths.index_dir / "chunks")
        self.dense = DenseIndex.load(self.cfg)
        self.sparse: SparseIndex | None = None
        if self.cfg.sparse.enabled:
            try:
                self.sparse = SparseIndex.load(self.cfg)
            except FileNotFoundError:
                self.sparse = None

        self.retriever = MultiViewRetriever(
            self.cfg, self.embedder, self.dense, self.store, self.sparse
        )

        # The reranker is the one component allowed to be absent: it is optional
        # by budget anyway, so a failed load degrades the same way a slow request
        # does, through the same reporting path.
        self.reranker: CrossEncoderReranker | None = None
        if self.cfg.rerank.enabled:
            try:
                self.reranker = CrossEncoderReranker(self.cfg)
            except Exception:  # noqa: BLE001
                self.reranker = None

        self.generator = ExtractiveGenerator(self.cfg.generation.extractive, self.embedder)

        g = self.cfg.guardrails
        self.input_guard = InputGuard(g.input)
        self.domain_guard = DomainGuard(g.domain)
        self.grounding_guard = GroundingGuard(g.grounding, self.embedder)
        self.conflict = ConflictDetector(
            min_disagreement=g.conflict.min_disagreement,
            min_topical_overlap=g.conflict.min_topical_overlap,
            enabled=g.conflict.enabled,
        )

        self.tracer = Tracer(self.cfg.paths.trace_dir / "spans.sqlite")
        self._stt = None  # lazily constructed; text queries never need it

    # ----------------------------------------------------------------------- #
    # Entry points
    # ----------------------------------------------------------------------- #
    async def answer_audio(self, audio: AudioInput) -> AnswerEnvelope:
        """Voice path: transcribe, then run the core."""
        request_id = uuid.uuid4().hex[:12]
        trace = RequestTrace(request_id)
        wall_start = time.perf_counter()

        envelope = AnswerEnvelope(request_id=request_id)

        try:
            with trace.span("stt") as span:
                transcript = await self._transcribe(audio)
                span.attributes.update(
                    provider=transcript.provider,
                    lang=transcript.lang,
                    confidence=round(transcript.confidence, 4),
                    chars=len(transcript.text),
                )
        except SttUnavailable as exc:
            envelope.total_latency_ms = round((time.perf_counter() - wall_start) * 1000, 3)
            envelope.timings_ms = trace.timings
            envelope.spans = trace.spans
            envelope.refused(RefusalReason.STT_UNAVAILABLE, str(exc))
            envelope.answer = (
                "Speech recognition is unavailable right now. You can type your question instead."
            )
            self.tracer.record(trace, envelope)
            return envelope
        except Exception as exc:  # noqa: BLE001
            envelope.total_latency_ms = round((time.perf_counter() - wall_start) * 1000, 3)
            envelope.refused(RefusalReason.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
            self.tracer.record(trace, envelope)
            return envelope

        return self._run_core(transcript, trace, envelope, wall_start)

    def answer_text(self, text: str, lang: str = "unknown") -> AnswerEnvelope:
        """Text path. Also the benchmark's entry point -- thousands of latency
        samples without paying for thousands of STT calls."""
        request_id = uuid.uuid4().hex[:12]
        trace = RequestTrace(request_id)
        transcript = Transcript(text=text, lang=lang, confidence=1.0, provider="text")
        return self._run_core(
            transcript, trace, AnswerEnvelope(request_id=request_id), time.perf_counter()
        )

    # ----------------------------------------------------------------------- #
    # The <200ms core
    # ----------------------------------------------------------------------- #
    def _run_core(
        self,
        transcript: Transcript,
        trace: RequestTrace,
        envelope: AnswerEnvelope,
        wall_start: float,
    ) -> AnswerEnvelope:
        budget = LatencyBudget(self.cfg.budget, GLOBAL_REGISTRY)
        core_start = time.perf_counter()

        envelope.transcript = transcript.text
        envelope.detected_lang = transcript.lang
        envelope.asr_confidence = round(transcript.confidence, 4)

        try:
            # -- guard: input ------------------------------------------------ #
            with trace.span("input_guard") as span:
                verdict = self.input_guard.check(transcript)
                span.attributes.update(verdict.signals)
                span.attributes["allowed"] = verdict.allowed
            if not verdict.allowed:
                return self._finish(
                    apply_refusal(envelope, verdict, RefusalReason.EMPTY_INPUT),
                    trace, budget, core_start, wall_start,
                )

            plan = plan_query(self.cfg, transcript)

            # -- embed ------------------------------------------------------- #
            with trace.span("embed_query") as span:
                t0 = time.perf_counter()
                vector = self.retriever.embed_query(plan)
                budget.record("embed_query", (time.perf_counter() - t0) * 1000)
                span.attributes["dim"] = int(vector.shape[0])

            # -- retrieve: dense --------------------------------------------- #
            with trace.span("dense_search") as span:
                t0 = time.perf_counter()
                runs = self.retriever.dense_search(vector, plan)
                budget.record("dense_search", (time.perf_counter() - t0) * 1000)
                span.attributes.update(
                    runs=len(runs), hits=sum(len(v) for v in runs.values()),
                    lang_filtered=bool(plan.lang_filter),
                )

            # -- retrieve: sparse (optional) --------------------------------- #
            if self.sparse is not None and budget.should_run("sparse_search"):
                with trace.span("sparse_search") as span:
                    t0 = time.perf_counter()
                    sparse_runs = self.retriever.sparse_search(plan)
                    budget.record("sparse_search", (time.perf_counter() - t0) * 1000)
                    span.attributes["hits"] = sum(len(v) for v in sparse_runs.values())
                runs.update(sparse_runs)
            else:
                trace.skipped("sparse_search", "budget" if self.sparse else "not_built")

            # -- fuse -------------------------------------------------------- #
            with trace.span("fuse") as span:
                t0 = time.perf_counter()
                candidates = self.retriever.fuse(runs)
                budget.record("fuse", (time.perf_counter() - t0) * 1000)
                span.attributes.update(
                    candidates=len(candidates),
                    multi_view=sum(1 for c in candidates if c.views_hit > 1),
                )

            evidence = self.retriever.hydrate(candidates)

            # -- rerank (optional) ------------------------------------------- #
            reranked = False
            if self.reranker is not None and budget.should_run("rerank"):
                with trace.span("rerank") as span:
                    t0 = time.perf_counter()
                    evidence = self.reranker.rerank(plan.normalized_query, evidence)
                    budget.record("rerank", (time.perf_counter() - t0) * 1000)
                    span.attributes["n"] = len(evidence)
                reranked = True
            else:
                trace.skipped("rerank", "budget" if self.reranker else "unavailable")

            context = RankedContext(
                evidence=evidence,
                top1_score=self.retriever.cosine_top1(plan),
                centroid_distance=self.retriever.centroid_distance(vector),
                reranked=reranked,
            )

            # -- guard: domain ----------------------------------------------- #
            with trace.span("domain_guard") as span:
                t0 = time.perf_counter()
                domain_verdict = self.domain_guard.check(context)
                budget.record("domain_guard", (time.perf_counter() - t0) * 1000)
                span.attributes.update(domain_verdict.signals)
                span.attributes["allowed"] = domain_verdict.allowed
            if not domain_verdict.allowed:
                return self._finish(
                    apply_refusal(envelope, domain_verdict, RefusalReason.OUT_OF_DOMAIN),
                    trace, budget, core_start, wall_start, evidence,
                )

            # -- guard: conflicting evidence --------------------------------- #
            conflicted, conflict_signals = self.conflict.check(evidence)
            if conflicted:
                trace.skipped("generate", "conflicting_evidence")
                envelope.citations = [
                    self._as_citation(e) for e in evidence[:3]
                ]
                envelope.refused(
                    RefusalReason.CONFLICTING_EVIDENCE,
                    "The retrieved sources give different figures, so I'm showing them "
                    "rather than choosing one.",
                )
                envelope.answer = envelope.refusal_detail
                envelope.timings_ms.update(conflict_signals)
                return self._finish(envelope, trace, budget, core_start, wall_start, evidence)

            # -- generate ----------------------------------------------------- #
            # Semantic sentence re-scoring is the expensive half of extraction --
            # measured at 65-127ms on this CPU versus ~1ms for lexical selection --
            # so it is budgeted as its own optional stage rather than hidden inside
            # `generate`. Dropping it still yields a grounded, cited answer; it just
            # loses paraphrase sensitivity.
            semantic = budget.should_run("generate_semantic")

            t0 = time.perf_counter()
            answer, semantic_ms = self.generator.generate(
                plan.normalized_query,
                evidence,
                query_lang=plan.lang,
                semantic_scoring=semantic,
                query_vector=vector,
            )
            total_ms = (time.perf_counter() - t0) * 1000
            lexical_ms = total_ms - semantic_ms

            # Two spans, not one. The budget manager treats these as separate
            # stages, so the trace has to as well -- a single `generate` span
            # covering both would report the lexical stage at the combined cost
            # and make the latency table disagree with the thing enforcing the
            # budget.
            budget.record("generate", lexical_ms)
            trace.add(Span(
                name="generate", duration_ms=lexical_ms,
                attributes={"strategy": answer.strategy, "chars": len(answer.text)},
            ))

            if semantic:
                budget.record("generate_semantic", semantic_ms)
                trace.add(Span(
                    name="generate_semantic", duration_ms=semantic_ms,
                    attributes={"candidates_rescored": True},
                ))
            else:
                trace.skipped("generate_semantic", "budget")

            # -- guard: grounding (optional tier) ----------------------------- #
            run_grounding = budget.should_run("grounding_guard")
            if run_grounding:
                with trace.span("grounding_guard") as span:
                    t0 = time.perf_counter()
                    grounding_verdict = self.grounding_guard.check(answer, evidence)
                    budget.record("grounding_guard", (time.perf_counter() - t0) * 1000)
                    span.attributes.update(grounding_verdict.signals)
                    span.attributes["allowed"] = grounding_verdict.allowed
                if not grounding_verdict.allowed:
                    return self._finish(
                        apply_refusal(envelope, grounding_verdict, RefusalReason.NO_GROUNDING),
                        trace, budget, core_start, wall_start, evidence,
                    )
            else:
                trace.skipped("grounding_guard", "budget")
                # Not silently unchecked: the extractive path is grounded by
                # construction, but the caller is told the verification did not run.
                budget.note_degradation("grounding_guard", "attribution check skipped")

            if not answer.text:
                return self._finish(
                    envelope.refused(
                        RefusalReason.NO_GROUNDING,
                        "I found related passages but none of them answer that question.",
                    ),
                    trace, budget, core_start, wall_start, evidence,
                )

            envelope.answer = answer.text
            envelope.citations = answer.citations
            envelope.strategy = answer.strategy
            envelope.confidence = round(
                answer.confidence * self.domain_guard.confidence_penalty(domain_verdict), 4
            )
            return self._finish(envelope, trace, budget, core_start, wall_start, evidence)

        except Exception as exc:  # noqa: BLE001 -- the last line of defence
            envelope.refused(RefusalReason.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
            envelope.answer = "Something went wrong on our side."
            return self._finish(envelope, trace, budget, core_start, wall_start)

    # ----------------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------------- #
    def _finish(
        self,
        envelope: AnswerEnvelope,
        trace: RequestTrace,
        budget: LatencyBudget,
        core_start: float,
        wall_start: float,
        evidence: list | None = None,
    ) -> AnswerEnvelope:
        envelope.core_latency_ms = round((time.perf_counter() - core_start) * 1000, 3)
        envelope.total_latency_ms = round((time.perf_counter() - wall_start) * 1000, 3)
        envelope.within_budget = envelope.core_latency_ms <= self.cfg.budget.core_budget_ms
        envelope.degradations = budget.degradations
        envelope.timings_ms = {**budget.timings, **trace.timings}
        envelope.spans = trace.spans

        if evidence and not envelope.citations:
            envelope.citations = [self._as_citation(e) for e in evidence[:3]]

        self.tracer.record(trace, envelope)
        return envelope

    @staticmethod
    def _as_citation(evidence):  # noqa: ANN001, ANN205
        from vrag.schemas import Citation

        return Citation(
            chunk_id=evidence.chunk_id,
            doc_id=evidence.doc_id,
            lang=evidence.lang,
            quote=evidence.text[:240],
            score=round(float(evidence.rerank_score or evidence.score), 4),
        )

    async def _transcribe(self, audio: AudioInput) -> Transcript:
        if self._stt is None:
            from vrag.stt import build_provider

            self._stt = build_provider(self.cfg, self.secrets)
        return await self._stt.transcribe(audio)

    def close(self) -> None:
        self.tracer.close()
