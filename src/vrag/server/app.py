"""FastAPI application.

Three principles:

* **The pipeline is built once, at startup.** Everything expensive -- ONNX
  sessions, the FAISS index, the mmapped store, the reranker -- loads during the
  lifespan hook. Nothing on the request path opens a file or allocates a model.
* **Every endpoint returns an ``AnswerEnvelope``.** Refusals, timeouts, and
  crashes all come back as HTTP 200 with ``abstained: true`` and a typed
  ``refusal_reason``. A guardrail declining to answer is a normal outcome, not an
  error, and modelling it as a 4xx would make it indistinguishable from a bug.
* **Timings ship with every answer.** The response carries per-stage latency, the
  budget verdict, and any degradations, so the UI can render the live HUD without
  a second round trip -- and so the demo can show the SLA being enforced rather
  than asserted.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vrag.config import get_config, get_secrets
from vrag.harness.budget import GLOBAL_REGISTRY
from vrag.harness.pipeline import Pipeline
from vrag.schemas import AnswerEnvelope, AudioInput

STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    text: str = Field(..., description="The question, as text.")
    lang: str = Field("unknown", description="ISO-639-1 hint; 'unknown' to skip filtering.")
    polish: bool = Field(False, description="Also run the optional LLM polish path.")


class Health(BaseModel):
    status: str
    chunks: int
    views: list[str]
    reranker: bool
    sparse: bool
    stt_configured: bool
    llm_configured: bool


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    cfg = get_config()
    app.state.cfg = cfg
    app.state.boot_started = time.perf_counter()
    app.state.pipeline = Pipeline(cfg)
    app.state.boot_seconds = round(time.perf_counter() - app.state.boot_started, 2)

    if cfg.server.warm_on_boot:
        # Warm with a BATCH, not a single request. Two different things need
        # warming and only one of them is fixed by one call:
        #
        #   1. ONNX graph optimization, arena allocation, page-faulting the mmapped
        #      store -- one request covers this.
        #   2. The budget manager's per-stage statistics. It needs enough samples
        #      to estimate a p90; below that it falls back to configured guesses,
        #      and those guesses are what let an early request overrun the budget.
        #      This was observed: request 3 came in at 282ms against a 200ms budget.
        #
        # Warming across languages also builds each language-filter id array, so no
        # user request pays for one.
        warm_queries = [
            ("what is a corporation", "en"),
            ("how long does digestion take", "en"),
            ("कॉर्पोरेशन क्या है?", "hi"),
            ("सामुदायिक कानूनी सेवा क्या है", "hi"),
            ("நிறுவனம் என்றால் என்ன?", "ta"),
            ("பிரசவம் என்றால் என்ன?", "ta"),
            ("কর্পোরেশন কি?", "bn"),
            ("দ্রুততম বাইক চাকা", "bn"),
        ]
        warm_start = time.perf_counter()
        for _ in range(cfg.server.warm_rounds):
            for text, lang in warm_queries:
                app.state.pipeline.answer_text(text, lang=lang)
        app.state.warm_seconds = round(time.perf_counter() - warm_start, 2)
        app.state.warm_requests = cfg.server.warm_rounds * len(warm_queries)
    else:
        app.state.warm_seconds = 0.0
        app.state.warm_requests = 0

    yield
    app.state.pipeline.close()


def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(
        title="vrag",
        description="Voice-enabled multilingual RAG with a latency-budgeted retrieval core.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------------------- #
    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        p: Pipeline = app.state.pipeline
        secrets = get_secrets()
        return Health(
            status="ok",
            chunks=len(p.store),
            views=[v.value for v in p.retriever.enabled_views],
            reranker=p.reranker is not None,
            sparse=p.sparse is not None,
            stt_configured=secrets.has_stt,
            llm_configured=secrets.has_llm,
        )

    @app.post("/api/ask", response_model=AnswerEnvelope)
    def ask(request: AskRequest) -> AnswerEnvelope:
        """Text path. Also what the benchmarks drive."""
        p: Pipeline = app.state.pipeline
        return p.answer_text(request.text, lang=request.lang)

    @app.post("/api/voice", response_model=AnswerEnvelope)
    async def voice(
        audio: UploadFile = File(...),
        lang_hint: str | None = Form(None),
    ) -> AnswerEnvelope:
        """Voice path: audio in, grounded answer out."""
        cfg = app.state.cfg
        raw = await audio.read()

        if not raw:
            raise HTTPException(status_code=400, detail="empty audio upload")
        if len(raw) > cfg.server.max_audio_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"audio is {len(raw)} bytes; limit is {cfg.server.max_audio_bytes}",
            )

        p: Pipeline = app.state.pipeline
        return await p.answer_audio(
            AudioInput(
                audio=raw,
                filename=audio.filename or "audio.wav",
                mime_type=audio.content_type or "audio/wav",
                lang_hint=lang_hint if lang_hint not in ("", "auto") else None,
            )
        )

    @app.get("/api/metrics")
    def metrics() -> JSONResponse:
        """Live per-stage percentiles from the running process.

        Same numbers the benchmark reports, read from the same span log -- so the
        deployed instance can be checked against the published table rather than
        taken on trust.
        """
        p: Pipeline = app.state.pipeline
        cfg = app.state.cfg
        return JSONResponse(
            {
                "boot_seconds": app.state.boot_seconds,
                "warm_seconds": getattr(app.state, "warm_seconds", 0.0),
                "warm_requests": getattr(app.state, "warm_requests", 0),
                # False means the budget manager is still estimating stage costs
                # from config rather than measurement, so early latencies are not
                # representative. Exposed rather than hidden.
                "budget_warm": GLOBAL_REGISTRY.warm,
                "budget_ms": cfg.budget.core_budget_ms,
                "stages": GLOBAL_REGISTRY.snapshot(),
                "spans": p.tracer.percentiles(cfg.bench.latency.percentiles),
            }
        )

    # ----------------------------------------------------------------------- #
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
