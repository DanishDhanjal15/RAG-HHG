# Voice-RAG -- deployable to Hugging Face Spaces (Docker SDK) or any container host.
#
# Two-stage so the ~2 GB of build tooling (torch, optimum, the ONNX exporter) never
# reaches the runtime image. Torch is needed ONLY to export the encoders to ONNX;
# at serve time the pipeline runs on onnxruntime alone, and shipping torch would
# roughly quadruple the image for code that is never called.
#
# The index is expected to be built beforehand and mounted or copied in -- see
# README. Building it inside the image would make every deploy a multi-hour job.

# ─── stage 1: export the ONNX encoders ──────────────────────────────────────
FROM python:3.11-slim AS exporter

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /export

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
 && pip install "optimum[onnxruntime]>=1.20" "transformers>=4.40,<5.0" onnx

COPY scripts/export_models.py .
RUN python export_models.py --out /models


# ─── stage 2: runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VRAG_CONFIG=configs/default.yaml \
    # ORT reads these before our session options exist; pin them low so a
    # container with 16 visible cores does not spawn 16 threads per session.
    OMP_NUM_THREADS=4 \
    ORT_DISABLE_ALL_TELEMETRY=1

# HF Spaces runs the container as uid 1000 and expects the app on port 7860.
RUN useradd -m -u 1000 app
WORKDIR /app

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src

# Runtime deps only -- no torch, no optimum, no onnx exporter.
RUN pip install --no-cache-dir \
      "numpy>=1.26,<2.0" "scipy>=1.11" "pyarrow>=16.0" "pandas>=2.1" \
      "faiss-cpu>=1.8.0" "bm25s>=0.2.0" "PyStemmer>=2.2.0" \
      "onnxruntime>=1.18" "tokenizers>=0.19" "transformers>=4.40,<5.0" \
      "huggingface-hub>=0.23" \
      "fastapi>=0.111" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" \
      "httpx>=0.27" "pydantic>=2.7" "pydantic-settings>=2.3" "pyyaml>=6.0" \
      "anthropic>=0.34" "orjson>=3.10" "rich>=13.7" "typer>=0.12" "python-dotenv>=1.0" \
 && pip install --no-cache-dir --no-deps -e .

COPY --chown=app:app configs ./configs
COPY --from=exporter --chown=app:app /models ./models

# The index is NOT baked into the image. It is ~170 MB of vectors and payload with
# its own lifecycle, so it lives in a Hugging Face dataset repo and is fetched on
# first boot (see vrag/index/fetch.py). That keeps the image small, keeps a code
# push from re-uploading the index, and keeps deploys from taking hours.
#
#   Hosted:  set remote_index.repo_id in configs/default.yaml (or VRAG_ config)
#   Local:   docker run -v "$(pwd)/data/index:/app/data/index:ro" ...
RUN mkdir -p /app/data/index /app/traces && chown -R app:app /app/data /app/traces

USER app

# HF Spaces expects 7860; Cloud Run injects its own $PORT and ignores EXPOSE.
# Defaulting the variable keeps one image working on both.
ENV PORT=7860
EXPOSE 7860

# start-period is generous on purpose: a cold start downloads the index and then
# warms ONNX and the budget manager before it will answer.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','7860'); sys.exit(0 if urllib.request.urlopen(f'http://localhost:{p}/api/health',timeout=4).status==200 else 1)"

# Shell form so $PORT expands. Cloud Run sets it; HF Spaces does not, hence the
# ENV default above.
CMD exec uvicorn vrag.server.app:app --host 0.0.0.0 --port ${PORT}
