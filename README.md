# वाणी-RAG · Voice-Enabled Multilingual RAG

Voice in, **grounded answer or an honest refusal** out — over
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
in Hindi, Tamil, Bengali and English.

Built for **HH Goa 2026 Shortlisting Task 2**.

```
🎙 → Sarvam STT → guardrails → 4-view retrieval → grounded answer + citations
                  └────── this part runs in <200 ms, enforced ──────┘
```

---

## The short version

| Requirement | How it's met |
|---|---|
| **1. Speech-to-text** | Sarvam `saaras:v4`, behind a swappable `SttProvider` protocol. Its `language_probability` is reused as an ASR confidence gate. |
| **2. Chunking** | **Five views** — atomic · sentence-window · token-fixed-overlap · semantic-breakpoint · LLM-proposition — co-indexed and RRF-fused, with a **published ablation that chose the config**. → [`docs/CHUNKING.md`](docs/CHUNKING.md) |
| **3. Latency** | <200 ms for the retrieval core, **enforced** by a budget manager that drops optional stages rather than overrunning. |
| **4. Analytics** | P50/P70/P90/P95/P99/P100 per stage, measured from the spans the pipeline emits while serving. → [`docs/LATENCY.md`](docs/LATENCY.md) |
| **5. Harness** | Typed stage graph, per-stage budgets, retries + circuit breakers, tool-calling generation, full span tracing. |
| **6. Guardrails** | Three layers, typed refusal enum, and an adversarial suite that **also measures over-refusal**. → [`docs/GUARDRAILS.md`](docs/GUARDRAILS.md) |

---

## What's actually interesting here

**The 200 ms number is honest about what it covers.** A cloud STT round trip is
~600 ms and cannot fit in a 200 ms budget on any hardware. So the contract is
scoped to the part the system controls — embed → retrieve → fuse → rerank →
guard → answer — and STT is measured and reported *separately* rather than
averaged in. Both numbers ship with every response.

**The budget is enforced, not hoped for.** Each stage is declared required or
optional. Before an optional stage runs, the manager compares its **measured
rolling p90** against the remaining budget and skips it if it won't fit,
recording a `degradation` in the response. A fast answer is never *silently* a
worse answer — the UI renders every skip.

**Generation is extractive, and that's the right call here.** The generator
selects text rather than writing it, so it runs locally in single-digit
milliseconds and **cannot hallucinate**. On MS MARCO, answers *are* short spans
from a passage an annotator marked relevant, so selecting the best-supported
sentence is what the task actually is. An optional Claude Haiku 4.5 path streams
a more fluent rewrite afterwards, using **tools** rather than a stuffed prompt.

**The guardrail eval measures over-refusal.** Anything can refuse everything. The
suite's control set includes legitimate questions containing alarming words
("kill a process", "bomb cyclone", "food poisoning", "ethical hacking") and
injection lookalikes ("ignore the noise", "what is a system prompt") — exactly
where lexicon-based guards fail invisibly. The false-positive rate is published
first, not buried.

**Thresholds are calibrated against measured distributions.** With
`multilingual-e5`, *clearly unrelated* text scores ~0.70 cosine — so "below 0.5
means unrelated" is simply wrong for this space. `bench/calibrate_thresholds.py`
sweeps the out-of-domain thresholds against a **natural** OOD set: real MSMARCO-XI
questions whose documents were never indexed. → [`docs/CALIBRATION.md`](docs/CALIBRATION.md)

---

## Engineering decisions that were measured, not assumed

Every one of these started as an assumption that turned out to be wrong:

| Assumption | Measurement | Outcome |
|---|---|---|
| int8 quantization is faster | 19 vs 13 chunks/sec vs fp32 | ✅ kept — but it was tested, because int8 is often *slower* for small transformers |
| more threads = faster indexing | 12 threads was **slower** than 4 (19→18/sec) | ❌ reverted to 4 — a 15 W part throttles |
| 256-token windows with a 220-token floor | fired on 0.6% of passages (4 chunks from 600) | ❌ re-sized to 96/24 above 128 tokens against the measured p50=79 / p90=130 distribution |
| RRF scores can drive the out-of-domain guard | RRF encodes **rank, not similarity** — a query with no good match still scores ~1/(k+1) | ❌ capture the raw cosine during dense search; it's free and exact |
| subsample the corpus by passage | would drop the gold passage out from under a surviving query | ❌ subsample **by query**, or the ablation measures the sampler |
| the reranker is worth ~90 ms | p90 was **196 ms**, so the budget manager skipped it on *every* request | ❌ re-sized to 4 pairs × 128 tokens; now runs on 98% of requests |
| the extractive generator costs ~5–25 ms | **65–127 ms** — the semantic sentence re-scoring dominates | ❌ split into two stages: lexical (0.8 ms, always) + semantic (droppable) |
| `language_probability` gates transcription quality | **1.00 on every clip, including the mis-transcribed ones**; 0.00 on a pure tone | ❌ it's a *language*-ID score. The gate catches non-speech, not mis-hearing — see below |

### The one that would have killed the live demo

`faiss.IDSelectorBatch` stores a **raw pointer** into a numpy array and does not
copy it. The original code wrote:

```python
faiss.swig_ptr(allowed_ids.astype(np.int64))   # astype COPIES
```

`astype` returns a copy, which is freed on the next line — so FAISS held a
pointer into freed memory. A textbook use-after-free, and it behaved like one:
170 unit tests passed, 29 end-to-end tests passed, 16 warm-up searches passed,
and then the **second real HTTP request took the whole server process down** with
no Python traceback, because the allocator had finally reused the page.

Fixed by pointing at the cached, already-int64 array (`ascontiguousarray` is a
no-op there, so no temporary is created) and holding it alive for the call.
`tests/test_dense_index.py` now runs 300 filtered searches with deliberate
allocation churn and forced GC — the shape of test that actually reproduces this
class of bug, rather than the shape that misses it.

Post-fix soak: **80/80 requests, 0 failures, 100% within budget.**

### The one that changed a design claim

Across 16 recorded clips in four languages, Sarvam returned
`language_probability: 1.00` on every single one — including `"chia seeds"` →
`"chair seeds"` and `"houston"` → `"Houseton"`. A 440 Hz tone returned `0.00`.

So the ASR confidence gate reliably rejects **non-speech** (silence, a mic tap, a
recording that captured nothing) and does **not** catch confident mis-hearing.
The architecture doc originally claimed the stronger thing; it now says this. The
actual defence against mis-transcription is that **the UI shows the user what was
heard** above every answer, so a misheard question is visibly a misheard question.

**Measured Sarvam STT latency** (16 clips, real API):

| | P50 | P70 | P90 | P100 |
|---|---:|---:|---:|---:|
| speech-to-text | 434 ms | 492 ms | 646 ms | 873 ms |

Which is the whole reason the 200 ms contract is scoped to the retrieval core and
reports STT separately instead of averaging it in.

---

## Quickstart

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
cp .env.example .env          # add VRAG_SARVAM_API_KEY

vrag doctor                   # check the environment
vrag ingest                   # download + normalize the parquet shards
vrag build                    # chunk, embed, index  (resumable)
vrag serve                    # http://localhost:7860
```

Try the pipeline without the server:

```bash
vrag ask "what is a corporation" --show-spans
vrag ask "कॉर्पोरेशन क्या है?" --lang hi
```

### Fast iteration

The full build is hours on CPU. A dev profile inherits everything from
`default.yaml` and changes **only** the corpus size, so dev and production cannot
drift behaviourally:

```bash
VRAG_CONFIG=configs/dev.yaml vrag build     # ~20 min, ~18k chunks
VRAG_CONFIG=configs/dev.yaml vrag serve
```

---

## Benchmarks

```bash
python bench/run_retrieval_eval.py     # chunking ablation      → docs/CHUNKING.md
python bench/run_latency.py --chart    # latency percentiles    → docs/LATENCY.md
python bench/run_guardrail_eval.py     # adversarial + control  → docs/GUARDRAILS.md
python bench/calibrate_thresholds.py   # OOD thresholds         → docs/CALIBRATION.md
```

All four write both a human-readable report under `docs/` and raw JSON under
`traces/`. The tables in those files are generated, never hand-written.

---

## Layout

```
src/vrag/
  chunking/     five views + the multi-view registry
  index/        ONNX embedder · FAISS · BM25 · mmapped chunk store
  retrieve/     multi-view search · RRF fusion · dedup · cross-encoder rerank
  generate/     extractive (fast path) · LLM (tool-calling polish path)
  guardrails/   input · domain · grounding · refusal policy
  harness/      stage graph · latency budget · resilience · tracing · tools
  stt/          provider protocol · Sarvam · ElevenLabs
  server/       FastAPI + mic UI with a live latency HUD
bench/          the four benchmarks above
docs/           ARCHITECTURE.md + generated result reports
```

Architecture and the reasoning behind each choice:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Deployment

```bash
docker build -t vrag .
docker run -p 7860:7860 -e VRAG_SARVAM_API_KEY=... \
  -v "$(pwd)/data/index:/app/data/index:ro" vrag
```

Two-stage build: torch and the ONNX exporter live in the build stage only and
never reach the runtime image. The **index is not baked in** — it is ~170 MB with
its own lifecycle, so it is published to a Hugging Face dataset repo
(`scripts/publish_index.py`) and fetched on first boot. Building it inside the
image would make every deploy a multi-hour job.

Full walkthrough, including how to verify a deployment rather than assume it:
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Tests

```bash
pytest                                          # unit tests, no index needed
VRAG_CONFIG=configs/dev.yaml pytest tests/test_pipeline.py   # end-to-end
```

Unit tests cover the sentence splitter (Devanagari/Bengali/Urdu terminators,
abbreviations, decimals, grapheme integrity), fusion arithmetic, dedup, the
budget manager, the resilience primitives, and every guardrail layer —
**including the cases where a guard must not fire**.

---

## License

MIT
