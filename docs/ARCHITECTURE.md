# Architecture

```
                    ┌──────────── HARNESS: typed stage graph, budget, tracing ────────────┐
                    │                                                                      │
  🎙 mic ──WAV 16k──▶ STT (Sarvam) ─▶ InputGuard ─▶ QueryPlan ─▶ ┌─ dense  (FAISS HNSW-SQ8) │
                    │  retry+breaker    5 checks     normalize   ├─ sparse (BM25, bm25s)    │
                    │  lang_probability ─────────────▶ ASR gate  └─ 4 views, one index      │
                    │                                                      │                │
                    │                                              RRF fusion + dedup       │
                    │                                                      │                │
                    │                                        budget-gated cross-encoder     │
                    │                                                      │                │
                    │                                              DomainGuard (2 signals)  │
                    │                                                      │                │
                    │                                          extractive generator (local) │
                    │                                                      │                │
                    │                                              GroundingGuard           │
                    │                                                      │                │
                    └──────────────────── AnswerEnvelope ──────────────────┘                │
                                          {answer, citations[], abstained, refusal_reason,
                                           timings{}, degradations[], spans[]}
```

Every arrow is a typed Pydantic boundary. Every box emits a span. The span log
*is* the latency dataset — `bench/run_latency.py` reads it back rather than
measuring separately, so the published numbers describe the code that serves
traffic.

---

## The <200 ms contract

The contract covers **embed → retrieve → fuse → rerank → guard → answer**. It
deliberately excludes speech-to-text, which is a network round trip to Sarvam and
cannot fit in 200 ms on any budget. Both numbers are always reported; the core
one is never presented as the end-to-end one.

What makes it a contract rather than an average is `harness/budget.py`. Each
stage is declared `required` or not. Before an optional stage runs, the manager
compares the **measured rolling p90** of that stage against the remaining budget
and skips it if it will not fit:

| stage | required | what dropping it costs |
|---|---|---|
| `embed_query` | ✅ | — |
| `dense_search` | ✅ | — |
| `sparse_search` | ❌ | rare-term recall (names, numbers, codes) |
| `fuse` | ✅ | — |
| `rerank` | ❌ | ordering precision within the shortlist |
| `domain_guard` | ✅ | — |
| `generate` | ✅ | — |
| `grounding_guard` | ❌ | attribution verification |

Two details matter:

- **p90, not the mean.** A stage averaging 40 ms with a p90 of 110 ms overruns
  roughly one request in ten if you budget against the mean. A ceiling has to be
  planned against the tail.
- **Every skip is reported.** Skips surface in the response as `degradations[]`
  and render in the UI. A fast answer is never *silently* a worse answer.

---

## Retrieval: four views, one index

The five chunking views (four enabled by default; propositions require an offline
LLM pass) all live in **one** FAISS index and **one** BM25 index, tagged by
`view`. A query therefore costs one dense search and one sparse search regardless
of how many views exist — multi-view retrieval that pays a round trip per view
does not fit a 200 ms budget.

Per-view structure is recovered by *bucketing* the single result list by view
before fusion. The dense search asks for `top_k_per_view × n_views` neighbours
rather than `top_k_fused`, so a view that never cracks the global top-40 still
contributes a run.

**Why RRF and not weighted score blending:** the views produce scores on
incomparable scales. A one-sentence chunk and a 96-token chunk, both perfectly
relevant, do not receive similar cosines — short texts concentrate, long texts
average out. The same is true across dense (cosine, [-1,1]) and BM25 (unbounded).
RRF discards magnitude and keeps rank, the one thing every run agrees on the
meaning of. It also gives **multi-view agreement** for free: a chunk found by
several views accumulates contributions from each, which is the actual payoff of
running more than one view.

Dedup runs *after* fusion so agreement is rewarded before duplicates are dropped,
and collapses chunks whose character spans overlap by IoU ≥ 0.6 — five views over
one passage produce overlapping text by construction, and without this the top-8
handed to the generator can be the same paragraph five times.

---

## Generation: extractive first

The extractive generator selects text rather than writing it, so it runs locally
in single-digit milliseconds and **cannot hallucinate** — every character it
emits came from a retrieved chunk. On this dataset that is not a compromise: MS
MARCO answers are short spans drawn from a passage an annotator marked relevant,
so selecting the best-supported sentence is what the task actually is.

The optional LLM path (Claude Haiku 4.5) runs *after* the grounded answer has
already been returned, streams a more fluent rewrite, and is given **tools**
(`search_corpus`, `fetch_neighbours`, `normalize_query`) rather than a stuffed
prompt — so it can recover from a thin shortlist, pulls only the context it
needs, and every action it takes is a span in the trace.

---

## Guardrails: three layers

| layer | when | signals |
|---|---|---|
| **Input** | before retrieval | ASR confidence, length, unsafe content, prompt injection, PII redaction |
| **Domain** | after retrieval | top-1 cosine **and** distance from the corpus centroid |
| **Grounding** | after generation | citation required, lexical attribution, embedding attribution |

### What the ASR gate actually catches — measured, not assumed

Speech recognition does not fail by returning nothing; it fails by returning a
fluent, plausible, *wrong* sentence. No downstream grounding check catches that,
because the answer is perfectly grounded in evidence for a question the user
never asked. So an ASR gate is the natural place to defend.

**Measurement changed what we claim for it.** Across 16 recorded clips in four
languages, Sarvam returned `language_probability: 1.00` on *every* clip —
including the ones it got wrong ("chia seeds" → "chair seeds", "houston" →
"Houseton"). A pure 440 Hz tone, by contrast, returned `0.00`.

`language_probability` is therefore a **language-identification** score, not a
transcription-quality score. The gate reliably catches *non-speech* — silence, an
accidental mic tap, background noise, a recording that captured nothing — and it
does **not** catch confident mis-hearing. Claiming otherwise would be the
flattering reading of a number we measured.

The real defence against mis-transcription is therefore **showing the user what
was heard**. The UI renders the transcript above every answer, so a user who
asked about chia seeds and sees "chair seeds" knows immediately why the answer is
odd, and can retry. That is a weaker guarantee than a confidence gate would have
been, and it is the honest one. A provider exposing per-word logprobs
(ElevenLabs does) would allow a real transcription-quality gate; the
`SttProvider` protocol exists partly so that swap stays cheap.

The **domain guard requires both signals to fail** before refusing. Top-1
similarity alone is fooled by a query that shares vocabulary with one unrelated
passage; centroid distance alone is fooled by an in-domain question phrased
unusually. Requiring either would abstain constantly — with `e5`, even clearly
unrelated text scores ~0.70 cosine, which is exactly why the thresholds are
calibrated rather than guessed.

---

## Key implementation decisions

| decision | why |
|---|---|
| ONNX Runtime at serve time, torch only at build time | ~35–60 ms → ~9 ms per query embed on this CPU; torch never ships in the runtime image |
| int8 dynamic quantization | **measured** 19 vs 13 chunks/sec against fp32 on this box — quantization is not always faster for small transformers, so it was tested rather than assumed |
| 4 intra-op threads, not 12 | **measured**: 12 threads was *slower* (19→18 chunks/sec). A 15 W part throttles; extra threads only add contention |
| Length-sorted batching at build time | padding is per batch to the longest member; one 192-token outlier in a batch of short sentence-chunks wastes most of the compute |
| FAISS in-process, not a vector DB server | a network hop costs 1–5 ms before any work happens — affordable at 1 s, not at a 25 ms retrieval budget |
| `bm25s`, not `rank_bm25` | `rank_bm25` scores in a Python loop: ~2 s/query at this corpus size, ten times the entire budget |
| HNSW + SQ8 scalar quantization | 384-d float32 → int8 cuts the index ~4×, the difference between fitting free-tier hosting and not |
| Columnar mmapped chunk store | turning 40 chunk ids into text on every request must not cost deserialization |
| Multilingual reranker (mMARCO mMiniLM-L6) | an English-only `ms-marco-MiniLM` would be actively wrong on a majority-Indic corpus |
| Purpose-built orchestrator, not LangChain | the requirement is to *show* structured orchestration; a `Chain` hides exactly the seams — the clock, the droppable stages, the failure paths — that are the point |

---

## Failure behaviour

There is exactly one response type. Every path — success, refusal, timeout,
dependency outage, unhandled exception — returns an `AnswerEnvelope`, and no
exception escapes `answer()`.

| failure | result |
|---|---|
| Sarvam key missing / provider down | breaker opens, `STT_UNAVAILABLE`, UI offers text input |
| Reranker fails to load | pipeline runs without it, degradation reported |
| Sparse index absent | dense-only, degradation reported |
| Budget exhausted mid-request | optional stages skipped, degradations listed |
| Unhandled exception | `INTERNAL_ERROR` envelope, request traced |
