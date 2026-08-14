"""ONNX int8 sentence embedder -- the first stage inside the 200 ms budget.

Design notes, all driven by "this must run on a CPU-only laptop and a free-tier
container":

* **ONNX Runtime, not PyTorch, at serve time.** Torch is a build-time dependency
  only (it performs the export). A single-sequence forward pass through
  ``multilingual-e5-small`` costs ~35-60 ms under torch on this hardware and
  ~5-15 ms under ORT with dynamic int8 quantization.
* **Direct ``InferenceSession``, not ``ORTModel``.** Optimum's wrapper adds
  per-call Python overhead that is invisible at batch-1024 and very visible at
  batch-1.
* **Threads pinned low.** ORT's default is one thread per core, which on a busy
  box means the query embed contends with the reranker. Four intra-op threads and
  one inter-op thread measured best and, more importantly, most *consistently* --
  P100 matters more than P50 when the SLA is a hard ceiling.
* **Warmed at construction.** The first inference pays graph-optimization and
  arena-allocation costs. Paying them at boot keeps them out of the P100.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from vrag.config import Config, EmbeddingCfg


def _onnx_dir(cfg: Config, model_id: str, tag: str) -> Path:
    return cfg.paths.model_dir / f"{model_id.replace('/', '__')}__{tag}"


def export_encoder(cfg: Config, model_id: str, tag: str, quantize: bool) -> Path:
    """Export a HF encoder to ONNX and (optionally) dynamically quantize to int8.

    Idempotent: returns immediately if the artifact already exists, so a rebuild
    does not re-export.
    """
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    target = _onnx_dir(cfg, model_id, tag)
    if (target / "model.onnx").exists():
        return target

    target.mkdir(parents=True, exist_ok=True)
    fp32_dir = target / "_fp32"

    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    model.save_pretrained(fp32_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(target)

    if quantize:
        quantizer = ORTQuantizer.from_pretrained(fp32_dir)
        # avx512_vnni where available, avx2 otherwise; `is_static=False` means
        # dynamic quantization, which needs no calibration set and costs ~1% of
        # retrieval quality on this model while roughly halving latency.
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
        quantizer.quantize(save_dir=target, quantization_config=qconfig)
        produced = next(target.glob("*quantized*.onnx"), None)
        if produced is None:
            raise RuntimeError(f"quantization produced no onnx file in {target}")
        produced.rename(target / "model.onnx")
        shutil.rmtree(fp32_dir, ignore_errors=True)
    else:
        (fp32_dir / "model.onnx").rename(target / "model.onnx")
        shutil.rmtree(fp32_dir, ignore_errors=True)

    return target


class OnnxEmbedder:
    """Mean-pooled, L2-normalized embeddings from an ONNX encoder."""

    def __init__(
        self, cfg: Config, ecfg: EmbeddingCfg | None = None, build_mode: bool = False
    ) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.cfg = cfg
        self.ecfg = ecfg or cfg.embedding
        self.build_mode = build_mode

        model_dir = export_encoder(
            cfg, self.ecfg.model_id, "embed", self.ecfg.onnx.quantize_int8
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Build and serve want opposite thread settings. Indexing is a throughput
        # problem -- use every core. Serving is a tail-latency problem: more
        # threads means more scheduling variance and more contention with the
        # reranker, and P100 is what the SLA is written against.
        options = ort.SessionOptions()
        options.intra_op_num_threads = (
            self.ecfg.onnx.build_intra_op_threads
            if build_mode
            else self.ecfg.onnx.intra_op_threads
        )
        options.inter_op_num_threads = self.ecfg.onnx.inter_op_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = [i.name for i in self.session.get_inputs()]
        self.dim = self.ecfg.dim

        self._warm()

    def _warm(self) -> None:
        """Force graph optimization and arena allocation out of the first request."""
        self.encode(["warmup"], prefix=self.ecfg.query_prefix)

    # -- core ---------------------------------------------------------------- #
    def _build_feeds(self, enc) -> dict[str, np.ndarray]:  # noqa: ANN001
        """Match the tokenizer's output to the ONNX graph's declared inputs.

        The exported graph declares ``token_type_ids`` because it descends from
        BERT, but ``multilingual-e5-small`` is XLM-R based and its tokenizer never
        emits them -- ORT then rejects the call for a missing required input. The
        model ignores the values (it has a single segment), so zeros are correct
        rather than merely accepted.

        Filling by declared-input name instead of by tokenizer key also makes the
        embedder work unchanged if the encoder is swapped for one with a different
        input signature.
        """
        feeds: dict[str, np.ndarray] = {}
        for name in self._input_names:
            value = enc.get(name)
            if value is None:
                value = np.zeros_like(enc["input_ids"])
            feeds[name] = np.ascontiguousarray(value, dtype=np.int64)
        return feeds

    def _forward(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.ecfg.max_seq_len,
            return_tensors="np",
        )
        feeds = self._build_feeds(enc)
        hidden = self.session.run(None, feeds)[0]          # (B, T, H)

        # e5 uses mean pooling over non-pad tokens. Pooling over pads would drag
        # every short query toward the same padding-dominated vector.
        mask = enc["attention_mask"].astype(np.float32)[..., None]
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)

    def encode(
        self,
        texts: list[str],
        prefix: str = "",
        batch_size: int | None = None,
        sort_by_length: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts, best-first batching, results in input order.

        ``sort_by_length`` is the single most valuable optimisation in the whole
        build. Padding is per batch, to the longest member: one 192-token passage
        dropped into a batch of 128 short sentence-chunks forces all 128 rows to
        192 tokens, and the model then spends most of its FLOPs on padding. The
        corpus is extremely skewed this way -- median passage is 79 tokens, p99 is
        205, max is 1503 -- so unsorted batching wastes the majority of the
        compute. Sorting by length first makes batches near-homogeneous.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        prepared = [f"{prefix}{t}" for t in texts] if prefix else texts
        batch = batch_size or self.ecfg.batch_size

        if len(prepared) <= batch:
            return self._forward(prepared)

        if not sort_by_length:
            return np.vstack(
                [self._forward(prepared[i : i + batch]) for i in range(0, len(prepared), batch)]
            )

        # Character length is a good enough proxy for token length here and costs
        # nothing; tokenizing twice just to sort would eat the gain.
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]))
        out = np.empty((len(prepared), self.dim), dtype=np.float32)

        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            out[idx] = self._forward([prepared[i] for i in idx])

        return out

    # -- convenience --------------------------------------------------------- #
    def encode_query(self, text: str) -> np.ndarray:
        """Single query -> (dim,) vector. This is the call on the hot path."""
        return self._forward([f"{self.ecfg.query_prefix}{text}"])[0]

    def encode_passages(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return self.encode(texts, prefix=self.ecfg.passage_prefix, batch_size=batch_size)

    def encode_raw(self, texts: list[str]) -> np.ndarray:
        """No prefix. Used by the semantic chunker, which compares sentences to
        each other rather than to a query, so the asymmetric e5 prefixes do not apply."""
        return self.encode(texts, prefix="")
