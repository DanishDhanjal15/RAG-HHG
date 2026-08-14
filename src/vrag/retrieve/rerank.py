"""Cross-encoder reranking -- the most expensive optional stage.

A bi-encoder embeds the query and the passage separately, so it can never model
the interaction between them; a cross-encoder reads the pair jointly and is
substantially more accurate at picking the best few from a shortlist. It is also
roughly an order of magnitude more expensive, because it runs once per candidate
instead of once per query.

That cost is exactly why this stage is declared ``required: false`` in the budget
config. On a CPU-only box, 8 pairs at 192 tokens through a 6-layer model is tens
of milliseconds -- affordable most of the time, not affordable under load. The
budget manager measures the rolling p90 and skips the stage when the remaining
budget cannot cover it, reporting a ``degradation`` instead of overrunning the SLA.

Latency controls, in order of impact:

* ``top_n`` -- cost is linear in candidates. This is the first knob to turn.
* ``max_seq_len`` -- attention is quadratic in sequence length.
* **Length-sorted batching** -- padding to the longest item in the batch is pure
  waste. Sorting by token length before batching cuts padded tokens substantially
  on a mixed-length candidate set.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from vrag.config import Config
from vrag.schemas import Evidence


def export_cross_encoder(cfg: Config, model_id: str, quantize: bool) -> Path:
    """Export a HF sequence-classification cross-encoder to ONNX (+ int8)."""
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    target = cfg.paths.model_dir / f"{model_id.replace('/', '__')}__rerank"
    if (target / "model.onnx").exists():
        return target

    target.mkdir(parents=True, exist_ok=True)
    fp32_dir = target / "_fp32"

    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(fp32_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(target)

    if quantize:
        quantizer = ORTQuantizer.from_pretrained(fp32_dir)
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
        quantizer.quantize(save_dir=target, quantization_config=qconfig)
        produced = next(target.glob("*quantized*.onnx"), None)
        if produced is None:
            raise RuntimeError(f"quantization produced no onnx file in {target}")
        produced.rename(target / "model.onnx")
    else:
        (fp32_dir / "model.onnx").rename(target / "model.onnx")

    shutil.rmtree(fp32_dir, ignore_errors=True)
    return target


class CrossEncoderReranker:
    def __init__(self, cfg: Config) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.cfg = cfg
        rcfg = cfg.rerank

        model_dir = export_cross_encoder(cfg, rcfg.model_id, rcfg.quantize_int8)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        options = ort.SessionOptions()
        options.intra_op_num_threads = rcfg.intra_op_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._warm()

    def _warm(self) -> None:
        self.score("warmup query", ["warmup passage"])

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.empty(0, dtype=np.float32)

        enc = self.tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=self.cfg.rerank.max_seq_len,
            return_tensors="np",
        )
        feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self._input_names}
        logits = self.session.run(None, feeds)[0]

        # mMiniLM rerankers emit a single relevance logit; some checkpoints emit
        # two-class logits instead. Handle both rather than assume.
        if logits.ndim == 2 and logits.shape[1] == 2:
            scores = logits[:, 1] - logits[:, 0]
        else:
            scores = logits.reshape(-1)
        return scores.astype(np.float32)

    def rerank(self, query: str, evidence: list[Evidence], top_n: int | None = None) -> list[Evidence]:
        """Rescore the shortlist and return it in the new order.

        Candidates are length-sorted before scoring so the batch pads to something
        near the median rather than to the single longest passage, then restored
        to score order. The original fused score is kept on each ``Evidence`` --
        the reranker refines the ordering, it does not erase the retrieval signal
        that produced the shortlist.
        """
        top_n = top_n or self.cfg.rerank.top_n
        shortlist = evidence[:top_n]
        if len(shortlist) < 2:
            return evidence

        order = sorted(range(len(shortlist)), key=lambda i: len(shortlist[i].text))
        sorted_texts = [shortlist[i].text for i in order]
        sorted_scores = self.score(query, sorted_texts)

        scores = np.empty(len(shortlist), dtype=np.float32)
        for position, original_index in enumerate(order):
            scores[original_index] = sorted_scores[position]

        for item, score in zip(shortlist, scores, strict=True):
            item.rerank_score = float(score)

        shortlist.sort(key=lambda e: -(e.rerank_score or 0.0))
        return shortlist + evidence[top_n:]
