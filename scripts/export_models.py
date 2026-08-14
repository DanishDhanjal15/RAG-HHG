"""Export the ONNX encoders. Runs in the Docker build stage, never at serve time.

Kept as a standalone script with no `vrag` import so the exporter build stage does
not need the application or its runtime dependencies -- it needs torch and optimum,
which the runtime image deliberately does not have.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

EMBED_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "nreimers/mmarco-mMiniLMv2-L6-H384-v1"


def export(model_id: str, out_dir: Path, task: str, quantize: bool = True) -> None:
    from optimum.onnxruntime import (
        ORTModelForFeatureExtraction,
        ORTModelForSequenceClassification,
        ORTQuantizer,
    )
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    if (out_dir / "model.onnx").exists():
        print(f"[skip] {out_dir} already exported")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "_fp32"

    cls = (
        ORTModelForFeatureExtraction
        if task == "feature-extraction"
        else ORTModelForSequenceClassification
    )
    print(f"[export] {model_id} -> {out_dir}")
    cls.from_pretrained(model_id, export=True).save_pretrained(fp32)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)

    if quantize:
        print(f"[quantize] {model_id}")
        quantizer = ORTQuantizer.from_pretrained(fp32)
        quantizer.quantize(
            save_dir=out_dir,
            quantization_config=AutoQuantizationConfig.avx2(is_static=False, per_channel=True),
        )
        produced = next(out_dir.glob("*quantized*.onnx"), None)
        if produced is None:
            raise RuntimeError(f"quantization produced no onnx file in {out_dir}")
        produced.rename(out_dir / "model.onnx")
    else:
        (fp32 / "model.onnx").rename(out_dir / "model.onnx")

    shutil.rmtree(fp32, ignore_errors=True)
    size_mb = (out_dir / "model.onnx").stat().st_size / 1e6
    print(f"[done] {out_dir/'model.onnx'}  {size_mb:.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models")
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--rerank-model", default=RERANK_MODEL)
    args = parser.parse_args()

    root = Path(args.out)
    export(args.embed_model, root / f"{args.embed_model.replace('/', '__')}__embed",
           "feature-extraction")
    export(args.rerank_model, root / f"{args.rerank_model.replace('/', '__')}__rerank",
           "sequence-classification")


if __name__ == "__main__":
    main()
