"""Guards against build-time dependencies leaking onto the serve path.

The runtime container deliberately ships neither `torch` nor `optimum`: they are
needed only to export the encoders to ONNX, which happens in a separate Docker
build stage. Excluding them keeps the image roughly a quarter of the size.

That split is invisible in development, where both are always installed — so a
stray import at module scope, or above an early-return, passes every local test
and then kills the container on deploy. It did exactly that:

    ModuleNotFoundError: No module named 'optimum'
      File "/app/src/vrag/index/embedder.py", line 41, in export_encoder
        from optimum.onnxruntime import ORTModelForFeatureExtraction

The models were baked into the image and the early-return that would have used
them sat *below* the import, so it was never reached.

These tests simulate the runtime image by making the build-only modules
unimportable.
"""

from __future__ import annotations

import builtins
import importlib

import pytest

BUILD_ONLY = ("torch", "optimum", "onnx")

# Modules that must be importable with only the runtime dependency set.
SERVE_PATH_MODULES = [
    "vrag.config",
    "vrag.schemas",
    "vrag.index.store",
    "vrag.index.dense",
    "vrag.index.sparse",
    "vrag.index.fetch",
    "vrag.retrieve.fusion",
    "vrag.retrieve.expand",
    "vrag.retrieve.multiview",
    "vrag.generate.extractive",
    "vrag.guardrails.input_guard",
    "vrag.guardrails.domain_guard",
    "vrag.guardrails.grounding",
    "vrag.guardrails.policy",
    "vrag.harness.budget",
    "vrag.harness.resilience",
    "vrag.harness.tracing",
    "vrag.harness.pipeline",
    "vrag.stt.sarvam",
    "vrag.server.app",
]


@pytest.fixture
def without_build_deps(monkeypatch):
    """Make torch/optimum/onnx raise ImportError, as they would in the image."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        root = name.split(".")[0]
        if root in BUILD_ONLY:
            raise ModuleNotFoundError(f"No module named '{root}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    return guarded


@pytest.mark.parametrize("module", SERVE_PATH_MODULES)
def test_serve_path_imports_without_build_dependencies(module, without_build_deps):
    importlib.reload(importlib.import_module(module))


class TestExportEarlyReturn:
    """The path production always takes: models already exported."""

    def test_embedder_export_returns_without_importing_optimum(
        self, tmp_path, without_build_deps
    ):
        from vrag.config import load_config
        from vrag.index.embedder import export_encoder

        cfg = load_config()
        cfg.paths.model_dir = tmp_path
        model_id = cfg.embedding.model_id

        target = tmp_path / f"{model_id.replace('/', '__')}__embed"
        target.mkdir(parents=True)
        (target / "model.onnx").write_bytes(b"stub")

        # Must return the existing export without touching optimum.
        assert export_encoder(cfg, model_id, "embed", quantize=True) == target

    def test_reranker_export_returns_without_importing_optimum(
        self, tmp_path, without_build_deps
    ):
        from vrag.config import load_config
        from vrag.retrieve.rerank import export_cross_encoder

        cfg = load_config()
        cfg.paths.model_dir = tmp_path
        model_id = cfg.rerank.model_id

        target = tmp_path / f"{model_id.replace('/', '__')}__rerank"
        target.mkdir(parents=True)
        (target / "model.onnx").write_bytes(b"stub")

        assert export_cross_encoder(cfg, model_id, quantize=True) == target

    def test_export_still_raises_when_nothing_is_exported(
        self, tmp_path, without_build_deps
    ):
        """Without a prior export there is genuinely nothing to do but fail --
        but it must fail on the missing dependency, not silently return a path
        to a model that does not exist."""
        from vrag.config import load_config
        from vrag.index.embedder import export_encoder

        cfg = load_config()
        cfg.paths.model_dir = tmp_path
        with pytest.raises(ModuleNotFoundError):
            export_encoder(cfg, cfg.embedding.model_id, "embed", quantize=True)
