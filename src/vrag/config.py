"""Configuration loading.

Two sources, deliberately kept separate:

* ``configs/default.yaml`` -- everything that affects behaviour (thresholds,
  budgets, chunk sizes). Checked into git so every benchmark number in the repo
  is reproducible from the config that produced it.
* Environment / ``.env`` -- secrets only.

Anything tunable lives in the YAML rather than in code, because the ablation and
the threshold calibration both need to sweep values without touching source.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #
class Secrets(BaseSettings):
    """API keys. Never logged, never serialized into a response envelope."""

    model_config = SettingsConfigDict(
        env_prefix="VRAG_",
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sarvam_api_key: str = ""
    elevenlabs_api_key: str = ""
    anthropic_api_key: str = ""
    hf_token: str = ""

    @property
    def has_stt(self) -> bool:
        return bool(self.sarvam_api_key or self.elevenlabs_api_key)

    @property
    def has_llm(self) -> bool:
        return bool(self.anthropic_api_key)


# --------------------------------------------------------------------------- #
# Config tree (mirrors configs/default.yaml)
# --------------------------------------------------------------------------- #
class PathsCfg(BaseModel):
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    corpus_dir: Path = Path("data/corpus")
    index_dir: Path = Path("data/index")
    model_dir: Path = Path("models")
    trace_dir: Path = Path("traces")

    def resolve(self, root: Path) -> PathsCfg:
        """Make every path absolute against the repo root."""
        return PathsCfg(
            **{k: (root / v if not Path(v).is_absolute() else Path(v)) for k, v in self}
        )

    def ensure(self) -> None:
        for _, v in self:
            Path(v).mkdir(parents=True, exist_ok=True)


class CorpusCfg(BaseModel):
    repo_id: str = "ai4bharat/MSMARCO-XI"
    split: str = "validation"
    languages: dict[str, str] = Field(default_factory=dict)
    queries_per_language: int = 8500
    shared_queries: int = 1500
    include_english_parallel: bool = True
    min_passage_chars: int = 40
    max_passages_per_query: int = 8
    seed: int = 20260814
    # Subsample the corpus at index-build time, BY QUERY. See build.py for why
    # sampling passages directly would silently corrupt the ablation.
    # null = index everything.
    max_queries: int | None = None


class SentenceWindowCfg(BaseModel):
    enabled: bool = True
    window: int = 1
    min_sentence_chars: int = 25
    merge_short_into_next: bool = True


class FixedOverlapCfg(BaseModel):
    enabled: bool = True
    chunk_tokens: int = 96
    overlap_tokens: int = 24
    apply_above_tokens: int = 128


class SemanticCfg(BaseModel):
    enabled: bool = True
    breakpoint_percentile: int = 25
    min_chunk_sentences: int = 1
    max_chunk_sentences: int = 8


class PropositionCfg(BaseModel):
    enabled: bool = False
    top_passages: int = 20000
    max_props_per_passage: int = 6


class AtomicCfg(BaseModel):
    enabled: bool = True


class ViewsCfg(BaseModel):
    atomic: AtomicCfg = AtomicCfg()
    sentence_window: SentenceWindowCfg = SentenceWindowCfg()
    fixed_overlap: FixedOverlapCfg = FixedOverlapCfg()
    semantic: SemanticCfg = SemanticCfg()
    proposition: PropositionCfg = PropositionCfg()

    def enabled_names(self) -> list[str]:
        return [name for name, _ in self if getattr(self, name).enabled]


class ChunkingCfg(BaseModel):
    views: ViewsCfg = ViewsCfg()
    contextual_prefix: bool = True


class OnnxCfg(BaseModel):
    quantize_int8: bool = True
    intra_op_threads: int = 4
    build_intra_op_threads: int = 4
    inter_op_threads: int = 1


class EmbeddingCfg(BaseModel):
    model_id: str = "intfloat/multilingual-e5-small"
    dim: int = 384
    max_seq_len: int = 192
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    onnx: OnnxCfg = OnnxCfg()
    batch_size: int = 128


class HnswCfg(BaseModel):
    m: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    quantizer: Literal["sq8", "flat"] = "sq8"


class IvfPqCfg(BaseModel):
    nlist: int = 4096
    m: int = 48
    nbits: int = 8
    nprobe: int = 16


class DenseCfg(BaseModel):
    index_type: Literal["hnsw", "ivfpq"] = "hnsw"
    hnsw: HnswCfg = HnswCfg()
    ivfpq: IvfPqCfg = IvfPqCfg()
    normalize: bool = True


class SparseCfg(BaseModel):
    enabled: bool = True
    k1: float = 1.2
    b: float = 0.75
    method: str = "lucene"


class FusionCfg(BaseModel):
    method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    view_weights: dict[str, float] = Field(default_factory=dict)
    dense_weight: float = 1.0
    sparse_weight: float = 0.8


class DedupCfg(BaseModel):
    enabled: bool = True
    span_iou_threshold: float = 0.6
    max_per_doc: int = 2


class LanguageFilterCfg(BaseModel):
    min_asr_confidence: float = 0.75
    always_include_english: bool = True


class RetrievalCfg(BaseModel):
    top_k_per_view: int = 30
    top_k_fused: int = 40
    top_k_final: int = 8
    fusion: FusionCfg = FusionCfg()
    dedup: DedupCfg = DedupCfg()
    language_filter: LanguageFilterCfg = LanguageFilterCfg()


class RerankCfg(BaseModel):
    enabled: bool = True
    model_id: str = "nreimers/mmarco-mMiniLMv2-L6-H384-v1"
    quantize_int8: bool = True
    top_n: int = 4
    max_seq_len: int = 128
    intra_op_threads: int = 4


class ExtractiveCfg(BaseModel):
    max_sentences: int = 3
    min_answer_chars: int = 8
    max_answer_chars: int = 600
    lead_context_sentences: int = 1


class LlmCfg(BaseModel):
    model: str = "claude-haiku-4-5"
    max_tokens: int = 400
    temperature: float = 0.0
    max_tool_rounds: int = 2
    hard_timeout_ms: int = 3000


class GenerationCfg(BaseModel):
    mode: Literal["extractive", "llm", "extractive_then_llm"] = "extractive"
    extractive: ExtractiveCfg = ExtractiveCfg()
    llm: LlmCfg = LlmCfg()


class StageBudget(BaseModel):
    required: bool = True
    soft_ms: float = 20.0


class BudgetCfg(BaseModel):
    core_budget_ms: float = 200.0
    safety_margin_ms: float = 15.0
    stages: dict[str, StageBudget] = Field(default_factory=dict)


class InputGuardCfg(BaseModel):
    min_chars: int = 3
    max_chars: int = 500
    min_asr_confidence: float = 0.45
    block_unsafe: bool = True
    block_injection: bool = True
    redact_pii: bool = True


class DomainGuardCfg(BaseModel):
    min_top1_score: float = 0.845
    # Off by default: calibration measured this signal at AUC ~0.50 (indistinguishable)
    # on this corpus. See docs/CALIBRATION.md.
    use_centroid: bool = False
    max_centroid_distance: float = 0.62
    min_supporting_chunks: int = 1
    # Cosine headroom above min_top1_score below which an allowed answer is still
    # flagged borderline and has its confidence discounted.
    borderline_margin: float = 0.03


class GroundingCfg(BaseModel):
    mode: Literal["lexical_embedding", "nli"] = "lexical_embedding"
    min_lexical_overlap: float = 0.35
    min_embedding_similarity: float = 0.70
    require_citation: bool = True


class ConflictCfg(BaseModel):
    # Off by default: measured 1 false positive, 0 true positives. See
    # configs/default.yaml for the reasoning.
    enabled: bool = False
    min_disagreement: float = 0.45
    min_topical_overlap: float = 0.55


class GuardrailsCfg(BaseModel):
    input: InputGuardCfg = InputGuardCfg()
    domain: DomainGuardCfg = DomainGuardCfg()
    grounding: GroundingCfg = GroundingCfg()
    conflict: ConflictCfg = ConflictCfg()


class ServerCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7860
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_audio_bytes: int = 10 * 1024 * 1024
    warm_on_boot: bool = True
    warm_rounds: int = 2


class SarvamCfg(BaseModel):
    endpoint: str = "https://api.sarvam.ai/speech-to-text"
    model: str = "saaras:v4"
    mode: str = "transcribe"
    language_code: str = "unknown"
    timeout_s: float = 8.0
    max_retries: int = 3
    backoff_base_s: float = 0.25
    circuit_breaker_failures: int = 5
    circuit_breaker_reset_s: float = 30.0


class ElevenLabsCfg(BaseModel):
    endpoint: str = "https://api.elevenlabs.io/v1/speech-to-text"
    model_id: str = "scribe_v2"
    timeout_s: float = 8.0
    max_retries: int = 3


class SttCfg(BaseModel):
    provider: Literal["sarvam", "elevenlabs"] = "sarvam"
    sarvam: SarvamCfg = SarvamCfg()
    elevenlabs: ElevenLabsCfg = ElevenLabsCfg()


class LatencyBenchCfg(BaseModel):
    n_queries: int = 500
    n_audio_clips: int = 50
    warmup: int = 25
    percentiles: list[int] = Field(default_factory=lambda: [50, 70, 90, 95, 99, 100])


class RetrievalBenchCfg(BaseModel):
    n_eval_queries: int = 2000
    k_values: list[int] = Field(default_factory=lambda: [1, 5, 20])


class RemoteIndexCfg(BaseModel):
    """Where to fetch a prebuilt index when one is not present locally.

    Empty ``repo_id`` disables the whole mechanism, which is what local
    development wants -- you build the index yourself and never touch the network.
    """

    repo_id: str = ""
    # "dataset" is the natural home for an index, but a plain model repo works
    # just as well and avoids making the user create a second repo. Both are
    # arbitrary file stores as far as snapshot_download is concerned.
    repo_type: Literal["dataset", "model"] = "model"
    revision: str = "main"
    fetch_on_boot: bool = True


class BenchCfg(BaseModel):
    latency: LatencyBenchCfg = LatencyBenchCfg()
    retrieval: RetrievalBenchCfg = RetrievalBenchCfg()


class Config(BaseModel):
    paths: PathsCfg = PathsCfg()
    corpus: CorpusCfg = CorpusCfg()
    chunking: ChunkingCfg = ChunkingCfg()
    embedding: EmbeddingCfg = EmbeddingCfg()
    dense: DenseCfg = DenseCfg()
    sparse: SparseCfg = SparseCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    rerank: RerankCfg = RerankCfg()
    generation: GenerationCfg = GenerationCfg()
    budget: BudgetCfg = BudgetCfg()
    guardrails: GuardrailsCfg = GuardrailsCfg()
    server: ServerCfg = ServerCfg()
    stt: SttCfg = SttCfg()
    remote_index: RemoteIndexCfg = RemoteIndexCfg()
    bench: BenchCfg = BenchCfg()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving an ``extends:`` chain.

    A profile (``configs/dev.yaml``) lists only what it changes and inherits the
    rest, so a dev run and a production run cannot drift behaviourally -- the only
    differences are the ones written down in the profile.
    """
    _seen = _seen or set()
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular config extends chain at {path}")
    _seen.add(path)

    if not path.exists():
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent_ref = raw.pop("extends", None)
    if parent_ref is None and path != DEFAULT_CONFIG_PATH.resolve():
        # Any profile other than the base implicitly extends the base.
        parent_ref = str(DEFAULT_CONFIG_PATH)

    if parent_ref:
        parent_path = Path(parent_ref)
        if not parent_path.is_absolute():
            parent_path = (path.parent / parent_path).resolve()
            if not parent_path.exists():
                parent_path = (REPO_ROOT / parent_ref).resolve()
        return _deep_merge(_load_yaml(parent_path, _seen), raw)

    return raw


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> Config:
    """Load YAML config, apply overrides, resolve paths against the repo root.

    ``overrides`` is an explicit parameter rather than ``**kwargs`` on purpose: with
    kwargs, a caller writing ``load_config(overrides={...})`` -- the natural
    spelling -- has the whole dict swallowed as a stray top-level key and silently
    ignored, so the override appears to do nothing and the config looks fine.
    """
    path = Path(path or os.environ.get("VRAG_CONFIG") or DEFAULT_CONFIG_PATH)
    raw = _load_yaml(path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    cfg = Config.model_validate(raw)
    cfg.paths = cfg.paths.resolve(REPO_ROOT)
    return cfg


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return Secrets()
