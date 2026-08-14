"""Config loading tests.

The `extends` chain is what guarantees the dev profile and the production profile
differ ONLY where the profile says so. If inheritance silently dropped a key, dev
runs would validate behaviour that production never has -- the most expensive kind
of bug to find, because everything looks fine until it ships.
"""

from __future__ import annotations

import textwrap

import pytest

from vrag.config import DEFAULT_CONFIG_PATH, Config, _deep_merge, load_config


class TestDeepMerge:
    def test_override_wins_on_scalars(self):
        assert _deep_merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}

    def test_nested_dicts_merge_rather_than_replace(self):
        merged = _deep_merge(
            {"corpus": {"seed": 1, "max_queries": 10}},
            {"corpus": {"max_queries": 20}},
        )
        assert merged["corpus"] == {"seed": 1, "max_queries": 20}

    def test_lists_replace_wholesale(self):
        merged = _deep_merge({"k": [1, 2, 3]}, {"k": [9]})
        assert merged["k"] == [9]

    def test_base_is_not_mutated(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


class TestConfigLoading:
    def test_default_config_is_valid(self):
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.embedding.dim == 384
        assert cfg.budget.core_budget_ms == 200

    def test_paths_resolve_to_absolute(self):
        cfg = load_config()
        assert cfg.paths.index_dir.is_absolute()
        assert cfg.paths.corpus_dir.is_absolute()

    def test_dev_profile_inherits_everything_it_does_not_override(self):
        """The whole point of `extends`: a profile lists what changes, and
        inherits the rest, so dev cannot drift from production behaviour."""
        base = load_config(DEFAULT_CONFIG_PATH)
        dev = load_config(DEFAULT_CONFIG_PATH.parent / "dev.yaml")

        # Overridden by the profile:
        assert dev.corpus.max_queries != base.corpus.max_queries
        assert dev.paths.index_dir != base.paths.index_dir

        # Everything behavioural must be identical.
        assert dev.budget.core_budget_ms == base.budget.core_budget_ms
        assert dev.chunking.views.enabled_names() == base.chunking.views.enabled_names()
        assert dev.retrieval.fusion.rrf_k == base.retrieval.fusion.rrf_k
        assert dev.guardrails.domain.min_top1_score == base.guardrails.domain.min_top1_score
        assert dev.embedding.model_id == base.embedding.model_id
        assert dev.rerank.model_id == base.rerank.model_id

    def test_explicit_overrides_apply(self):
        cfg = load_config(overrides={"budget": {"core_budget_ms": 50}})
        assert cfg.budget.core_budget_ms == 50

    def test_missing_file_falls_back_to_model_defaults(self):
        cfg = load_config("does/not/exist.yaml")
        assert isinstance(cfg, Config)

    def test_circular_extends_is_rejected(self, tmp_path):
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text(textwrap.dedent(f"extends: {b}\n"), encoding="utf-8")
        b.write_text(textwrap.dedent(f"extends: {a}\n"), encoding="utf-8")
        with pytest.raises(ValueError, match="circular"):
            load_config(a)


class TestConfigSanity:
    """Config values that must stay mutually consistent, or the system misbehaves
    in ways no single unit test would catch."""

    def test_safety_margin_is_smaller_than_the_budget(self):
        cfg = load_config()
        assert cfg.budget.safety_margin_ms < cfg.budget.core_budget_ms

    def test_required_stages_fit_inside_the_budget(self):
        """If the mandatory stages alone cannot fit, the budget is unachievable
        and no amount of skipping optional stages will save it."""
        cfg = load_config()
        required = sum(
            s.soft_ms for s in cfg.budget.stages.values() if s.required
        )
        assert required < cfg.budget.core_budget_ms - cfg.budget.safety_margin_ms

    def test_top_k_final_is_not_larger_than_the_fused_pool(self):
        cfg = load_config()
        assert cfg.retrieval.top_k_final <= cfg.retrieval.top_k_fused

    def test_rerank_top_n_matches_the_context_size(self):
        cfg = load_config()
        assert cfg.rerank.top_n <= cfg.retrieval.top_k_fused

    def test_fixed_overlap_window_is_larger_than_its_overlap(self):
        cfg = load_config()
        fo = cfg.chunking.views.fixed_overlap
        assert fo.overlap_tokens < fo.chunk_tokens, "stride would be <= 0"

    def test_every_enabled_view_has_a_fusion_weight(self):
        cfg = load_config()
        weights = cfg.retrieval.fusion.view_weights
        for view in cfg.chunking.views.enabled_names():
            assert view in weights, f"view {view} has no fusion weight"

    def test_embedding_max_seq_len_is_within_model_limits(self):
        cfg = load_config()
        assert cfg.embedding.max_seq_len <= 512

    def test_every_budgeted_stage_appears_in_the_latency_report(self):
        """A stage missing from the report's stage list is silently invisible.

        Regression: `generate_semantic` -- the single most expensive optional
        stage -- was budgeted and skipped correctly but absent from
        `STAGE_ORDER`, so the published table showed neither its cost nor the
        fact that it was being dropped.
        """
        import sys
        from pathlib import Path

        bench = Path(__file__).resolve().parents[1] / "bench"
        sys.path.insert(0, str(bench))
        try:
            from run_latency import STAGE_ORDER
        finally:
            sys.path.remove(str(bench))

        missing = set(load_config().budget.stages) - set(STAGE_ORDER)
        assert not missing, f"budgeted but never reported: {sorted(missing)}"
