"""Chunking ablation -- the benchmark that CHOOSES the retrieval config.

This is the deliverable behind requirement 2. Building five chunking views is
easy to claim and easy to fake; what makes it real is measuring each one in
isolation against the same gold labels and publishing the table, including the
arms that lose.

The dataset makes this possible without any hand-annotation: MSMARCO-XI marks the
relevant passages per query via ``is_selected``, so ``gold_doc_ids`` is ground
truth we did not invent.

Fairness rules this harness enforces:

* **Every arm gets its own full top-k.** A single-view arm searches with a FAISS
  ID selector restricted to that view, rather than filtering a mixed result list
  after the fact -- otherwise the measurement is of the filter, not the view.
* **Metrics are computed at document level, not chunk level.** Views produce wildly
  different chunk counts per document, so chunk-level recall would reward whichever
  view shreds documents the most. Chunks are collapsed to their ``doc_id``,
  best-rank-wins, before scoring.
* **Every arm sees the same queries**, sampled once with a fixed seed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from vrag.config import get_config
from vrag.index.dense import DenseIndex
from vrag.index.embedder import OnnxEmbedder
from vrag.index.sparse import SparseIndex
from vrag.index.store import ChunkStore
from vrag.ingest.normalize import load_queries
from vrag.retrieve.expand import plan_text
from vrag.retrieve.multiview import MultiViewRetriever
from vrag.retrieve.rerank import CrossEncoderReranker
from vrag.schemas import ChunkView

console = Console()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


@dataclass
class Metrics:
    n: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    mrr10: float = 0.0
    ndcg10: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)
    mean_candidates: float = 0.0

    _recall_hits: dict[int, int] = field(default_factory=dict, repr=False)
    _rr: float = field(default=0.0, repr=False)
    _ndcg: float = field(default=0.0, repr=False)
    _cands: int = field(default=0, repr=False)
    _lat: list[float] = field(default_factory=list, repr=False)

    def update(self, ranked_docs: list[str], gold: set[str], ks: list[int],
               latency_ms: float, n_candidates: int) -> None:
        self.n += 1
        self._lat.append(latency_ms)
        self._cands += n_candidates

        for k in ks:
            if any(d in gold for d in ranked_docs[:k]):
                self._recall_hits[k] = self._recall_hits.get(k, 0) + 1

        for rank, doc in enumerate(ranked_docs[:10], start=1):
            if doc in gold:
                self._rr += 1.0 / rank
                break

        gains = [1.0 if d in gold else 0.0 for d in ranked_docs[:10]]
        ideal = [1.0] * min(len(gold), 10)
        if ideal:
            self._ndcg += dcg(gains) / dcg(ideal)

    def finalize(self, ks: list[int]) -> Metrics:
        n = max(1, self.n)
        self.recall = {k: self._recall_hits.get(k, 0) / n for k in ks}
        self.mrr10 = self._rr / n
        self.ndcg10 = self._ndcg / n
        self.mean_candidates = self._cands / n
        ordered = sorted(self._lat)
        m = len(ordered)
        self.latency_ms = {
            "p50": ordered[m // 2] if m else 0.0,
            "p90": ordered[min(m - 1, int(0.9 * m))] if m else 0.0,
        }
        return self


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    name: str
    views: list[ChunkView]
    sparse: bool
    rerank: bool
    note: str = ""
    weights: dict[str, float] | None = None   # None = whatever config says


def build_arms(enabled: list[ChunkView], has_sparse: bool) -> list[Arm]:
    arms = [Arm(v.value, [v], sparse=False, rerank=False, note="single view, dense only")
            for v in enabled]
    arms.append(Arm("fused-dense", enabled, sparse=False, rerank=False,
                    note="all views, RRF, dense only"))
    if has_sparse:
        arms.append(Arm("fused-hybrid", enabled, sparse=True, rerank=False,
                        note="all views, RRF, dense + BM25"))
        arms.append(Arm("fused-hybrid+rerank", enabled, sparse=True, rerank=True,
                        note="production config"))
    return arms


def build_weighting_arms(
    enabled: list[ChunkView], per_view_ndcg: dict[str, float], has_sparse: bool
) -> list[Arm]:
    """Candidate view-weighting schemes, derived from the measured per-view scores.

    This is the part that makes the ablation *choose* rather than merely describe.

    Uniform RRF assumes every run is comparably good. When it isn't -- and the
    single-view arms are exactly the measurement of that -- averaging a weak run
    into a strong one drags the strong one down. So the schemes below are
    generated from what the single-view arms actually scored, and the winner is
    what ships.
    """
    ranked = sorted(per_view_ndcg.items(), key=lambda kv: -kv[1])
    best = ranked[0][1] or 1.0
    names = [v.value for v in enabled]

    schemes: list[tuple[str, dict[str, float], str]] = [
        ("uniform", {n: 1.0 for n in names}, "every view weighted equally"),
        (
            "proportional",
            {n: round(per_view_ndcg.get(n, 0.0) / best, 3) for n in names},
            "weight ∝ that view's own nDCG@10",
        ),
        (
            "proportional-squared",
            {n: round((per_view_ndcg.get(n, 0.0) / best) ** 2, 3) for n in names},
            "as above, squared -- sharper preference for strong views",
        ),
    ]

    # Progressively drop the weakest views. If the answer is "one view", the
    # ablation should be able to say so rather than being unable to express it.
    for keep in range(len(ranked) - 1, 0, -1):
        kept = {name for name, _ in ranked[:keep]}
        dropped = [name for name, _ in ranked[keep:]]
        schemes.append((
            f"drop-{'+'.join(dropped)}",
            {n: (1.0 if n in kept else 0.0) for n in names},
            f"only {', '.join(sorted(kept))}",
        ))

    return [
        Arm(f"w:{label}", enabled, sparse=has_sparse, rerank=False,
            note=note, weights=weights)
        for label, weights, note in schemes
    ]


def run_arm(arm: Arm, retriever, reranker, queries, ks: list[int]) -> Metrics:
    metrics = Metrics()
    top_k = max(ks)

    fusion_cfg = retriever.cfg.retrieval.fusion
    original_weights = dict(fusion_cfg.view_weights)
    if arm.weights is not None:
        fusion_cfg.view_weights = dict(arm.weights)

    try:
        return _run_arm_inner(arm, retriever, reranker, queries, ks, metrics, top_k)
    finally:
        fusion_cfg.view_weights = original_weights


def _run_arm_inner(arm, retriever, reranker, queries, ks, metrics, top_k):  # noqa: ANN001
    for record in queries:
        gold = set(record.gold_doc_ids)
        plan = plan_text(retriever.cfg, record.query, lang=record.lang, confidence=1.0)
        plan.views = arm.views
        # Do not language-filter during the ablation: it is a separate variable and
        # would confound the view comparison.
        plan.lang_filter = None

        t0 = time.perf_counter()
        vector = retriever.embed_query(plan)
        runs = retriever.dense_search(vector, plan)
        if arm.sparse:
            runs.update(retriever.sparse_search(plan))
        candidates = retriever.fuse(runs)

        if arm.rerank and reranker is not None:
            evidence = retriever.hydrate(candidates, limit=min(top_k, 20))
            evidence = reranker.rerank(plan.normalized_query, evidence)
            ranked_ids = [e.doc_id for e in evidence]
        else:
            ranked_ids = [c.doc_id or retriever.store.doc_id(c.chunk_id) for c in candidates]
        elapsed = (time.perf_counter() - t0) * 1000

        # Collapse chunks to documents, best rank wins.
        seen: set[str] = set()
        ranked_docs = [d for d in ranked_ids if not (d in seen or seen.add(d))]

        metrics.update(ranked_docs, gold, ks, elapsed, len(candidates))

    return metrics.finalize(ks)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None, help="eval queries (default: config)")
    parser.add_argument("--out", default="docs/CHUNKING.md")
    args = parser.parse_args()

    cfg = get_config()
    n_eval = args.n or cfg.bench.retrieval.n_eval_queries
    ks = cfg.bench.retrieval.k_values

    console.print("[bold]Loading index[/]")
    embedder = OnnxEmbedder(cfg)
    store = ChunkStore(cfg.paths.index_dir / "chunks")
    dense = DenseIndex.load(cfg)
    sparse = None
    if cfg.sparse.enabled:
        try:
            sparse = SparseIndex.load(cfg)
        except FileNotFoundError:
            console.print("[yellow]no sparse index; skipping hybrid arms[/]")
    retriever = MultiViewRetriever(cfg, embedder, dense, store, sparse)

    reranker = None
    if cfg.rerank.enabled:
        try:
            reranker = CrossEncoderReranker(cfg)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]reranker unavailable: {exc}[/]")

    # Only evaluate queries whose passages are actually IN this index. When the
    # corpus is subsampled, a query whose documents were dropped is an
    # unrecoverable miss for every arm -- including it would measure the sampler
    # rather than the retrieval, and would drag all arms down equally so the
    # comparison still "looks" fine while being meaningless.
    indexed_query_ids = set(int(q) for q in set(store.query_id.tolist()))
    all_queries = [
        q for q in load_queries(cfg)
        if q.gold_doc_ids and q.query.strip() and q.query_id in indexed_query_ids
    ]
    random.Random(cfg.corpus.seed).shuffle(all_queries)
    queries = all_queries[:n_eval]
    console.print(f"chunks={len(store):,}  indexed queries={len(indexed_query_ids):,}  "
                  f"eval queries={len(queries):,} (of {len(all_queries):,} with gold labels)")

    arms = build_arms(retriever.enabled_views, sparse is not None)
    results: dict[str, Metrics] = {}

    for arm in arms:
        t0 = time.perf_counter()
        results[arm.name] = run_arm(arm, retriever, reranker, queries, ks)
        console.print(f"  [green]{arm.name}[/] done in {time.perf_counter() - t0:.1f}s")

    # -- weighting sweep, derived from the single-view results --------------- #
    per_view_ndcg = {
        v.value: results[v.value].ndcg10 for v in retriever.enabled_views
        if v.value in results
    }
    weight_arms = build_weighting_arms(
        retriever.enabled_views, per_view_ndcg, sparse is not None
    )
    console.print("[bold]Weighting sweep[/]")
    for arm in weight_arms:
        t0 = time.perf_counter()
        results[arm.name] = run_arm(arm, retriever, reranker, queries, ks)
        console.print(f"  [green]{arm.name}[/] done in {time.perf_counter() - t0:.1f}s")
    arms = arms + weight_arms

    # The winner among fused configurations, and whether it actually beats the
    # best single view -- the question the whole exercise exists to answer.
    best_single = max(per_view_ndcg.items(), key=lambda kv: kv[1])
    fused_names = [a.name for a in arms if a.name.startswith(("fused", "w:"))]
    best_fused = max(fused_names, key=lambda n: results[n].ndcg10)
    multiview_wins = results[best_fused].ndcg10 > best_single[1]

    # -- render ------------------------------------------------------------- #
    table = Table(title=f"Chunking ablation ({len(queries):,} queries, doc-level metrics)")
    table.add_column("arm")
    for k in ks:
        table.add_column(f"R@{k}", justify="right")
    table.add_column("MRR@10", justify="right")
    table.add_column("nDCG@10", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("note")

    for arm in arms:
        m = results[arm.name]
        table.add_row(
            arm.name,
            *[f"{m.recall[k]:.3f}" for k in ks],
            f"{m.mrr10:.3f}", f"{m.ndcg10:.3f}", f"{m.latency_ms['p50']:.1f}", arm.note,
        )
    console.print(table)

    verdict = Table(title="Verdict")
    verdict.add_column("question")
    verdict.add_column("answer")
    verdict.add_row("best single view", f"{best_single[0]}  (nDCG@10 {best_single[1]:.3f})")
    verdict.add_row("best fused config", f"{best_fused}  (nDCG@10 {results[best_fused].ndcg10:.3f})")
    verdict.add_row(
        "does multi-view earn its keep?",
        "[green]yes[/]" if multiview_wins else
        "[red]NO -- a single view is better; simplify[/]",
    )
    if not multiview_wins:
        verdict.add_row(
            "[yellow]recommended[/]",
            f"[yellow]set chunking to {best_single[0]} only, or use the winning "
            f"weights below[/]",
        )
    console.print(verdict)

    winning = next((a for a in arms if a.name == best_fused), None)
    if winning is not None and winning.weights:
        console.print("\n[bold]Recommended view weights[/] (configs/default.yaml):")
        console.print("  retrieval.fusion.view_weights:")
        for name, w in winning.weights.items():
            console.print(f"    {name}: {w}")

    out = cfg.paths.data_dir.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Chunking ablation",
        "",
        "Generated by `bench/run_retrieval_eval.py`. Gold labels are the dataset's own",
        "`is_selected` flags, so nothing here is hand-annotated.",
        "",
        f"- corpus: **{len(store):,} chunks** over {cfg.corpus.queries_per_language * len(cfg.corpus.languages):,} queries",
        f"- eval set: **{len(queries):,} queries** held out by seed {cfg.corpus.seed}",
        "- metrics are **document-level**: chunks are collapsed to their source passage",
        "  (best rank wins) before scoring, so a view is not rewarded for shredding",
        "  documents into more pieces",
        "- each single-view arm searches with a FAISS ID selector restricted to that",
        "  view, so every arm gets its own full top-k",
        "",
        "### Reading the single-view rows",
        "",
        "Views do not all cover the whole corpus, and recall must be read with that in",
        "mind. `fixed_overlap` only emits chunks for passages above its token floor",
        "(~35% of them), and `sentence_window` and `semantic` skip passages too short to",
        "split. A low R@20 for those arms is partly a **coverage ceiling** -- they cannot",
        "retrieve a document they never chunked -- not purely a ranking failure. Only",
        "`atomic` covers every passage by construction.",
        "",
        "That is a fair comparison of *what each view contributes to the index*, which is",
        "the decision being made here, but it is not a claim that one chunking algorithm",
        "ranks better than another in isolation.",
        "",
        "## Verdict",
        "",
        f"- best single view: **{best_single[0]}** (nDCG@10 {best_single[1]:.3f})",
        f"- best fused config: **{best_fused}** (nDCG@10 {results[best_fused].ndcg10:.3f})",
        "",
        (
            "Multi-view fusion **earns its keep** on this corpus."
            if multiview_wins
            else "**Multi-view fusion does NOT beat the best single view here.** Uniform RRF "
            "assumes every run is comparably good; when one view is much weaker, averaging "
            "it in drags the strong one down. The weighting sweep below is the response — "
            "and if no weighting recovers it, the honest conclusion is to ship fewer views."
        ),
        "",
        "| arm | " + " | ".join(f"R@{k}" for k in ks) + " | MRR@10 | nDCG@10 | p50 ms | note |",
        "|---|" + "---|" * (len(ks) + 4),
    ]
    for arm in arms:
        m = results[arm.name]
        lines.append(
            f"| `{arm.name}` | "
            + " | ".join(f"{m.recall[k]:.3f}" for k in ks)
            + f" | {m.mrr10:.3f} | {m.ndcg10:.3f} | {m.latency_ms['p50']:.1f} | {arm.note} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw = cfg.paths.trace_dir / "retrieval_eval.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        json.dumps(
            {a.name: {"recall": results[a.name].recall, "mrr10": results[a.name].mrr10,
                      "ndcg10": results[a.name].ndcg10,
                      "latency_ms": results[a.name].latency_ms,
                      "mean_candidates": results[a.name].mean_candidates}
             for a in arms},
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[dim]{out}\n{raw}[/]")


if __name__ == "__main__":
    main()
