"""Calibrate the out-of-domain guard's thresholds against measured distributions.

A hand-picked cosine threshold is a guess about a distribution nobody looked at.
With `multilingual-e5`, even *clearly unrelated* text scores ~0.70 cosine against
a query -- so an intuition like "0.5 means unrelated" is simply wrong, and a
guard built on it either never fires or never stops firing.

**Where the out-of-domain set comes from.** Not synthetic, and not hand-written:
the index is subsampled by query (see `index/build.py:subsample_by_query`), which
leaves thousands of *real* MSMARCO-XI questions whose documents were never
indexed. Those are genuine, natural, in-distribution questions that this corpus
truly cannot answer -- exactly the population the guard has to recognise, and far
more honest than asking about the capital of Mars.

The script sweeps both thresholds jointly (the guard requires **both** signals to
fail before refusing, so they cannot be tuned independently) and reports the
frontier, then recommends the operating point that maximises out-of-domain recall
subject to a ceiling on over-refusal -- because refusing a legitimate question is
the more expensive error for a demo.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from vrag.config import get_config
from vrag.index.dense import DenseIndex
from vrag.index.embedder import OnnxEmbedder
from vrag.index.store import ChunkStore
from vrag.ingest.normalize import load_queries
from vrag.retrieve.expand import plan_text
from vrag.retrieve.multiview import MultiViewRetriever

console = Console()


@dataclass
class Sample:
    top1: float
    centroid_distance: float
    in_domain: bool


def collect(retriever: MultiViewRetriever, queries, in_domain: bool) -> list[Sample]:
    samples: list[Sample] = []
    for record in queries:
        plan = plan_text(retriever.cfg, record.query, lang=record.lang, confidence=1.0)
        plan.lang_filter = None
        vector = retriever.embed_query(plan)
        retriever.dense_search(vector, plan)
        samples.append(
            Sample(
                top1=plan.top_dense_score,
                centroid_distance=retriever.centroid_distance(vector),
                in_domain=in_domain,
            )
        )
    return samples


def evaluate(samples: list[Sample], min_top1: float, max_dist: float) -> dict[str, float]:
    """The guard refuses only when BOTH signals fail."""
    tp = fp = tn = fn = 0
    for s in samples:
        refused = (s.top1 < min_top1) and (s.centroid_distance > max_dist)
        if not s.in_domain:
            tp += refused
            fn += not refused
        else:
            fp += refused
            tn += not refused

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "min_top1": min_top1,
        "max_dist": max_dist,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "over_refusal": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def describe(values: list[float], label: str) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "label": label,
        "n": len(arr),
        "mean": float(arr.mean()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400, help="samples per class")
    parser.add_argument("--max-over-refusal", type=float, default=0.05,
                        help="ceiling on wrongly-refused legitimate questions")
    parser.add_argument("--out", default="docs/CALIBRATION.md")
    args = parser.parse_args()

    cfg = get_config()
    console.print("[bold]Loading index[/]")
    embedder = OnnxEmbedder(cfg)
    store = ChunkStore(cfg.paths.index_dir / "chunks")
    retriever = MultiViewRetriever(cfg, embedder, DenseIndex.load(cfg), store)

    if retriever.centroid is None:
        console.print("[red]No corpus centroid found -- rerun `vrag build`.[/]")
        return

    indexed = set(int(q) for q in set(store.query_id.tolist()))
    all_queries = [q for q in load_queries(cfg) if q.query.strip()]
    rng = random.Random(cfg.corpus.seed + 7)

    in_domain = [q for q in all_queries if q.query_id in indexed]
    out_domain = [q for q in all_queries if q.query_id not in indexed]

    if len(out_domain) < 50:
        console.print(
            "[red]The whole corpus is indexed, so there is no natural out-of-domain "
            "set. Rerun with corpus.max_queries set.[/]"
        )
        return

    rng.shuffle(in_domain)
    rng.shuffle(out_domain)
    in_domain = in_domain[: args.n]
    out_domain = out_domain[: args.n]

    console.print(f"in-domain: {len(in_domain)}   out-of-domain: {len(out_domain)} "
                  f"(real questions whose documents were never indexed)")

    samples = collect(retriever, in_domain, True) + collect(retriever, out_domain, False)

    # -- distributions ------------------------------------------------------ #
    dist_table = Table(title="Signal distributions")
    dist_table.add_column("signal"); dist_table.add_column("class")
    for col in ("n", "p05", "p25", "p50", "p75", "p95"):
        dist_table.add_column(col, justify="right")

    stats = {}
    for signal, getter in (("top1 cosine", lambda s: s.top1),
                           ("centroid distance", lambda s: s.centroid_distance)):
        for label, flag in (("in-domain", True), ("out-of-domain", False)):
            values = [getter(s) for s in samples if s.in_domain is flag]
            d = describe(values, f"{signal}/{label}")
            stats[d["label"]] = d
            dist_table.add_row(
                signal, label, str(d["n"]),
                *[f"{d[k]:.3f}" for k in ("p05", "p25", "p50", "p75", "p95")],
            )
    console.print(dist_table)
    console.print(
        "[dim]Note how high the out-of-domain cosines sit -- that overlap is exactly "
        "why one signal alone cannot separate the classes.[/]"
    )

    # -- sweep -------------------------------------------------------------- #
    top1_grid = [round(v, 3) for v in np.arange(0.60, 0.95, 0.01)]
    dist_grid = [round(v, 3) for v in np.arange(0.20, 0.90, 0.02)]
    results = [evaluate(samples, t, d) for t in top1_grid for d in dist_grid]

    feasible = [r for r in results if r["over_refusal"] <= args.max_over_refusal]
    if feasible:
        best = max(feasible, key=lambda r: (r["recall"], r["f1"]))
        rationale = (f"maximises out-of-domain recall subject to over-refusal "
                     f"<= {args.max_over_refusal:.0%}")
    else:
        best = max(results, key=lambda r: r["f1"])
        rationale = ("no operating point met the over-refusal ceiling; "
                     "falling back to best F1")
        console.print(f"[yellow]{rationale}[/]")

    best_f1 = max(results, key=lambda r: r["f1"])

    rec = Table(title="Recommended operating point")
    rec.add_column("setting"); rec.add_column("value", justify="right")
    rec.add_row("guardrails.domain.min_top1_score", f"{best['min_top1']:.3f}")
    rec.add_row("guardrails.domain.max_centroid_distance", f"{best['max_dist']:.3f}")
    rec.add_row("", "")
    rec.add_row("out-of-domain recall", f"{best['recall']:.1%}")
    rec.add_row("over-refusal rate", f"{best['over_refusal']:.1%}")
    rec.add_row("precision", f"{best['precision']:.1%}")
    rec.add_row("F1", f"{best['f1']:.3f}")
    console.print(rec)
    console.print(f"[dim]{rationale}[/]")
    console.print(
        f"\n[bold]Current config:[/] min_top1={cfg.guardrails.domain.min_top1_score} "
        f"max_dist={cfg.guardrails.domain.max_centroid_distance} -> "
        + json.dumps(
            {k: round(v, 4) for k, v in
             evaluate(samples, cfg.guardrails.domain.min_top1_score,
                      cfg.guardrails.domain.max_centroid_distance).items()
             if k in ("recall", "over_refusal", "f1")}
        )
    )

    _write(cfg, args.out, stats, best, best_f1, rationale, args, samples)


def _write(cfg, out_rel, stats, best, best_f1, rationale, args, samples) -> None:  # noqa: ANN001
    current = evaluate(samples, cfg.guardrails.domain.min_top1_score,
                       cfg.guardrails.domain.max_centroid_distance)
    lines = [
        "# Out-of-domain threshold calibration",
        "",
        "Generated by `bench/calibrate_thresholds.py`.",
        "",
        "## Why calibrate at all",
        "",
        "With `multilingual-e5`, clearly unrelated text still scores around **0.70**",
        "cosine against a query. An intuition like \"below 0.5 means unrelated\" is",
        "simply wrong for this embedding space, and a threshold picked that way either",
        "never fires or never stops firing. The numbers below are measured.",
        "",
        "## Where the out-of-domain set comes from",
        "",
        "Not synthetic. The index is subsampled **by query**, which leaves thousands of",
        "real MSMARCO-XI questions whose documents were never indexed -- genuine,",
        "in-distribution questions this corpus cannot answer. That is the population the",
        "guard actually has to recognise.",
        "",
        "## Measured distributions",
        "",
        "| signal | class | n | p05 | p25 | p50 | p75 | p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, d in stats.items():
        signal, cls = label.split("/")
        lines.append(
            f"| {signal} | {cls} | {d['n']} | "
            + " | ".join(f"{d[k]:.3f}" for k in ("p05", "p25", "p50", "p75", "p95"))
            + " |"
        )

    lines += [
        "",
        "The two classes **overlap substantially** on either signal alone. That overlap",
        "is the entire argument for requiring both signals to fail before refusing:",
        "top-1 similarity alone is fooled by a query that shares vocabulary with one",
        "unrelated passage, and centroid distance alone is fooled by an in-domain",
        "question phrased unusually.",
        "",
        "## Recommended operating point",
        "",
        f"_{rationale}._",
        "",
        "```yaml",
        "guardrails:",
        "  domain:",
        f"    min_top1_score: {best['min_top1']:.3f}",
        f"    max_centroid_distance: {best['max_dist']:.3f}",
        "```",
        "",
        "| | recommended | best-F1 | currently configured |",
        "|---|---:|---:|---:|",
        f"| min_top1_score | {best['min_top1']:.3f} | {best_f1['min_top1']:.3f} | {cfg.guardrails.domain.min_top1_score:.3f} |",
        f"| max_centroid_distance | {best['max_dist']:.3f} | {best_f1['max_dist']:.3f} | {cfg.guardrails.domain.max_centroid_distance:.3f} |",
        f"| out-of-domain recall | {best['recall']:.1%} | {best_f1['recall']:.1%} | {current['recall']:.1%} |",
        f"| over-refusal | {best['over_refusal']:.1%} | {best_f1['over_refusal']:.1%} | {current['over_refusal']:.1%} |",
        f"| F1 | {best['f1']:.3f} | {best_f1['f1']:.3f} | {current['f1']:.3f} |",
        "",
        "## On the objective",
        "",
        f"The recommendation maximises out-of-domain recall subject to over-refusal",
        f"staying under {args.max_over_refusal:.0%}, rather than maximising F1. F1 treats both",
        "errors as equally costly; they are not. Wrongly refusing a legitimate question",
        "is the failure a user notices and cannot work around, so it gets a hard ceiling",
        "and the other error is minimised underneath it.",
    ]

    out = cfg.paths.data_dir.parent / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw = Path(cfg.paths.trace_dir) / "calibration.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({"recommended": best, "best_f1": best_f1,
                               "current": current, "distributions": stats}, indent=2),
                   encoding="utf-8")
    console.print(f"[dim]{out}\n{raw}[/]")


if __name__ == "__main__":
    main()
