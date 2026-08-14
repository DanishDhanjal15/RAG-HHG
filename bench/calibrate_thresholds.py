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
    population: str


def collect(
    retriever: MultiViewRetriever, texts: list[tuple[str, str]],
    in_domain: bool, population: str,
) -> list[Sample]:
    samples: list[Sample] = []
    for text, lang in texts:
        plan = plan_text(retriever.cfg, text, lang=lang, confidence=1.0)
        plan.lang_filter = None
        vector = retriever.embed_query(plan)
        retriever.dense_search(vector, plan)
        samples.append(
            Sample(
                top1=plan.top_dense_score,
                centroid_distance=retriever.centroid_distance(vector),
                in_domain=in_domain,
                population=population,
            )
        )
    return samples


def auc(positive: list[float], negative: list[float]) -> float:
    """Probability that a random positive scores below a random negative.

    Used as a blunt measure of whether a signal discriminates AT ALL. 0.5 means
    the two distributions are indistinguishable and the signal is worthless --
    which is exactly what this script found for one of them, and the reason it is
    reported rather than assumed.
    """
    if not positive or not negative:
        return 0.5
    pos = np.asarray(positive)
    neg = np.asarray(negative)
    comparisons = (pos[:, None] < neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(comparisons / (len(pos) * len(neg)))


def load_offtopic_cases() -> list[tuple[str, str]]:
    """Genuinely off-topic questions from the adversarial suite.

    A different population from "real question we happen not to have indexed",
    and the one the domain guard is actually for.
    """
    import yaml

    path = Path(__file__).parent / "adversarial_queries.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        (row["text"], row.get("lang", "en"))
        for row in (raw.get("must_refuse") or [])
        if row.get("expect") == "OUT_OF_DOMAIN" and row.get("text", "").strip()
    ]


def evaluate(
    samples: list[Sample], min_top1: float, max_dist: float | None
) -> dict[str, float]:
    """Score an operating point.

    ``max_dist=None`` evaluates the **cosine-only** rule. That option exists
    because the centroid signal turned out not to discriminate, and a guard is
    better off with one honest signal than two where the second only adds a
    condition that is always true.
    """
    tp = fp = tn = fn = 0
    for s in samples:
        weak = s.top1 < min_top1
        refused = weak if max_dist is None else (weak and s.centroid_distance > max_dist)
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
        "max_dist": -1.0 if max_dist is None else max_dist,
        "cosine_only": 1.0 if max_dist is None else 0.0,
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
    in_texts = [(q.query, q.lang) for q in in_domain[: args.n]]
    unanswerable_texts = [(q.query, q.lang) for q in out_domain[: args.n]]
    offtopic_texts = load_offtopic_cases()

    console.print(
        f"in-domain: {len(in_texts)}   "
        f"unanswerable: {len(unanswerable_texts)} (real questions, documents not indexed)   "
        f"off-topic: {len(offtopic_texts)} (from the adversarial suite)"
    )

    in_samples = collect(retriever, in_texts, True, "in_domain")
    unanswerable = collect(retriever, unanswerable_texts, False, "unanswerable")
    offtopic = collect(retriever, offtopic_texts, False, "offtopic")

    # -- distributions ------------------------------------------------------ #
    dist_table = Table(title="Signal distributions")
    dist_table.add_column("signal"); dist_table.add_column("population")
    for col in ("n", "p05", "p25", "p50", "p75", "p95"):
        dist_table.add_column(col, justify="right")

    stats = {}
    populations = [("in-domain", in_samples), ("unanswerable", unanswerable),
                   ("off-topic", offtopic)]
    for signal, getter in (("top1 cosine", lambda s: s.top1),
                           ("centroid dist", lambda s: s.centroid_distance)):
        for label, group in populations:
            values = [getter(s) for s in group]
            if not values:
                continue
            d = describe(values, f"{signal}/{label}")
            stats[d["label"]] = d
            dist_table.add_row(
                signal, label, str(d["n"]),
                *[f"{d[k]:.3f}" for k in ("p05", "p25", "p50", "p75", "p95")],
            )
    console.print(dist_table)

    # -- how much does each signal actually discriminate? -------------------- #
    disc = Table(title="Discriminative power (AUC; 0.50 = useless)")
    disc.add_column("signal"); disc.add_column("vs unanswerable", justify="right")
    disc.add_column("vs off-topic", justify="right")
    aucs = {
        "top1 cosine": (
            auc([s.top1 for s in unanswerable], [s.top1 for s in in_samples]),
            auc([s.top1 for s in offtopic], [s.top1 for s in in_samples]),
        ),
        "centroid distance": (
            auc([-s.centroid_distance for s in unanswerable],
                [-s.centroid_distance for s in in_samples]),
            auc([-s.centroid_distance for s in offtopic],
                [-s.centroid_distance for s in in_samples]),
        ),
    }
    for name, (a_unans, a_off) in aucs.items():
        disc.add_row(name, f"{a_unans:.3f}", f"{a_off:.3f}")
    console.print(disc)

    # -- sweep -------------------------------------------------------------- #
    # Calibrated against the OFF-TOPIC population, because that is what a domain
    # guard is for. The unanswerable population is reported but not optimised
    # against: separating "real question we don't have the document for" from
    # "real question we do" is a grounding problem, not a domain problem.
    sweep_samples = in_samples + offtopic
    top1_grid = [round(v, 3) for v in np.arange(0.60, 0.96, 0.005)]
    dist_grid: list[float | None] = [None] + [
        round(v, 3) for v in np.arange(0.05, 0.50, 0.01)
    ]
    results = [evaluate(sweep_samples, t, d) for t in top1_grid for d in dist_grid]

    feasible = [r for r in results if r["over_refusal"] <= args.max_over_refusal]
    if feasible:
        # Prefer the cosine-only rule on ties: a second condition that is always
        # true is a false sense of rigour, not extra safety.
        best = max(feasible, key=lambda r: (r["recall"], r["cosine_only"], r["f1"]))
        rationale = (f"maximises off-topic recall subject to over-refusal "
                     f"<= {args.max_over_refusal:.0%}")
    else:
        best = max(results, key=lambda r: r["f1"])
        rationale = ("no operating point met the over-refusal ceiling; "
                     "falling back to best F1")
        console.print(f"[yellow]{rationale}[/]")

    best_f1 = max(results, key=lambda r: r["f1"])
    samples = in_samples + unanswerable + offtopic
    unans_at_best = evaluate(
        in_samples + unanswerable, best["min_top1"],
        None if best["cosine_only"] else best["max_dist"],
    )

    rec = Table(title="Recommended operating point")
    rec.add_column("setting"); rec.add_column("value", justify="right")
    rec.add_row("guardrails.domain.min_top1_score", f"{best['min_top1']:.3f}")
    rec.add_row(
        "guardrails.domain.use_centroid",
        "false (signal does not discriminate)" if best["cosine_only"] else "true",
    )
    if not best["cosine_only"]:
        rec.add_row("guardrails.domain.max_centroid_distance", f"{best['max_dist']:.3f}")
    rec.add_row("", "")
    rec.add_row("off-topic recall", f"{best['recall']:.1%}")
    rec.add_row("over-refusal rate", f"{best['over_refusal']:.1%}")
    rec.add_row("precision", f"{best['precision']:.1%}")
    rec.add_row("F1", f"{best['f1']:.3f}")
    rec.add_row("", "")
    rec.add_row("[dim]unanswerable recall at this point[/]",
                f"[dim]{unans_at_best['recall']:.1%}[/]")
    console.print(rec)
    console.print(f"[dim]{rationale}[/]")

    current = evaluate(
        in_samples + offtopic,
        cfg.guardrails.domain.min_top1_score,
        cfg.guardrails.domain.max_centroid_distance,
    )
    console.print(
        f"\n[bold]Current config[/] (min_top1={cfg.guardrails.domain.min_top1_score}, "
        f"max_dist={cfg.guardrails.domain.max_centroid_distance}) on off-topic: "
        + json.dumps({k: round(float(v), 4) for k, v in current.items()
                      if k in ("recall", "over_refusal", "f1")})
    )

    _write(cfg, args.out, stats, best, best_f1, rationale, args, samples,
           aucs, unans_at_best, current)


def _write(cfg, out_rel, stats, best, best_f1, rationale, args, samples,  # noqa: ANN001
           aucs, unans_at_best, current) -> None:
    cosine_only = bool(best["cosine_only"])
    lines = [
        "# Out-of-domain threshold calibration",
        "",
        "Generated by `bench/calibrate_thresholds.py`.",
        "",
        "## Why calibrate at all",
        "",
        "With `multilingual-e5`, clearly unrelated text still scores around **0.70-0.85**",
        "cosine against a query. An intuition like \"below 0.5 means unrelated\" is simply",
        "wrong for this embedding space: a threshold picked that way never fires. Every",
        "number below is measured.",
        "",
        "## Two different problems that look like one",
        "",
        "Calibration surfaced a distinction the original design missed. There are two",
        "populations a RAG system can fail to answer, and they are **not** the same:",
        "",
        "| population | what it is | example |",
        "|---|---|---|",
        "| **off-topic** | not what this corpus is about | *\"what is the capital of Mars\"* |",
        "| **unanswerable** | exactly what this corpus is about, but the document isn't indexed | a real MS MARCO question whose passages were left out |",
        "",
        "The unanswerable set here is not synthetic: subsampling the index **by query**",
        "leaves thousands of real MSMARCO-XI questions whose documents were never indexed.",
        "",
        "## Measured distributions",
        "",
        "| signal | population | n | p05 | p25 | p50 | p75 | p95 |",
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
        "## Does each signal actually discriminate?",
        "",
        "AUC, where 0.50 means the distributions are indistinguishable and the signal is",
        "worthless:",
        "",
        "| signal | vs unanswerable | vs off-topic |",
        "|---|---:|---:|",
    ]
    for name, (a_unans, a_off) in aucs.items():
        lines.append(f"| {name} | {a_unans:.3f} | {a_off:.3f} |")

    lines += [
        "",
        "### The negative result",
        "",
        "**Centroid distance does not discriminate.** The original design required *both*",
        "top-1 similarity and distance-from-corpus-centroid to fail before refusing, on",
        "the theory that each covers the other's blind spot. Measurement says the second",
        "signal has no blind spot to cover because it has no sight: in-domain and",
        "out-of-domain queries sit at essentially the same distance from the centroid.",
        "",
        "In hindsight the reason is obvious. The centroid is the mean of the corpus, and",
        "every query in every population is a web-search-style question drawn from the",
        "same distribution. The signal measures *\"does this look like an MS MARCO",
        "question\"* -- to which the answer is always yes -- not *\"can this corpus answer",
        "it\"*.",
        "",
        "Requiring a second condition that is always true does not make a guard safer. It",
        "makes it a one-signal guard wearing a second signal as decoration. So the",
        "recommendation below "
        + ("**drops it**." if cosine_only else "retains it only where it measurably helps."),
        "",
        "### The unanswerable population: partly detectable, and that was a surprise",
        "",
        "The working hypothesis was that unanswerable-but-in-distribution questions",
        "would be invisible to a domain guard, since the query is genuinely in-domain",
        "and only the evidence is missing. The AUC says otherwise: top-1 cosine",
        f"separates them from in-domain queries at **{aucs['top1 cosine'][0]:.3f}**, and at the",
        f"recommended operating point the guard catches **{unans_at_best['recall']:.1%}** of them.",
        "",
        "The reason it works at all: a query whose own passages were never indexed has",
        "no *near*-duplicate in the corpus, only topical neighbours, and that shows up as",
        "a measurably lower best-match cosine. It is a weaker signal than for off-topic",
        "queries (which are further away still), but it is not noise.",
        "",
        "It is not a complete defence, and it is not meant to be. The remaining",
        f"{1 - unans_at_best['recall']:.0%} is what the **grounding guard** is for: after generation, checking",
        "that the answer is actually attributable to retrieved text catches the cases",
        "where retrieval returned plausible-looking but non-answering passages.",
        "",
        "## Recommended operating point",
        "",
        f"_{rationale}._",
        "",
        "```yaml",
        "guardrails:",
        "  domain:",
        f"    min_top1_score: {best['min_top1']:.3f}",
        f"    use_centroid: {'false' if cosine_only else 'true'}",
    ]
    if not cosine_only:
        lines.append(f"    max_centroid_distance: {best['max_dist']:.3f}")
    lines += [
        "```",
        "",
        "| | recommended | best-F1 | currently configured |",
        "|---|---:|---:|---:|",
        f"| min_top1_score | {best['min_top1']:.3f} | {best_f1['min_top1']:.3f} | {cfg.guardrails.domain.min_top1_score:.3f} |",
        f"| off-topic recall | {best['recall']:.1%} | {best_f1['recall']:.1%} | {current['recall']:.1%} |",
        f"| over-refusal | {best['over_refusal']:.1%} | {best_f1['over_refusal']:.1%} | {current['over_refusal']:.1%} |",
        f"| F1 | {best['f1']:.3f} | {best_f1['f1']:.3f} | {current['f1']:.3f} |",
        "",
        "## On the objective",
        "",
        "The recommendation maximises off-topic recall subject to over-refusal staying",
        f"under {args.max_over_refusal:.0%}, rather than maximising F1. F1 treats both errors as equally",
        "costly; they are not. Wrongly refusing a legitimate question is the failure a",
        "user notices and cannot work around, so it gets a hard ceiling and the other",
        "error is minimised underneath it.",
    ]

    out = cfg.paths.data_dir.parent / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw = Path(cfg.paths.trace_dir) / "calibration.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        json.dumps(
            {
                "recommended": best,
                "best_f1": best_f1,
                "current": current,
                "unanswerable_at_recommended": unans_at_best,
                "auc": {k: {"vs_unanswerable": v[0], "vs_offtopic": v[1]}
                        for k, v in aucs.items()},
                "distributions": stats,
            },
            indent=2,
            # numpy scalars leak in from the threshold grid; without this the whole
            # run dies at the final write after doing all the work.
            default=float,
        ),
        encoding="utf-8",
    )
    console.print(f"[dim]{out}\n{raw}[/]")


if __name__ == "__main__":
    main()
