"""Guardrail evaluation -- does the system know when NOT to answer?

Requirement 6 asks the system to show it knows when not to answer. The easy half
is refusing obviously bad input; the hard half, and the one this harness is built
around, is **not refusing everything else**.

So the suite has two sides and both are scored:

* ``must_refuse`` -- unsafe requests, prompt injections, out-of-domain questions,
  and empty/degenerate transcripts. Missing one is a **false negative**.
* ``must_answer`` -- in-domain questions, several of which deliberately contain
  alarming vocabulary in entirely legitimate contexts ("kill a process", "bomb
  cyclone", "food poisoning", "ethical hacking") or phrasing that superficially
  resembles injection ("ignore the noise", "what is a system prompt"). Refusing
  one is a **false positive**.

Reporting only the first half is how paranoid systems get shipped looking safe.
The false-positive rate is the number that decides whether these guardrails are
calibrated or merely loud, so it is printed first and never averaged away.

Also reported: whether the system refused for the RIGHT reason. Catching an
injection attempt via the out-of-domain threshold is a lucky accident, not a
working injection guard, and the confusion matrix makes that visible.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from vrag.config import get_config
from vrag.harness.pipeline import Pipeline

console = Console()
SUITE = Path(__file__).parent / "adversarial_queries.yaml"


@dataclass
class Case:
    text: str
    lang: str
    expect: str | None
    tag: str
    side: str  # "must_refuse" | "must_answer"


@dataclass
class Outcome:
    case: Case
    refused: bool
    reason: str | None
    detail: str

    @property
    def correct_decision(self) -> bool:
        return self.refused == (self.case.expect is not None)

    @property
    def correct_reason(self) -> bool:
        return self.correct_decision and (
            self.case.expect is None or self.reason == self.case.expect
        )


@dataclass
class Counts:
    tp: int = 0   # correctly refused
    fp: int = 0   # refused something it should have answered
    tn: int = 0   # correctly answered
    fn: int = 0   # answered something it should have refused
    wrong_reason: int = 0
    by_tag: dict[str, list[bool]] = field(default_factory=lambda: defaultdict(list))

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Over-refusal: share of legitimate questions wrongly declined."""
        total = self.fp + self.tn
        return self.fp / total if total else 0.0


def load_cases() -> list[Case]:
    raw = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    cases: list[Case] = []
    for side in ("must_refuse", "must_answer"):
        for row in raw.get(side) or []:
            cases.append(
                Case(
                    text=row.get("text", ""),
                    lang=row.get("lang", "en"),
                    expect=row.get("expect"),
                    tag=row.get("tag", "untagged"),
                    side=side,
                )
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/GUARDRAILS.md")
    args = parser.parse_args()

    cfg = get_config()
    console.print("[bold]Booting pipeline[/]")
    pipeline = Pipeline(cfg)

    cases = load_cases()
    console.print(
        f"{len(cases)} cases  "
        f"({sum(c.side == 'must_refuse' for c in cases)} must-refuse, "
        f"{sum(c.side == 'must_answer' for c in cases)} must-answer)"
    )

    outcomes: list[Outcome] = []
    counts = Counts()

    for case in cases:
        env = pipeline.answer_text(case.text, lang=case.lang)
        outcome = Outcome(
            case=case,
            refused=env.abstained,
            reason=env.refusal_reason.value if env.refusal_reason else None,
            detail=env.refusal_detail or env.answer[:120],
        )
        outcomes.append(outcome)

        should_refuse = case.expect is not None
        if should_refuse and outcome.refused:
            counts.tp += 1
            if not outcome.correct_reason:
                counts.wrong_reason += 1
        elif should_refuse and not outcome.refused:
            counts.fn += 1
        elif not should_refuse and outcome.refused:
            counts.fp += 1
        else:
            counts.tn += 1

        counts.by_tag[case.tag].append(outcome.correct_decision)

    # -- headline ----------------------------------------------------------- #
    summary = Table(title="Guardrail performance")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_column("meaning")
    summary.add_row("over-refusal rate", f"{counts.false_positive_rate:.1%}",
                    "legitimate questions wrongly declined [the number that matters]")
    summary.add_row("recall", f"{counts.recall:.1%}", "of things that should be refused, how many were")
    summary.add_row("precision", f"{counts.precision:.1%}", "of refusals, how many were correct")
    summary.add_row("F1", f"{counts.f1:.3f}", "")
    summary.add_row("wrong reason", f"{counts.wrong_reason}",
                    "refused correctly but attributed to the wrong guard")
    console.print(summary)

    matrix = Table(title="Confusion matrix")
    matrix.add_column("")
    matrix.add_column("system refused", justify="right")
    matrix.add_column("system answered", justify="right")
    matrix.add_row("should refuse", f"[green]{counts.tp}[/]", f"[red]{counts.fn}[/]")
    matrix.add_row("should answer", f"[red]{counts.fp}[/]", f"[green]{counts.tn}[/]")
    console.print(matrix)

    # -- failures ----------------------------------------------------------- #
    failures = [o for o in outcomes if not o.correct_decision]
    if failures:
        ftable = Table(title=f"{len(failures)} incorrect decisions")
        ftable.add_column("side")
        ftable.add_column("tag")
        ftable.add_column("query")
        ftable.add_column("expected")
        ftable.add_column("got")
        for o in failures:
            ftable.add_row(
                o.case.side.replace("must_", ""),
                o.case.tag,
                (o.case.text[:44] or "<empty>"),
                o.case.expect or "answer",
                o.reason or "answered",
            )
        console.print(ftable)
    else:
        console.print("[bold green]All decisions correct.[/]")

    mis = [o for o in outcomes if o.correct_decision and not o.correct_reason]
    if mis:
        mtable = Table(title=f"{len(mis)} refused for the wrong reason")
        mtable.add_column("query"); mtable.add_column("expected"); mtable.add_column("got")
        for o in mis:
            mtable.add_row(o.case.text[:44] or "<empty>", o.case.expect or "-", o.reason or "-")
        console.print(mtable)

    _write_report(cfg, args.out, counts, outcomes, failures, mis)
    pipeline.close()


def _write_report(cfg, out_rel, counts, outcomes, failures, mis) -> None:  # noqa: ANN001
    lines = [
        "# Guardrails",
        "",
        "Generated by `bench/run_guardrail_eval.py` against `bench/adversarial_queries.yaml`.",
        "",
        "## Why the control set exists",
        "",
        "Any system can refuse everything. What makes a guardrail useful is refusing",
        "the right things **and leaving the rest alone**, so the suite carries a",
        "`must_answer` control set alongside the adversarial cases -- including",
        "legitimate questions that contain alarming words (\"kill a process\", \"bomb",
        "cyclone\", \"food poisoning\", \"ethical hacking\") and phrasing that resembles",
        "injection (\"ignore the noise\", \"what is a system prompt\"). Those are exactly",
        "where a lexicon-based layer fails, and the failure is invisible unless tested.",
        "",
        "## Results",
        "",
        "| metric | value | meaning |",
        "|---|---:|---|",
        f"| **over-refusal rate** | **{counts.false_positive_rate:.1%}** | legitimate questions wrongly declined |",
        f"| recall | {counts.recall:.1%} | of things that should be refused, how many were |",
        f"| precision | {counts.precision:.1%} | of refusals, how many were correct |",
        f"| F1 | {counts.f1:.3f} | |",
        f"| refused for the wrong reason | {counts.wrong_reason} | correct decision, wrong guard credited |",
        "",
        "### Confusion matrix",
        "",
        "| | system refused | system answered |",
        "|---|---:|---:|",
        f"| **should refuse** | {counts.tp} | {counts.fn} |",
        f"| **should answer** | {counts.fp} | {counts.tn} |",
        "",
        "### Per-category accuracy",
        "",
        "| tag | correct | n |",
        "|---|---:|---:|",
    ]
    for tag, results in sorted(counts.by_tag.items()):
        lines.append(f"| `{tag}` | {sum(results)}/{len(results)} | {len(results)} |")

    if failures:
        lines += ["", "### Incorrect decisions", "",
                  "| side | tag | query | expected | got |", "|---|---|---|---|---|"]
        for o in failures:
            q = (o.case.text[:60] or "&lt;empty&gt;").replace("|", "\\|")
            lines.append(
                f"| {o.case.side.replace('must_', '')} | `{o.case.tag}` | {q} | "
                f"`{o.case.expect or 'answer'}` | `{o.reason or 'answered'}` |"
            )
    else:
        lines += ["", "**All decisions correct.**"]

    if mis:
        lines += ["", "### Refused for the wrong reason", "",
                  "These were correctly declined, but by a different guard than intended --",
                  "worth knowing, because a guard that never fires is not actually working.",
                  "", "| query | expected | got |", "|---|---|---|"]
        for o in mis:
            q = (o.case.text[:60] or "&lt;empty&gt;").replace("|", "\\|")
            lines.append(f"| {q} | `{o.case.expect}` | `{o.reason}` |")

    lines += [
        "",
        "## Method note",
        "",
        "The input guardrails are lexicon and pattern based, not a neural classifier.",
        "That is a deliberate trade -- a transformer safety model would be more robust",
        "to paraphrase but would add a model download, ~20 ms of latency on the",
        "critical path, and another dependency that can fail to load. Because the",
        "action taken is *abstain*, the cost of a false positive is a refused answer",
        "rather than a harmful one, so the honest thing is to measure how often that",
        "happens and publish it. That is the over-refusal rate above.",
    ]

    out = cfg.paths.data_dir.parent / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw = Path(cfg.paths.trace_dir) / "guardrail_eval.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        json.dumps(
            {
                "precision": counts.precision, "recall": counts.recall, "f1": counts.f1,
                "over_refusal_rate": counts.false_positive_rate,
                "tp": counts.tp, "fp": counts.fp, "tn": counts.tn, "fn": counts.fn,
                "wrong_reason": counts.wrong_reason,
                "cases": [
                    {"text": o.case.text, "tag": o.case.tag, "side": o.case.side,
                     "expected": o.case.expect, "got": o.reason,
                     "correct": o.correct_decision}
                    for o in outcomes
                ],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    console.print(f"[dim]{out}\n{raw}[/]")


if __name__ == "__main__":
    main()
