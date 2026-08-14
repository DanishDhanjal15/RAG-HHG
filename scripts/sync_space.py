"""Sync the source tree into a cloned Hugging Face Space repo.

The Space and the GitHub repo hold the same code but are not the same repo, and
two things differ:

* the Space needs a ``README.md`` carrying HF's YAML frontmatter (``sdk: docker``,
  ``app_port: 7860``), which would be noise at the top of the GitHub README;
* the Space must not carry ``data/``, ``models/`` or ``traces/`` -- the index and
  the ONNX encoders are fetched or built rather than committed, and pushing ~400 MB
  of vectors into a Space repo would make every subsequent code push drag them.

Doing this by hand invites forgetting one of those, so it is a script.

    python scripts/sync_space.py --space D:/space
    cd D:/space && git add -A && git commit -m "Deploy" && git push
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rich.console import Console

console = Console()

# Copied wholesale.
DIRS = ["src", "configs", "docs", "tests"]

# Individual files.
FILES = [
    "Dockerfile",
    ".dockerignore",
    ".gitattributes",
    "pyproject.toml",
    "scripts/export_models.py",
    "scripts/publish_index.py",
    "bench/run_latency.py",
    "bench/run_retrieval_eval.py",
    "bench/run_guardrail_eval.py",
    "bench/calibrate_thresholds.py",
    "bench/adversarial_queries.yaml",
]

# Never sync: build outputs, secrets, caches. `data/` and `models/` are the
# reason this list exists -- see the module docstring.
EXCLUDE = {
    "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", ".git",
    "data", "models", "traces", ".env",
}


def ignored(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in EXCLUDE or n.endswith((".pyc", ".sqlite"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True, help="path to the cloned Space repo")
    parser.add_argument("--repo", default=".", help="path to the source repo")
    args = parser.parse_args()

    src = Path(args.repo).resolve()
    dst = Path(args.space).resolve()

    if not (dst / ".git").exists():
        raise SystemExit(f"{dst} is not a git repo -- clone the Space there first")

    copied = 0

    for name in DIRS:
        source = src / name
        if not source.exists():
            continue
        target = dst / name
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target, ignore=ignored)
        copied += sum(1 for _ in target.rglob("*") if _.is_file())
        console.print(f"  [green]{name}/[/]")

    for name in FILES:
        source = src / name
        if not source.exists():
            console.print(f"  [yellow]missing, skipped:[/] {name}")
            continue
        target = dst / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
        console.print(f"  [green]{name}[/]")

    # The Space README is the one with HF frontmatter, and it must land as
    # README.md -- HF reads the SDK and port from it, and gets them wrong (or
    # defaults to a static Space) if it is missing.
    space_readme = src / "deploy" / "README.space.md"
    if space_readme.exists():
        shutil.copy2(space_readme, dst / "README.md")
        console.print("  [green]README.md[/] [dim](from deploy/README.space.md)[/]")

    # Remove the default static-Space scaffolding, which would otherwise sit
    # alongside the app and confuse anyone reading the repo.
    for leftover in ("index.html", "style.css"):
        path = dst / leftover
        if path.exists():
            path.unlink()
            console.print(f"  [dim]removed template {leftover}[/]")

    console.print(f"\n[bold]{copied} files synced to {dst}[/]")
    console.print("[dim]next: cd into it, git add -A, commit, push[/]")


if __name__ == "__main__":
    main()
