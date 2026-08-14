"""Publish a built index to a Hugging Face dataset repo.

Run once after `vrag build`. The deployed Space then downloads it on first boot
(see `vrag.index.fetch`) instead of building it, which on CPU would take hours per
deploy.

A dataset repo rather than the Space repo itself, deliberately: the index is data
with its own lifecycle, it is versioned independently of the code, and keeping it
out of the Space repo means a code push does not re-upload 170 MB.

    huggingface-cli login          # or set VRAG_HF_TOKEN
    python scripts/publish_index.py --repo-id <user>/vrag-index
"""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from vrag.config import get_config, get_secrets
from vrag.index.fetch import index_is_complete, missing_pieces

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. your-name/vrag-index")
    parser.add_argument("--repo-type", default=None, choices=["dataset", "model"],
                        help="defaults to remote_index.repo_type in config")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default="Publish prebuilt index")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    cfg = get_config()
    index_dir = cfg.paths.index_dir

    if not index_is_complete(index_dir):
        console.print(f"[red]Index at {index_dir} is incomplete.[/]")
        console.print(f"missing: {missing_pieces(index_dir)}")
        console.print("Run `vrag build` first.")
        raise SystemExit(1)

    files = [f for f in index_dir.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)

    table = Table(title="Index to publish")
    table.add_column("file")
    table.add_column("size", justify="right")
    for f in sorted(files, key=lambda p: -p.stat().st_size)[:10]:
        table.add_row(str(f.relative_to(index_dir)), f"{f.stat().st_size / 1e6:.1f} MB")
    table.add_row("[dim]...[/]", "")
    table.add_row("[bold]total[/]", f"[bold]{total / 1e6:.0f} MB[/] ({len(files)} files)")
    console.print(table)

    token = get_secrets().hf_token or None
    api = HfApi(token=token)
    repo_type = args.repo_type or cfg.remote_index.repo_type

    console.print(f"creating/updating {repo_type} repo [bold]{args.repo_id}[/]")
    api.create_repo(repo_id=args.repo_id, repo_type=repo_type,
                    private=args.private, exist_ok=True)

    # Ship the manifest so the deployed index is traceable to the build that
    # produced it -- corpus size, view counts, embedding model, index type.
    manifest = index_dir / "manifest.json"
    if manifest.exists():
        console.print("[dim]manifest:[/] " + json.dumps(
            json.loads(manifest.read_text(encoding="utf-8")).get("chunk", {}), indent=None
        )[:160])

    console.print("uploading (this takes a few minutes)…")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=repo_type,
        folder_path=str(index_dir),
        commit_message=args.message,
        # Never publish build markers -- they would make a downloaded index look
        # already-built to a machine that has not built anything.
        ignore_patterns=[".*.done", "vectors.f32"],
    )

    console.print(f"\n[bold green]done[/] https://huggingface.co/datasets/{args.repo_id}")
    console.print("\nNow set this in configs/default.yaml (or as a Space secret):")
    console.print(f"  [cyan]remote_index.repo_id: {args.repo_id}[/]")


if __name__ == "__main__":
    main()
