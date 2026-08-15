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
import time
from pathlib import Path

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

    # Upload FILE BY FILE, with retries, skipping what is already there.
    #
    # `upload_folder` builds one commit and is all-or-nothing: a dropped
    # connection 300 MB into a 370 MB transfer loses the whole thing. On a home
    # connection that is not a hypothetical -- it happened twice here. Per-file
    # commits are slightly noisier in the repo history and vastly more likely to
    # finish.
    already = {}
    try:
        info = api.repo_info(args.repo_id, repo_type=repo_type, files_metadata=True)
        already = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
    except Exception:  # noqa: BLE001 -- fresh repo
        pass

    skip_names = {".embed.progress", "vectors.f32"}
    pending: list[tuple[Path, str]] = []
    for f in sorted(files, key=lambda p: p.stat().st_size):
        rel = f.relative_to(index_dir).as_posix()
        if f.name.startswith(".") or f.name in skip_names:
            continue
        if rel in ("chunks/embed.bin", "chunks/embed_offsets.npy"):
            continue
        if already.get(rel) == f.stat().st_size:
            console.print(f"  [dim]skip (already uploaded)[/] {rel}")
            continue
        pending.append((f, rel))

    console.print(f"uploading {len(pending)} file(s), "
                  f"{sum(f.stat().st_size for f, _ in pending) / 1e6:.0f} MB")

    for f, rel in pending:
        size_mb = f.stat().st_size / 1e6
        for attempt in range(1, 5):
            try:
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=rel,
                    repo_id=args.repo_id,
                    repo_type=repo_type,
                    commit_message=f"{args.message} ({rel})",
                )
                console.print(f"  [green]ok[/] {rel} [dim]{size_mb:.1f} MB[/]")
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 4:
                    console.print(f"  [red]FAILED[/] {rel}: {type(exc).__name__}")
                    raise
                wait = 5 * attempt
                console.print(f"  [yellow]retry {attempt}/3[/] {rel} "
                              f"({type(exc).__name__}) in {wait}s")
                time.sleep(wait)


    console.print(f"\n[bold green]done[/] https://huggingface.co/datasets/{args.repo_id}")
    console.print("\nNow set this in configs/default.yaml (or as a Space secret):")
    console.print(f"  [cyan]remote_index.repo_id: {args.repo_id}[/]")


if __name__ == "__main__":
    main()
