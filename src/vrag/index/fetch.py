"""Fetch a prebuilt index at boot.

The index is ~170 MB of vectors, graph and payload store. That does not belong in
a git repository (GitHub rejects files over 100 MB, and even under the limit a
binary artifact rebuilt on every corpus change makes the history useless), and it
cannot be built inside the Docker image either -- on CPU that is a multi-hour job
that would run on every deploy.

So the index is published once to a Hugging Face **dataset** repo and downloaded
on first boot if it is not already present. Subsequent boots find it on disk and
skip the download entirely, which matters because a Space restarting under memory
pressure should not re-download 170 MB each time.

This is opt-in: with no `remote_index.repo_id` configured, nothing here runs and
the app expects a locally built index -- which is exactly what local development
wants.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from vrag.config import Config, get_secrets

# The files that constitute a usable index. Presence of the dense index alone is
# not enough -- a half-downloaded index that boots and then fails on the first
# query is worse than one that refuses to boot.
REQUIRED = (
    "dense.faiss",
    "centroid.npy",
    "chunks/texts.bin",
    "chunks/offsets.npy",
    "chunks/embed.bin",
    "chunks/embed_offsets.npy",
    "chunks/meta.npz",
    "chunks/vocab.json",
)


def index_is_complete(index_dir: Path) -> bool:
    return all((index_dir / name).exists() for name in REQUIRED)


def missing_pieces(index_dir: Path) -> list[str]:
    return [name for name in REQUIRED if not (index_dir / name).exists()]


def ensure_index(cfg: Config, force: bool = False) -> bool:
    """Make sure a usable index exists locally. Returns True if one is present.

    Downloads to a temporary directory and moves into place only once complete,
    so an interrupted download can never leave a partial index that looks valid.
    """
    index_dir = cfg.paths.index_dir

    if not force and index_is_complete(index_dir):
        return True

    repo_id = cfg.remote_index.repo_id
    if not repo_id:
        return False

    from huggingface_hub import snapshot_download

    staging = index_dir.parent / f"{index_dir.name}.incoming"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=cfg.remote_index.revision,
        local_dir=str(staging),
        token=get_secrets().hf_token or None,
        # The sparse index is optional -- the pipeline degrades to dense-only and
        # says so -- but everything else must arrive.
        allow_patterns=["*.faiss", "*.npy", "*.json", "chunks/*", "sparse/*"],
    )

    if not index_is_complete(staging):
        missing = missing_pieces(staging)
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"index download from {repo_id} is incomplete; missing: {missing}"
        )

    # Atomic-ish swap: the app never observes a partially-populated index_dir.
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)
    staging.rename(index_dir)

    elapsed = time.perf_counter() - t0
    total = sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())
    print(f"[vrag] fetched index from {repo_id}: {total / 1e6:.0f} MB in {elapsed:.0f}s")
    return True
