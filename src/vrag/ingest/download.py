"""Fetch MSMARCO-XI parquet shards.

Why not ``datasets.load_dataset``: the HF dataset viewer for ``ai4bharat/MSMARCO-XI``
currently reports "dataset generation failed" with parquet errors, and the repo has
no dataset-config structure -- it is a flat tree of per-language files
(``validation/hinval.parquet``, ``validation/tamval.parquet``, ...). Reading the
parquet directly is both more reliable and lets us pull only the row groups we need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from vrag.config import Config, get_secrets


@dataclass
class Shard:
    lang: str
    remote_path: str
    local_path: Path


def _token() -> str | None:
    return get_secrets().hf_token or None


def download_shards(cfg: Config, langs: list[str] | None = None) -> list[Shard]:
    """Download one parquet shard per configured language into ``paths.raw_dir``.

    ``hf_hub_download`` is content-addressed and resumable, so re-running is cheap
    and a half-finished download never yields a corrupt file.
    """
    cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    wanted = langs or list(cfg.corpus.languages)
    shards: list[Shard] = []

    for lang in wanted:
        remote = cfg.corpus.languages[lang]
        local = hf_hub_download(
            repo_id=cfg.corpus.repo_id,
            filename=remote,
            repo_type="dataset",
            local_dir=str(cfg.paths.raw_dir),
            token=_token(),
        )
        shards.append(Shard(lang=lang, remote_path=remote, local_path=Path(local)))

    return shards


def open_remote(cfg: Config, remote_path: str):
    """Open a shard over HTTP without downloading it.

    Used by ``vrag probe`` to inspect the real arrow schema (a few hundred KB of
    range requests) before committing to 1.4 GB of downloads.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=_token())
    return fs.open(f"datasets/{cfg.corpus.repo_id}/{remote_path}", "rb")
