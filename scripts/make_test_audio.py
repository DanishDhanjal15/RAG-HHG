"""Generate spoken test clips from real dataset queries, via Sarvam TTS.

Why this exists: measuring the voice path needs audio, and hand-recording dozens
of clips in four languages is not practical. Synthesising them from *real
MSMARCO-XI queries* gives clips that are linguistically representative of what
the system will actually be asked, in the scripts it will actually receive.

**Honest caveat, stated here and in the latency report:** TTS speech is cleaner
than human speech -- no background noise, no disfluencies, no accent variation,
no clipping. So these clips are a fair basis for measuring **latency** (the
network round trip does not care where the audio came from) but they will
*overstate* transcription **accuracy** versus real users in a noisy room. They
are a functional and timing test, not a WER benchmark.

Usage:
    python scripts/make_test_audio.py --n 24 --out bench/audio
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import time

import httpx
from rich.console import Console

from vrag.config import get_config, get_secrets
from vrag.ingest.normalize import load_queries

console = Console()

ENDPOINT = "https://api.sarvam.ai/text-to-speech"

# Sarvam BCP-47 codes, and a speaker per language so the clips are not all one voice.
LANG_CODES = {"hi": "hi-IN", "ta": "ta-IN", "bn": "bn-IN", "en": "en-IN"}
SPEAKERS = ["anushka", "abhilash", "karun", "hitesh"]


def synthesize(client: httpx.Client, key: str, text: str, lang: str,
               speaker: str, model: str) -> bytes | None:
    response = client.post(
        ENDPOINT,
        headers={"api-subscription-key": key, "content-type": "application/json"},
        json={
            "text": text,
            "language_code": LANG_CODES.get(lang, "en-IN"),
            "model": model,
            "speaker": speaker,
            # 16 kHz because that is what Sarvam STT works best at -- generating at
            # a higher rate only to resample it down would add loss for nothing.
            "speech_sample_rate": "16000",
        },
        timeout=60.0,
    )
    if response.status_code >= 400:
        console.print(f"[red]{response.status_code}[/] {response.text[:200]}")
        return None
    audios = response.json().get("audios") or []
    return base64.b64decode(audios[0]) if audios else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=24, help="clips to generate")
    parser.add_argument("--out", default="bench/audio")
    parser.add_argument("--model", default="bulbul:v2")
    parser.add_argument("--max-chars", type=int, default=120)
    args = parser.parse_args()

    cfg = get_config()
    key = get_secrets().sarvam_api_key
    if not key:
        console.print("[red]VRAG_SARVAM_API_KEY is not set[/]")
        return

    out_dir = cfg.paths.data_dir.parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Real queries, balanced across languages, short enough to speak naturally.
    #
    # Note on English: `QueryRecord.lang` is the shard language (hi/ta/bn only) --
    # there is no English shard. The English form of every query lives in
    # `eng_query`, so English clips are drawn from there rather than filtered for.
    records = load_queries(cfg)
    rng = random.Random(cfg.corpus.seed + 11)
    rng.shuffle(records)

    per_lang = max(1, args.n // len(LANG_CODES))
    chosen: list[tuple[str, str, int]] = []   # (lang, text, query_id)

    for lang in ("hi", "ta", "bn"):
        picked = [
            (lang, q.query, q.query_id)
            for q in records
            if q.lang == lang and q.query.strip() and len(q.query) <= args.max_chars
        ][:per_lang]
        chosen.extend(picked)

    chosen.extend(
        [
            ("en", q.eng_query.lstrip(". ").strip(), q.query_id)
            for q in records
            if q.eng_query.strip() and len(q.eng_query) <= args.max_chars
        ][:per_lang]
    )
    rng.shuffle(chosen)
    chosen = chosen[: args.n]

    console.print(f"generating {len(chosen)} clips -> {out_dir}")

    manifest = []
    written = 0
    with httpx.Client() as client:
        for i, (lang, text, query_id) in enumerate(chosen):
            speaker = SPEAKERS[i % len(SPEAKERS)]
            t0 = time.perf_counter()
            audio = synthesize(client, key, text, lang, speaker, args.model)
            if audio is None:
                continue
            name = f"{lang}_{query_id}.wav"
            (out_dir / name).write_bytes(audio)
            written += 1
            manifest.append({
                "file": name,
                "lang": lang,
                "query_id": query_id,
                "text": text,
                "speaker": speaker,
                "bytes": len(audio),
                "tts_ms": round((time.perf_counter() - t0) * 1000, 1),
            })
            console.print(f"  [green]{name}[/] {len(audio) / 1024:.0f} KB  "
                          f"[dim]{text[:60]}[/]")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"\n[bold]{written}/{len(chosen)} clips written[/]  "
                  f"manifest: {out_dir / 'manifest.json'}")
    console.print(
        "[dim]The manifest carries the source text, so transcription can be checked "
        "against ground truth -- but remember these are synthetic voices.[/]"
    )


if __name__ == "__main__":
    main()
