"""Sarvam AI speech-to-text (``saaras:v4``).

Chosen over ElevenLabs Scribe for this build because the corpus is Indic: Sarvam
covers every language in MSMARCO-XI natively, is tuned for Indian-accented speech
and code-mixing (people genuinely do ask "GDP kya hai iska"), and returns a
``language_probability`` that we reuse directly as the ASR confidence gate.

Everything here runs under the harness's resilience primitives -- retry with
jittered backoff, a hard timeout, and a circuit breaker -- so a dead key or a
provider outage produces a typed ``SttUnavailable`` and a clean "type your
question instead" rather than a stack trace or a hanging spinner.
"""

from __future__ import annotations

import httpx

from vrag.config import Config, Secrets
from vrag.harness.resilience import (
    CircuitBreaker,
    PermanentError,
    RetryPolicy,
    SttUnavailable,
    classify_http,
)
from vrag.schemas import AudioInput, Transcript

# Sarvam speaks BCP-47 with a region; the corpus uses ISO-639-1. Mapped in both
# directions so a language hint from the UI reaches the API and the detected
# language matches the `lang` field on indexed chunks.
_TO_SARVAM = {
    "hi": "hi-IN", "ta": "ta-IN", "bn": "bn-IN", "en": "en-IN",
    "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "gu": "gu-IN",
    "pa": "pa-IN", "te": "te-IN", "or": "od-IN", "as": "as-IN",
    "ur": "ur-IN", "ne": "ne-IN", "sa": "sa-IN",
}
_FROM_SARVAM = {v: k for k, v in _TO_SARVAM.items()}


class SarvamStt:
    name = "sarvam"

    def __init__(self, cfg: Config, secrets: Secrets) -> None:
        self.cfg = cfg.stt.sarvam
        self.api_key = secrets.sarvam_api_key
        self.breaker = CircuitBreaker(
            name="sarvam",
            failure_threshold=self.cfg.circuit_breaker_failures,
            reset_after_s=self.cfg.circuit_breaker_reset_s,
        )
        self.policy = RetryPolicy(
            max_attempts=self.cfg.max_retries, base_delay_s=self.cfg.backoff_base_s
        )
        self._client = httpx.AsyncClient(timeout=self.cfg.timeout_s)

    async def transcribe(self, audio: AudioInput) -> Transcript:
        if not self.api_key:
            raise SttUnavailable("sarvam: VRAG_SARVAM_API_KEY is not set")

        from vrag.harness.resilience import call_with_resilience

        language_code = (
            _TO_SARVAM.get(audio.lang_hint, self.cfg.language_code)
            if audio.lang_hint
            else self.cfg.language_code
        )

        async def call() -> Transcript:
            response = await self._client.post(
                self.cfg.endpoint,
                headers={"api-subscription-key": self.api_key},
                files={"file": (audio.filename, audio.audio, audio.mime_type)},
                data={
                    "model": self.cfg.model,
                    "mode": self.cfg.mode,
                    "language_code": language_code,
                },
            )
            if response.status_code >= 400:
                raise classify_http(response.status_code)(
                    f"sarvam {response.status_code}: {response.text[:200]}"
                )
            return self._parse(response.json())

        transcript, result = await call_with_resilience(
            call,
            policy=self.policy,
            breaker=self.breaker,
            timeout_s=self.cfg.timeout_s,
            stage="sarvam_stt",
        )
        transcript.request_id = transcript.request_id or ""
        _ = result
        return transcript

    def _parse(self, payload: dict) -> Transcript:
        text = (payload.get("transcript") or "").strip()
        raw_lang = payload.get("language_code") or "unknown"
        confidence = payload.get("language_probability")

        # `language_probability` is the provider's confidence in the *language*,
        # not the transcript. It is the best confidence signal the REST endpoint
        # exposes and it correlates well with transcription quality -- but the
        # distinction is real, so the guardrail treats it as a floor rather than
        # as a transcription score, and the threshold is set low (0.45) to match.
        if confidence is None:
            confidence = 1.0 if text else 0.0

        return Transcript(
            text=text,
            lang=_FROM_SARVAM.get(raw_lang, raw_lang.split("-")[0] if raw_lang else "unknown"),
            confidence=float(confidence),
            provider=self.name,
            request_id=payload.get("request_id", ""),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class ElevenLabsStt:
    """Drop-in alternative, kept working so a Sarvam outage or quota exhaustion
    is a one-line config change rather than a rewrite the night before a deadline."""

    name = "elevenlabs"

    def __init__(self, cfg: Config, secrets: Secrets) -> None:
        self.cfg = cfg.stt.elevenlabs
        self.api_key = secrets.elevenlabs_api_key
        self.breaker = CircuitBreaker(name="elevenlabs")
        self.policy = RetryPolicy(max_attempts=self.cfg.max_retries)
        self._client = httpx.AsyncClient(timeout=self.cfg.timeout_s)

    async def transcribe(self, audio: AudioInput) -> Transcript:
        if not self.api_key:
            raise SttUnavailable("elevenlabs: VRAG_ELEVENLABS_API_KEY is not set")

        from vrag.harness.resilience import call_with_resilience

        async def call() -> Transcript:
            data = {"model_id": self.cfg.model_id}
            if audio.lang_hint:
                data["language_code"] = audio.lang_hint
            response = await self._client.post(
                self.cfg.endpoint,
                headers={"xi-api-key": self.api_key},
                files={"file": (audio.filename, audio.audio, audio.mime_type)},
                data=data,
            )
            if response.status_code >= 400:
                raise classify_http(response.status_code)(
                    f"elevenlabs {response.status_code}: {response.text[:200]}"
                )
            payload = response.json()
            return Transcript(
                text=(payload.get("text") or "").strip(),
                lang=payload.get("language_code", "unknown"),
                confidence=float(payload.get("language_probability") or 0.0),
                provider=self.name,
                audio_duration_s=payload.get("audio_duration_secs"),
            )

        transcript, _ = await call_with_resilience(
            call,
            policy=self.policy,
            breaker=self.breaker,
            timeout_s=self.cfg.timeout_s,
            stage="elevenlabs_stt",
        )
        return transcript

    async def aclose(self) -> None:
        await self._client.aclose()


def build_provider(cfg: Config, secrets: Secrets):  # noqa: ANN201
    if cfg.stt.provider == "sarvam":
        return SarvamStt(cfg, secrets)
    if cfg.stt.provider == "elevenlabs":
        return ElevenLabsStt(cfg, secrets)
    raise PermanentError(f"unknown stt provider {cfg.stt.provider!r}")
