"""STT provider tests, using a mock transport rather than real API calls.

The valuable paths here are the failure paths -- retry, backoff, breaker,
malformed responses -- and those are precisely the ones you cannot exercise
against a healthy live API. `httpx.MockTransport` lets us drive them
deterministically, so "what happens when Sarvam returns a 503 three times" is a
test rather than a hope.
"""

from __future__ import annotations

import httpx
import pytest

from vrag.config import load_config
from vrag.harness.resilience import PermanentError, SttUnavailable, TransientError
from vrag.schemas import AudioInput
from vrag.stt.base import ReplayProvider
from vrag.stt.sarvam import SarvamStt, build_provider

WAV = b"RIFF" + b"\x00" * 40


def audio() -> AudioInput:
    return AudioInput(audio=WAV, filename="q.wav", mime_type="audio/wav")


class FakeSecrets:
    sarvam_api_key = "test-key"
    elevenlabs_api_key = "test-key"
    anthropic_api_key = ""
    hf_token = ""


def make_provider(handler, **overrides) -> SarvamStt:
    cfg = load_config()
    for key, value in overrides.items():
        setattr(cfg.stt.sarvam, key, value)
    provider = SarvamStt(cfg, FakeSecrets())
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


OK_BODY = {
    "request_id": "req_1",
    "transcript": "कॉर्पोरेशन क्या है",
    "language_code": "hi-IN",
    "language_probability": 0.97,
}


# --------------------------------------------------------------------------- #
class TestParsing:
    @pytest.mark.asyncio
    async def test_successful_transcription(self):
        provider = make_provider(lambda r: httpx.Response(200, json=OK_BODY))
        result = await provider.transcribe(audio())
        assert result.text == "कॉर्पोरेशन क्या है"
        assert result.confidence == pytest.approx(0.97)
        assert result.provider == "sarvam"

    @pytest.mark.asyncio
    async def test_bcp47_is_mapped_to_iso_639_1(self):
        """Chunks are tagged `hi`, not `hi-IN`. If this mapping breaks, the
        language filter silently matches nothing and recall collapses."""
        provider = make_provider(lambda r: httpx.Response(200, json=OK_BODY))
        assert (await provider.transcribe(audio())).lang == "hi"

    @pytest.mark.asyncio
    async def test_unknown_language_code_degrades_gracefully(self):
        provider = make_provider(
            lambda r: httpx.Response(200, json={**OK_BODY, "language_code": "xx-YY"})
        )
        assert (await provider.transcribe(audio())).lang == "xx"

    @pytest.mark.asyncio
    async def test_missing_confidence_defaults_by_whether_there_is_text(self):
        provider = make_provider(
            lambda r: httpx.Response(200, json={"transcript": "hello"})
        )
        assert (await provider.transcribe(audio())).confidence == 1.0

        empty = make_provider(lambda r: httpx.Response(200, json={"transcript": ""}))
        assert (await empty.transcribe(audio())).confidence == 0.0

    @pytest.mark.asyncio
    async def test_lang_hint_is_sent_as_bcp47(self):
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = request.content.decode("utf-8", "replace")
            return httpx.Response(200, json=OK_BODY)

        provider = make_provider(handler)
        await provider.transcribe(
            AudioInput(audio=WAV, filename="q.wav", lang_hint="ta")
        )
        assert "ta-IN" in seen["body"]


# --------------------------------------------------------------------------- #
class TestResilience:
    @pytest.mark.asyncio
    async def test_missing_key_fails_immediately(self):
        cfg = load_config()

        class NoKey(FakeSecrets):
            sarvam_api_key = ""

        provider = SarvamStt(cfg, NoKey())
        with pytest.raises(SttUnavailable):
            await provider.transcribe(audio())

    @pytest.mark.asyncio
    async def test_transient_error_is_retried_then_succeeds(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(503, text="upstream unavailable")
            return httpx.Response(200, json=OK_BODY)

        provider = make_provider(handler, max_retries=4, backoff_base_s=0.01)
        result = await provider.transcribe(audio())
        assert result.text
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_auth_error_is_not_retried(self):
        """A 401 will be a 401 next time too. Retrying wastes the user's latency
        budget to arrive at the same answer."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(401, text="invalid key")

        provider = make_provider(handler, max_retries=4, backoff_base_s=0.01)
        with pytest.raises(PermanentError):
            await provider.transcribe(audio())
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_rate_limit_is_retried(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(429, text="slow down")

        provider = make_provider(handler, max_retries=3, backoff_base_s=0.01)
        with pytest.raises(TransientError):
            await provider.transcribe(audio())
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_breaker_opens_and_then_fails_fast(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500, text="boom")

        provider = make_provider(
            handler, max_retries=1, backoff_base_s=0.01, circuit_breaker_failures=2
        )
        for _ in range(2):
            with pytest.raises(TransientError):
                await provider.transcribe(audio())

        before = calls["n"]
        # Breaker is open: the next call must not touch the network at all.
        with pytest.raises(SttUnavailable):
            await provider.transcribe(audio())
        assert calls["n"] == before

    @pytest.mark.asyncio
    async def test_breaker_message_explains_the_wait(self):
        provider = make_provider(
            lambda r: httpx.Response(500), max_retries=1,
            backoff_base_s=0.01, circuit_breaker_failures=1,
        )
        with pytest.raises(TransientError):
            await provider.transcribe(audio())
        with pytest.raises(SttUnavailable, match="circuit breaker open"):
            await provider.transcribe(audio())


# --------------------------------------------------------------------------- #
class TestProviderSelection:
    def test_sarvam_is_default(self):
        cfg = load_config()
        assert build_provider(cfg, FakeSecrets()).name == "sarvam"

    def test_elevenlabs_is_selectable(self):
        cfg = load_config()
        cfg.stt.provider = "elevenlabs"
        assert build_provider(cfg, FakeSecrets()).name == "elevenlabs"

    def test_unknown_provider_is_rejected_loudly(self):
        cfg = load_config()
        cfg.stt.provider = "nonexistent"  # type: ignore[assignment]
        with pytest.raises(PermanentError):
            build_provider(cfg, FakeSecrets())


class TestReplayProvider:
    @pytest.mark.asyncio
    async def test_cycles_through_canned_transcripts(self):
        from vrag.schemas import Transcript

        provider = ReplayProvider([
            Transcript(text="one", lang="en", confidence=1.0, provider="replay"),
            Transcript(text="two", lang="en", confidence=1.0, provider="replay"),
        ])
        assert (await provider.transcribe(audio())).text == "one"
        assert (await provider.transcribe(audio())).text == "two"
        assert (await provider.transcribe(audio())).text == "one"
