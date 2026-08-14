"""Speech-to-text provider protocol.

The pipeline depends on this protocol, never on a vendor SDK. That indirection
earns its keep in two places: the benchmark substitutes a replay provider so
5,000 latency samples do not cost 5,000 API calls, and switching providers is a
new file rather than a change to the pipeline.

Every implementation must return a ``confidence``. It feeds the ASR gate in the
input guardrail, which is the check that stops the system answering a question
the user never asked.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vrag.schemas import AudioInput, Transcript


@runtime_checkable
class SttProvider(Protocol):
    name: str

    async def transcribe(self, audio: AudioInput) -> Transcript: ...


class ReplayProvider:
    """Returns canned transcripts. Used by the benchmark and by tests.

    Real STT is the slowest and most expensive stage; measuring the RAG core
    against it thousands of times would be a bad use of both time and quota, and
    would add network variance to numbers meant to describe local compute.
    """

    name = "replay"

    def __init__(self, transcripts: list[Transcript]) -> None:
        self._transcripts = transcripts
        self._i = 0

    async def transcribe(self, audio: AudioInput) -> Transcript:  # noqa: ARG002
        transcript = self._transcripts[self._i % len(self._transcripts)]
        self._i += 1
        return transcript
