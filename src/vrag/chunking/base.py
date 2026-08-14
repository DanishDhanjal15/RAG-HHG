"""Chunker protocol, the shared chunk record, and an Indic-aware sentence splitter.

Three of the five views are built on sentence boundaries, so the splitter is the
single highest-leverage piece of text handling in the project. It has to cope with
Devanagari and Bengali danda (``।`` / ``॥``), Urdu full stop (``۔``) and question
mark (``؟``), Latin punctuation inside the same corpus, and the abbreviation /
decimal cases that make a naive ``text.split('.')`` produce garbage chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from vrag.schemas import ChunkView, Passage

# --------------------------------------------------------------------------- #
# Sentence splitting
# --------------------------------------------------------------------------- #

# Terminators across every script in the corpus.
_TERMINATORS = "।॥۔؟।!?."

# Common abbreviations that must not end a sentence. Kept deliberately small --
# over-listing costs recall on real sentence ends.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd",
    "co", "corp", "no", "fig", "eg", "ie", "approx", "dept", "univ", "govt",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

# A candidate break: one or more terminators, then whitespace.
_BREAK_RE = re.compile(rf"([{re.escape(_TERMINATORS)}]+)(\s+)")

_WS = re.compile(r"\s+")


@dataclass(slots=True)
class Sentence:
    text: str
    start: int
    end: int


def _is_false_break(text: str, term_start: int, term_char: str) -> bool:
    """True when a terminator does not actually end a sentence.

    Only ASCII ``.`` is ambiguous -- danda and Urdu full stop are unambiguous
    terminators, so we never spend cycles second-guessing them.
    """
    if term_char != ".":
        return False

    # Decimal number: digit before AND after the dot.
    before = text[:term_start]
    after = text[term_start + 1 :]
    if before[-1:].isdigit() and after[:1].isdigit():
        return True

    # Single-letter initial ("J. K. Rowling").
    tail = before.rstrip()
    if len(tail) >= 1 and tail[-1:].isalpha() and (len(tail) == 1 or not tail[-2:-1].isalpha()):
        return True

    # Known abbreviation.
    word = re.split(r"[\s(\[]", tail)[-1].lower().strip(".,;:")
    return word in _ABBREVIATIONS


def split_sentences(text: str, min_chars: int = 0, merge_short: bool = True) -> list[Sentence]:
    """Split into sentences, preserving character offsets into ``text``.

    Offsets matter: they are what the fusion-time dedup uses to recognise that a
    ``sentence_window`` chunk and a ``semantic`` chunk cover the same span.
    """
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    cursor = 0

    for match in _BREAK_RE.finditer(text):
        terminators = match.group(1)
        term_start = match.start(1)
        if _is_false_break(text, term_start, terminators[-1]):
            continue
        end = match.end(1)
        if end > cursor:
            spans.append((cursor, end))
        cursor = match.end(2)

    if cursor < len(text):
        spans.append((cursor, len(text)))

    sentences = [
        Sentence(text=text[s:e].strip(), start=s, end=e)
        for s, e in spans
        if text[s:e].strip()
    ]

    if merge_short and min_chars > 0:
        sentences = _merge_short(sentences, min_chars)

    return sentences


def _merge_short(sentences: list[Sentence], min_chars: int) -> list[Sentence]:
    """Fold fragments shorter than ``min_chars`` into the following sentence.

    A four-word fragment is a bad retrieval unit -- it embeds to noise and pollutes
    the candidate pool. Merging forward (not backward) keeps the fragment attached
    to the clause it introduces, which is the usual relationship.
    """
    if not sentences:
        return []

    out: list[Sentence] = []
    pending: Sentence | None = None

    for sent in sentences:
        if pending is not None:
            sent = Sentence(
                text=f"{pending.text} {sent.text}".strip(),
                start=pending.start,
                end=sent.end,
            )
            pending = None
        if len(sent.text) < min_chars:
            pending = sent
            continue
        out.append(sent)

    if pending is not None:
        if out:
            last = out[-1]
            out[-1] = Sentence(
                text=f"{last.text} {pending.text}".strip(),
                start=last.start,
                end=pending.end,
            )
        else:
            out.append(pending)

    return out


# --------------------------------------------------------------------------- #
# Chunker protocol
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class RawChunk:
    """A chunk before it gets a global id.

    ``text`` is what gets embedded; ``context_text`` is what gets returned to the
    user and passed to the generator. Keeping them separate is the whole point of
    the sentence-window view -- embed something small and precise, return something
    large enough to answer from.
    """

    text: str
    context_text: str
    char_start: int
    char_end: int
    local_idx: int = 0
    extra: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class Chunker(Protocol):
    view: ChunkView

    def chunk(self, passage: Passage) -> list[RawChunk]: ...


def contextualize(passage: Passage, raw: RawChunk, max_prefix_chars: int = 120) -> str:
    """Prepend a short document-level anchor to chunks that don't start the passage.

    A mid-passage chunk read in isolation often loses its subject ("It was founded
    in 1911"). Prefixing the passage's opening clause restores just enough context
    to embed correctly, at a cost of a few tokens.
    """
    if raw.char_start == 0:
        return raw.text
    head = passage.text[:max_prefix_chars].strip()
    if not head or head in raw.text:
        return raw.text
    head = _WS.sub(" ", head)
    return f"{head} … {raw.text}"
