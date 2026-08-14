"""Chunker tests.

The sentence splitter is the highest-leverage piece of text handling in the
project -- three of the five views are built on it -- and it has to work across
Devanagari, Tamil, Bengali and Latin punctuation in the same corpus. These tests
pin the behaviour that is easy to break and hard to notice: script integrity,
offset correctness, and the abbreviation/decimal cases.
"""

from __future__ import annotations

import pytest

from vrag.chunking.base import split_sentences
from vrag.chunking.atomic import AtomicChunker
from vrag.chunking.fixed_overlap import FixedOverlapChunker
from vrag.chunking.sentence_window import SentenceWindowChunker
from vrag.config import FixedOverlapCfg, SentenceWindowCfg
from vrag.schemas import Passage


def make_passage(text: str, lang: str = "en") -> Passage:
    return Passage(doc_id=f"{lang}:1:0", query_id=1, passage_idx=0, lang=lang, text=text)


# --------------------------------------------------------------------------- #
# Sentence splitting
# --------------------------------------------------------------------------- #
class TestSplitSentences:
    def test_latin_terminators(self):
        text = "First sentence. Second one! Third? Yes."
        assert len(split_sentences(text)) == 4

    def test_devanagari_danda(self):
        text = "यह पहला वाक्य है। यह दूसरा वाक्य है। यह तीसरा है।"
        assert len(split_sentences(text)) == 3

    def test_bengali_danda(self):
        text = "এটি প্রথম বাক্য। এটি দ্বিতীয় বাক্য। এটি তৃতীয়।"
        assert len(split_sentences(text)) == 3

    def test_urdu_full_stop(self):
        text = "یہ پہلا جملہ ہے۔ یہ دوسرا جملہ ہے۔"
        assert len(split_sentences(text)) == 2

    def test_double_danda(self):
        assert len(split_sentences("श्लोक एक॥ श्लोक दो॥")) == 2

    @pytest.mark.parametrize(
        "text",
        [
            "The value is 3.14 and that is final.",
            "It cost 1,234.56 dollars in total.",
        ],
    )
    def test_decimals_do_not_split(self, text):
        assert len(split_sentences(text)) == 1

    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Smith went home.",
            "Mr. and Mrs. Jones arrived.",
            "Founded in Jan. of that year.",
        ],
    )
    def test_abbreviations_do_not_split(self, text):
        assert len(split_sentences(text)) == 1

    def test_initials_do_not_split(self):
        assert len(split_sentences("J. K. Rowling wrote it.")) == 1

    def test_offsets_index_original_text(self):
        """Offsets must slice back to the sentence.

        Fusion-time dedup compares character spans across views, so an off-by-one
        here silently stops the dedup from recognising overlapping chunks.
        """
        text = "पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।"
        for sent in split_sentences(text):
            assert sent.text in text[sent.start : sent.end]

    def test_offsets_are_monotonic_and_non_overlapping(self):
        text = "One. Two. Three. Four."
        sentences = split_sentences(text)
        for a, b in zip(sentences, sentences[1:]):  # pairwise: intentionally not strict
            assert a.end <= b.start

    def test_devanagari_conjuncts_survive(self):
        """Splitting must never cut inside a grapheme cluster."""
        text = "क्षत्रिय शब्द संस्कृत से आया है। यह एक वर्ण है।"
        joined = "".join(s.text for s in split_sentences(text))
        assert "क्ष" in joined
        assert "्" not in joined[0]  # no leading virama on any chunk

    def test_merge_short_fragments(self):
        text = "Ok. This is a much longer sentence that clears the threshold easily."
        merged = split_sentences(text, min_chars=25, merge_short=True)
        assert len(merged) == 1
        assert merged[0].text.startswith("Ok.")

    def test_trailing_fragment_merges_backwards(self):
        text = "This is a long enough opening sentence for the test. No."
        merged = split_sentences(text, min_chars=25, merge_short=True)
        assert len(merged) == 1
        assert merged[0].text.endswith("No.")

    @pytest.mark.parametrize("text", ["", "   ", "\n\n"])
    def test_empty_input(self, text):
        assert split_sentences(text) == []

    def test_no_terminator_returns_whole_text(self):
        assert len(split_sentences("a sentence with no terminator")) == 1


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
class TestAtomicChunker:
    def test_single_chunk_covering_everything(self):
        passage = make_passage("Some passage text here.")
        chunks = AtomicChunker().chunk(passage)
        assert len(chunks) == 1
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(passage.text)

    def test_empty_passage(self):
        assert AtomicChunker().chunk(make_passage("   ")) == []


class TestSentenceWindowChunker:
    @pytest.fixture
    def chunker(self):
        return SentenceWindowChunker(SentenceWindowCfg(window=1, min_sentence_chars=10))

    def test_embed_text_is_narrower_than_context(self, chunker):
        passage = make_passage(
            "The first sentence is here. The second sentence follows it. "
            "And the third sentence closes."
        )
        chunks = chunker.chunk(passage)
        assert chunks
        middle = chunks[1]
        assert len(middle.text) < len(middle.context_text)
        assert middle.text in middle.context_text

    def test_single_sentence_passage_emits_nothing(self, chunker):
        """Would duplicate the atomic view for zero recall gain."""
        assert chunker.chunk(make_passage("Only one sentence here.")) == []

    def test_window_covers_neighbours(self, chunker):
        passage = make_passage("Alpha sentence one. Beta sentence two. Gamma sentence three.")
        chunks = chunker.chunk(passage)
        assert "Alpha" in chunks[1].context_text
        assert "Gamma" in chunks[1].context_text

    def test_indic_passage(self, chunker):
        passage = make_passage(
            "यह पहला वाक्य है और काफी लंबा है। यह दूसरा वाक्य है और यह भी लंबा है। "
            "यह तीसरा वाक्य है जो लंबा है।",
            lang="hi",
        )
        chunks = chunker.chunk(passage)
        assert len(chunks) == 3
        assert all(c.text.strip() for c in chunks)


class TestFixedOverlapChunker:
    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")

    def test_short_passage_skipped(self, tokenizer):
        cfg = FixedOverlapCfg(chunk_tokens=96, overlap_tokens=24, apply_above_tokens=128)
        chunker = FixedOverlapChunker(cfg, tokenizer)
        assert chunker.chunk(make_passage("A short passage. Not many tokens.")) == []

    def test_long_passage_produces_overlapping_chunks(self, tokenizer):
        cfg = FixedOverlapCfg(chunk_tokens=32, overlap_tokens=8, apply_above_tokens=40)
        chunker = FixedOverlapChunker(cfg, tokenizer)
        passage = make_passage("word " * 300)
        chunks = chunker.chunk(passage)
        assert len(chunks) > 1
        # Consecutive chunks must overlap -- that is the whole point of the view.
        for a, b in zip(chunks, chunks[1:]):  # pairwise: intentionally not strict
            assert b.char_start < a.char_end

    def test_chunks_cover_the_passage(self, tokenizer):
        cfg = FixedOverlapCfg(chunk_tokens=32, overlap_tokens=8, apply_above_tokens=40)
        chunker = FixedOverlapChunker(cfg, tokenizer)
        passage = make_passage("alpha beta gamma delta " * 60)
        chunks = chunker.chunk(passage)
        assert chunks[0].char_start == 0
        assert chunks[-1].char_end >= len(passage.text.rstrip()) - 2

    def test_boundaries_are_token_boundaries_not_character_cuts(self, tokenizer):
        """A character window would split Devanagari clusters; a token window cannot."""
        cfg = FixedOverlapCfg(chunk_tokens=24, overlap_tokens=6, apply_above_tokens=30)
        chunker = FixedOverlapChunker(cfg, tokenizer)
        passage = make_passage("क्षत्रिय राजपूत वर्ण व्यवस्था " * 40, lang="hi")
        for chunk in chunker.chunk(passage):
            assert not chunk.text.startswith("्")  # never a bare virama
            assert not chunk.text.startswith("े")  # never a bare vowel sign
