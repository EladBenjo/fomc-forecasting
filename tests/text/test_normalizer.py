"""
Tests for fedtext.text.cleaning.normalizer.

These are pure-function tests — no model, no DB, no network.
"""

import pytest
from fedtext.text.cleaning.normalizer import normalize, split_sentences


class TestNormalize:
    def test_strips_leading_trailing_whitespace(self):
        assert normalize("  hello  ") == "hello"

    def test_collapses_multiple_blank_lines(self):
        result = normalize("para one\n\n\n\npara two")
        assert result == "para one\n\npara two"

    def test_collapses_inline_spaces(self):
        result = normalize("too  many   spaces")
        assert result == "too many spaces"

    def test_normalizes_crlf(self):
        result = normalize("line one\r\nline two")
        assert "\r" not in result
        assert "line one\nline two" == result

    def test_empty_string(self):
        assert normalize("") == ""

    def test_preserves_original_casing(self):
        # FOMC-RoBERTa is case-sensitive — must NOT lowercase
        text = "The Federal Reserve maintained the target range."
        assert normalize(text) == text


class TestSplitSentences:
    def test_splits_on_period(self):
        sents = split_sentences("First sentence. Second sentence.")
        assert len(sents) == 2
        assert sents[0] == "First sentence."

    def test_splits_on_contrastive_but(self):
        sents = split_sentences("Inflation is contained but risks remain elevated.")
        assert len(sents) == 2
        assert sents[0].lower().startswith("inflation")
        assert "risks" in sents[1].lower()

    def test_splits_on_however(self):
        # "however" mid-sentence should produce two parts
        # Note: when "however" starts a sentence it gets consumed by the split —
        # this tests the meaningful case where it separates two clauses.
        sents = split_sentences("Growth is solid however uncertainty about inflation persists.")
        assert len(sents) == 2
        assert "solid" in sents[0].lower()
        assert "uncertainty" in sents[1].lower()

    def test_no_empty_strings(self):
        sents = split_sentences("One. Two. Three.")
        assert all(s.strip() for s in sents)

    def test_empty_input(self):
        assert split_sentences("") == []

    def test_single_sentence_no_contrastive(self):
        text = "The committee voted unanimously."
        sents = split_sentences(text)
        assert len(sents) == 1
        assert sents[0] == text
