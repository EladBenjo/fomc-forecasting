"""Tests for fedtext.text.features.document_features."""

from __future__ import annotations

from fedtext.text.features.document_features import (
    compute_target_sentences_ratio,
    count_words,
    infer_role,
)


def test_count_words_handles_punctuation_and_spacing():
    text = "Inflation remains elevated.\n\nPolicy is restrictive, for now."
    assert count_words(text) == 8


def test_compute_target_sentences_ratio_basic():
    ratio = compute_target_sentences_ratio(n_target_sentences=3, total_sentences_count=12)
    assert ratio == 0.25


def test_compute_target_sentences_ratio_zero_denominator():
    ratio = compute_target_sentences_ratio(n_target_sentences=2, total_sentences_count=0)
    assert ratio == 0.0


def test_infer_role_returns_none_for_documents():
    role = infer_role(source_type="document", speaker="Alan Greenspan")
    assert role is None


def test_infer_role_from_speaker_name_map():
    role = infer_role(source_type="speech", speaker="Alan Greenspan")
    assert role == "Chairman"


def test_infer_role_from_metadata_keywords():
    role = infer_role(
        source_type="speech",
        speaker="Jane Doe",
        title="Vice Chair for Supervision",
        event="Fireside chat",
    )
    assert role == "Vice Chairman"


def test_infer_role_unresolved_returns_none():
    role = infer_role(source_type="speech", speaker="Unknown Speaker")
    assert role is None
