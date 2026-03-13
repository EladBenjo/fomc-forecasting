"""
Tests for fedtext.text.features.novelty.

Pure-function tests — no model, no DB, no network.
"""

import math
import pytest
from fedtext.text.features.novelty import compute_novelty, compute_novelty_by_type


class TestComputeNovelty:
    def test_empty_input(self):
        assert compute_novelty({}) == {}

    def test_single_document_gets_1(self):
        result = compute_novelty({"2020-01-01": "some text about inflation"})
        assert result == {"2020-01-01": 1.0}

    def test_identical_documents_score_zero(self):
        text = "inflation interest rates monetary policy employment"
        result = compute_novelty({
            "2020-01-01": text,
            "2020-02-01": text,
        })
        assert result["2020-01-01"] == 1.0
        assert result["2020-02-01"] == pytest.approx(0.0, abs=1e-6)

    def test_completely_different_documents_score_near_one(self):
        result = compute_novelty({
            "2020-01-01": "inflation interest rates monetary policy employment",
            "2020-02-01": "quantum entanglement photon wavelength laser optics",
        })
        assert result["2020-01-01"] == 1.0
        assert result["2020-02-01"] > 0.9

    def test_chronological_order_by_date_key(self):
        # Later date should be compared to earlier date, not the other way around
        result = compute_novelty({
            "2020-03-01": "completely different words alpha beta gamma",
            "2020-01-01": "inflation interest rates monetary policy",
        })
        # First chronologically (Jan) should get 1.0
        assert result["2020-01-01"] == 1.0
        # March compared to January — novel vocabulary
        assert result["2020-03-01"] > 0.5

    def test_scores_in_valid_range(self):
        result = compute_novelty({
            "2020-01-01": "inflation expectations monetary policy",
            "2020-02-01": "inflation slightly higher interest rates unchanged",
            "2020-03-01": "labor market strengthened wages accelerating",
        })
        for score in result.values():
            assert 0.0 <= score <= 1.0


class TestComputeNoveltyByType:
    def test_groups_by_source_type(self):
        records = [
            {"doc_id": "sp1", "source_type": "speech",   "date": "2020-01-01", "text": "inflation rates"},
            {"doc_id": "sp2", "source_type": "speech",   "date": "2020-02-01", "text": "inflation rates"},
            {"doc_id": "d1",  "source_type": "document", "date": "2020-01-15", "text": "inflation rates"},
        ]
        result = compute_novelty_by_type(records)
        # Both source types have first doc as 1.0
        assert result["sp1"] == 1.0
        assert result["d1"]  == 1.0
        # sp2 compared only to sp1, not to d1
        assert "sp2" in result

    def test_returns_doc_id_keys(self):
        records = [
            {"doc_id": "abc", "source_type": "speech", "date": "2020-01-01", "text": "some text"},
        ]
        result = compute_novelty_by_type(records)
        assert "abc" in result

    def test_empty_records(self):
        assert compute_novelty_by_type([]) == {}
