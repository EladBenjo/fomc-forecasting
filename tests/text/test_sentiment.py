"""
Tests for fedtext.text.features.sentiment.

The FOMC-RoBERTa model is NOT loaded here — the HuggingFace pipeline is mocked.
This keeps tests fast (<1s) and dependency-free (no transformers download needed).

Mock contract: pipeline(sentences, ...) returns a list of {"label": ..., "score": ...}
dicts matching the real model's output format.
"""

import pytest
from unittest.mock import MagicMock
from fedtext.text.features.sentiment import score_document, SentimentResult


def _make_pipeline(*labels: str):
    """Return a mock pipeline that assigns labels in order to input sentences."""
    def _pipeline(sentences, **kwargs):
        return [{"label": lbl, "score": 0.99} for lbl in labels]
    return _pipeline


class TestScoreDocument:
    # ------------------------------------------------------------------
    # Score formula
    # ------------------------------------------------------------------

    def test_all_hawkish_score_plus_one(self):
        # 3 economic sentences, all hawkish → score = (3-0)/3 = 1.0
        sentences = [
            "Inflation expectations have risen significantly.",
            "Interest rate hikes are necessary to restore price stability.",
            "Monetary policy must remain restrictive.",
        ]
        pipe = _make_pipeline("LABEL_1", "LABEL_1", "LABEL_1")
        result = score_document("", sentences, pipeline=pipe)
        assert result.hawkish_score == pytest.approx(1.0)
        assert result.n_hawkish == 3
        assert result.n_dovish == 0
        assert result.n_neutral == 0

    def test_all_dovish_score_minus_one(self):
        sentences = [
            "Inflation expectations remain well-anchored.",
            "Interest rates can remain accommodative.",
            "Monetary policy supports employment growth.",
        ]
        pipe = _make_pipeline("LABEL_0", "LABEL_0", "LABEL_0")
        result = score_document("", sentences, pipeline=pipe)
        assert result.hawkish_score == pytest.approx(-1.0)
        assert result.n_dovish == 3

    def test_mixed_score(self):
        # 1 hawkish, 1 dovish, 1 neutral → score = (1-1)/3 = 0.0
        sentences = [
            "Inflation remains above target.",
            "Labor market conditions are accommodative.",
            "The committee is monitoring developments.",
        ]
        pipe = _make_pipeline("LABEL_1", "LABEL_0", "LABEL_2")
        result = score_document("", sentences, pipeline=pipe)
        assert result.hawkish_score == pytest.approx(0.0)
        assert result.n_hawkish == 1
        assert result.n_dovish == 1
        assert result.n_neutral == 1

    # ------------------------------------------------------------------
    # Economic keyword filter
    # ------------------------------------------------------------------

    def test_boilerplate_filtered_out(self):
        # None of these sentences mention economic keywords — pipeline never called
        sentences = [
            "The meeting was adjourned at 4:00 PM.",
            "The vote was unanimous.",
            "Attendance was noted for the record.",
        ]
        pipe = MagicMock()
        result = score_document("", sentences, pipeline=pipe)
        pipe.assert_not_called()
        assert result.n_target_sentences == 0
        assert result.hawkish_score == pytest.approx(0.0)

    def test_economic_sentences_pass_filter(self):
        sentences = [
            "Inflation has remained above the 2 percent target.",  # "inflation" → passes
            "The meeting was adjourned.",                          # boilerplate → filtered
        ]
        pipe = _make_pipeline("LABEL_1")  # only 1 call expected
        result = score_document("", sentences, pipeline=pipe)
        assert result.n_target_sentences == 1

    def test_interest_rate_keyword_passes(self):
        sentences = ["Interest rate decisions depend on incoming data."]
        pipe = _make_pipeline("LABEL_2")
        result = score_document("", sentences, pipeline=pipe)
        assert result.n_target_sentences == 1

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_no_sentences_returns_neutral(self):
        result = score_document("", [], pipeline=MagicMock())
        assert result.hawkish_score == pytest.approx(0.0)
        assert result.n_target_sentences == 0

    def test_result_type(self):
        sentences = ["Inflation expectations are well-anchored."]
        pipe = _make_pipeline("LABEL_2")
        result = score_document("", sentences, pipeline=pipe)
        assert isinstance(result, SentimentResult)

    def test_score_in_valid_range(self):
        sentences = [
            "Inflation remains elevated above target.",
            "Interest rates must rise to restore price stability.",
            "Labor market is strong but wage growth is moderate.",
        ]
        pipe = _make_pipeline("LABEL_1", "LABEL_1", "LABEL_0")
        result = score_document("", sentences, pipeline=pipe)
        assert -1.0 <= result.hawkish_score <= 1.0
