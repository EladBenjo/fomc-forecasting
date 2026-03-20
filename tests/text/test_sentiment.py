"""Tests for fedtext.text.features.sentiment using mocked ZettaQuant API calls."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from fedtext.text.features import sentiment
from fedtext.text.features.sentiment import SentimentResult, ZettaQuantClient, load_client, score_document


class _Resp:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            exc = requests.HTTPError(f"HTTP {self.status_code}")
            exc.response = type("Resp", (), {"status_code": self.status_code})()
            raise exc

    def json(self) -> dict:
        return self._payload


class TestScoreDocument:
    def test_relevancy_filter_then_stance_counts(self):
        calls: list[dict] = []

        def _post(url, headers, json, timeout):
            calls.append(json)
            if json["model_id"] == sentiment.ZQ_RELEVANCY_MODEL_ID:
                return _Resp(
                    {
                        "predictions": [
                            {"label": "Relevant"},
                            {"label": "Irrelevant"},
                            {"label": "Relevant"},
                        ]
                    }
                )
            return _Resp({"predictions": [{"label": "Hawkish"}, {"label": "Neutral"}]})

        client = ZettaQuantClient(api_key="k", batch_size=10, max_req_per_min=1_000_000, max_retries=0)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(sentiment.requests, "post", _post)
            result = score_document(
                "",
                [
                    "Inflation remains elevated above target.",
                    "The meeting was adjourned.",
                    "Policy should remain restrictive for longer.",
                ],
                client=client,
            )
        client.close()

        assert isinstance(result, SentimentResult)
        assert result.n_hawkish == 1
        assert result.n_dovish == 0
        assert result.n_neutral == 1
        assert result.n_target_sentences == 2
        assert result.hawkish_score == pytest.approx(0.5)
        assert len(calls) == 2

    def test_stance_irrelevant_is_excluded_from_target_count(self):
        def _post(url, headers, json, timeout):
            if json["model_id"] == sentiment.ZQ_RELEVANCY_MODEL_ID:
                return _Resp({"predictions": [{"label": "Relevant"}, {"label": "Relevant"}]})
            return _Resp({"predictions": [{"label": "Hawkish"}, {"label": "Irrelevant"}]})

        client = ZettaQuantClient(api_key="k", max_req_per_min=1_000_000, max_retries=0)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(sentiment.requests, "post", _post)
            result = score_document(
                "",
                [
                    "Inflation is above target and broadening.",
                    "The vote was unanimous and procedural.",
                ],
                client=client,
            )
        client.close()

        assert result.n_hawkish == 1
        assert result.n_target_sentences == 1
        assert result.hawkish_score == pytest.approx(1.0)

    def test_no_relevant_sentences_returns_neutral(self):
        def _post(url, headers, json, timeout):
            return _Resp({"predictions": [{"label": "Irrelevant"}] * len(json["instances"])})

        client = ZettaQuantClient(api_key="k", max_req_per_min=1_000_000, max_retries=0)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(sentiment.requests, "post", _post)
            result = score_document("", ["This is long enough sentence."], client=client)
        client.close()

        assert result.n_target_sentences == 0
        assert result.hawkish_score == pytest.approx(0.0)


class TestZettaQuantClient:
    def test_cache_hit_skips_repeated_api_calls(self, tmp_path: Path):
        call_count = 0

        def _post(url, headers, json, timeout):
            nonlocal call_count
            call_count += 1
            return _Resp({"predictions": [{"label": "Relevant"}] * len(json["instances"])})

        client = ZettaQuantClient(
            api_key="k",
            batch_size=2,
            max_req_per_min=1_000_000,
            max_retries=0,
            cache_db_path=tmp_path / "state.sqlite3",
        )
        sentences = [
            "Sentence one long enough.",
            "Sentence two long enough.",
            "Sentence three long enough.",
        ]

        with pytest.MonkeyPatch.context() as m:
            m.setattr(sentiment.requests, "post", _post)
            labels1 = client.classify_relevancy(sentences)
            labels2 = client.classify_relevancy(sentences)
        client.close()

        assert call_count == 2
        assert labels1 == ["Relevant", "Relevant", "Relevant"]
        assert labels2 == ["Relevant", "Relevant", "Relevant"]

    def test_retry_backoff_on_timeout_then_success(self):
        attempts = 0

        def _post(url, headers, json, timeout):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise requests.Timeout("network timeout")
            return _Resp({"predictions": [{"label": "Relevant"}] * len(json["instances"])})

        client = ZettaQuantClient(api_key="k", max_req_per_min=1_000_000, max_retries=5)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(sentiment.requests, "post", _post)
            m.setattr(sentiment.time, "sleep", lambda _seconds: None)
            labels = client.classify_relevancy(["This sentence is long enough."])
        client.close()

        assert attempts == 3
        assert labels == ["Relevant"]

    def test_raises_when_retry_budget_exhausted(self):
        def _post(url, headers, json, timeout):
            raise requests.Timeout("still failing")

        client = ZettaQuantClient(api_key="k", max_req_per_min=1_000_000, max_retries=2)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(sentiment.requests, "post", _post)
            m.setattr(sentiment.time, "sleep", lambda _seconds: None)
            with pytest.raises(requests.Timeout):
                client.classify_relevancy(["This sentence is long enough."])
        client.close()

    def test_raises_on_malformed_response(self):
        def _post(url, headers, json, timeout):
            return _Resp({"predictions": "not-a-list"})

        client = ZettaQuantClient(api_key="k", max_req_per_min=1_000_000, max_retries=0)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(sentiment.requests, "post", _post)
            with pytest.raises(ValueError, match="predictions"):
                client.classify_relevancy(["This sentence is long enough."])
        client.close()


class TestLoadClient:
    def test_load_client_requires_api_key(self):
        with pytest.MonkeyPatch.context() as m:
            m.delenv("ZQ_API_KEY", raising=False)
            with pytest.raises(RuntimeError, match="ZQ_API_KEY"):
                load_client()

    def test_load_client_reads_optional_env(self):
        with pytest.MonkeyPatch.context() as m:
            m.setenv("ZQ_API_KEY", "abc")
            m.setenv("ZQ_BATCH_SIZE", "12")
            m.setenv("ZQ_MAX_REQ_PER_MIN", "34")
            m.setenv("ZQ_TIMEOUT_SECONDS", "9.5")
            m.setenv("ZQ_MAX_RETRIES", "7")
            client = load_client()

        assert client.batch_size == 12
        assert client.max_req_per_min == 34
        assert client.timeout_seconds == 9.5
        assert client.max_retries == 7
        client.close()
