"""Tests for fedtext.text.features.sentiment using mocked ZettaQuant API calls."""

from __future__ import annotations

import requests
import pytest

from fedtext.text.features import sentiment
from fedtext.text.features.sentiment import SentimentResult, ZettaQuantClient, load_client, score_document


class _Resp:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class TestScoreDocument:
    def test_relevancy_filter_then_stance_counts(self):
        calls: list[dict] = []

        def _post(url, headers, json, timeout):
            calls.append(json)
            if json["model_id"] == sentiment.ZQ_RELEVANCY_MODEL_ID:
                # 3 inputs -> keep only first and third
                return _Resp({
                    "predictions": [
                        {"label": "Relevant"},
                        {"label": "Irrelevant"},
                        {"label": "Relevant"},
                    ]
                })
            return _Resp({
                "predictions": [
                    {"label": "Hawkish"},
                    {"label": "Neutral"},
                ]
            })

        client = ZettaQuantClient(api_key="k", batch_size=10, max_req_per_min=1_000_000)
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

        client = ZettaQuantClient(api_key="k", max_req_per_min=1_000_000)
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

        assert result.n_hawkish == 1
        assert result.n_target_sentences == 1
        assert result.hawkish_score == pytest.approx(1.0)

    def test_no_relevant_sentences_returns_neutral(self):
        def _post(url, headers, json, timeout):
            return _Resp({"predictions": [{"label": "Irrelevant"}] * len(json["instances"])})

        client = ZettaQuantClient(api_key="k", max_req_per_min=1_000_000)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(sentiment.requests, "post", _post)
            result = score_document("", ["This is long enough sentence."], client=client)

        assert result.n_target_sentences == 0
        assert result.hawkish_score == pytest.approx(0.0)


class TestZettaQuantClient:
    def test_batches_instances(self):
        seen_sizes: list[int] = []

        def _post(url, headers, json, timeout):
            seen_sizes.append(len(json["instances"]))
            return _Resp({"predictions": [{"label": "Relevant"}] * len(json["instances"])})

        client = ZettaQuantClient(api_key="k", batch_size=2, max_req_per_min=1_000_000)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(sentiment.requests, "post", _post)
            labels = client.classify_relevancy([
                "Sentence one long enough.",
                "Sentence two long enough.",
                "Sentence three long enough.",
                "Sentence four long enough.",
                "Sentence five long enough.",
            ])

        assert seen_sizes == [2, 2, 1]
        assert labels == ["Relevant"] * 5

    def test_raises_on_http_error(self):
        def _post(url, headers, json, timeout):
            return _Resp({"predictions": []}, status_code=429)

        client = ZettaQuantClient(api_key="k", max_req_per_min=1_000_000)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(sentiment.requests, "post", _post)
            with pytest.raises(requests.HTTPError):
                client.classify_relevancy(["This sentence is long enough."])

    def test_raises_on_malformed_response(self):
        def _post(url, headers, json, timeout):
            return _Resp({"predictions": "not-a-list"})

        client = ZettaQuantClient(api_key="k", max_req_per_min=1_000_000)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(sentiment.requests, "post", _post)
            with pytest.raises(ValueError, match="predictions"):
                client.classify_relevancy(["This sentence is long enough."])


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
            client = load_client()

        assert client.batch_size == 12
        assert client.max_req_per_min == 34
        assert client.timeout_seconds == 9.5
