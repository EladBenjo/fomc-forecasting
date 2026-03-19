"""
Hawkish/dovish sentiment scoring using ZettaQuant classifiers.

Models:
  - Relevancy: cb_inflation_relevancy_label
  - Stance:    cb_stance_label

Design:
  - Run at sentence level
  - Filter with API-based inflation relevancy model
  - Score = (n_hawkish - n_dovish) / n_target_sentences in [-1, 1]
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API and model settings
# ---------------------------------------------------------------------------

ZQ_BASE_URL = "https://api.zettaquant.ai"
ZQ_INFER_PATH = "/v1/models/infer"
ZQ_RELEVANCY_MODEL_ID = "cb_inflation_relevancy_label"
ZQ_STANCE_MODEL_ID = "cb_stance_label"

DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_REQ_PER_MIN = 10
DEFAULT_TIMEOUT_SECONDS = 30.0

_RELEVANT_LABELS = {
    "relevant",
    "inflation relevant",
    "inflation_relevant",
    "yes",
    "true",
    "1",
}

_STANCE_LABEL_MAP = {
    "hawkish": "Hawkish",
    "dovish": "Dovish",
    "neutral": "Neutral",
    "irrelevant": "Irrelevant",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    hawkish_score: float    # (n_hawkish - n_dovish) / n_target; in [-1, 1]
    n_hawkish: int
    n_dovish: int
    n_neutral: int
    n_target_sentences: int


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class ZettaQuantClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = ZQ_BASE_URL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_req_per_min: int = DEFAULT_MAX_REQ_PER_MIN,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if max_req_per_min <= 0:
            raise ValueError("max_req_per_min must be > 0")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.max_req_per_min = max_req_per_min
        self.timeout_seconds = timeout_seconds
        self._headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }
        self._min_interval_seconds = 60.0 / float(max_req_per_min)
        self._last_request_monotonic: float | None = None

    def classify_relevancy(self, sentences: list[str]) -> list[str]:
        return self._infer_labels(model_id=ZQ_RELEVANCY_MODEL_ID, sentences=sentences)

    def classify_stance(self, sentences: list[str]) -> list[str]:
        return self._infer_labels(model_id=ZQ_STANCE_MODEL_ID, sentences=sentences)

    def _infer_labels(self, *, model_id: str, sentences: list[str]) -> list[str]:
        labels: list[str] = []
        for chunk in _chunks(sentences, self.batch_size):
            self._pace()
            payload = {
                "model_id": model_id,
                "instances": [{"text": s} for s in chunk],
            }
            url = f"{self.base_url}{ZQ_INFER_PATH}"
            response = requests.post(
                url,
                headers=self._headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            predictions = data.get("predictions")
            if not isinstance(predictions, list):
                raise ValueError("Invalid ZettaQuant response: 'predictions' must be a list.")
            if len(predictions) != len(chunk):
                raise ValueError(
                    "Invalid ZettaQuant response: predictions count does not match instances count."
                )
            for pred in predictions:
                if not isinstance(pred, dict) or "label" not in pred:
                    raise ValueError("Invalid ZettaQuant response: missing prediction label.")
                labels.append(str(pred["label"]).strip())
        return labels

    def _pace(self) -> None:
        now = time.monotonic()
        if self._last_request_monotonic is not None:
            elapsed = now - self._last_request_monotonic
            if elapsed < self._min_interval_seconds:
                time.sleep(self._min_interval_seconds - elapsed)
        self._last_request_monotonic = time.monotonic()


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_document(
    text: str,
    sentences: list[str],
    *,
    client: ZettaQuantClient,
) -> SentimentResult:
    """
    Classify each relevancy-filtered sentence and return aggregate sentiment.

    Parameters
    ----------
    text:       Full document text (unused here, kept for future use).
    sentences:  Pre-split sentences from normalizer.split_sentences().
    client:     ZettaQuant API client.
    """
    del text  # reserved for future use

    candidates = [s for s in sentences if len(s) > 10]
    if not candidates:
        return SentimentResult(
            hawkish_score=0.0,
            n_hawkish=0,
            n_dovish=0,
            n_neutral=0,
            n_target_sentences=0,
        )

    relevancy_labels = client.classify_relevancy(candidates)
    target = [
        sentence
        for sentence, label in zip(candidates, relevancy_labels)
        if label.strip().lower() in _RELEVANT_LABELS
    ]

    if not target:
        logger.debug("No relevant sentences found; returning neutral result.")
        return SentimentResult(
            hawkish_score=0.0,
            n_hawkish=0,
            n_dovish=0,
            n_neutral=0,
            n_target_sentences=0,
        )

    stance_labels = client.classify_stance(target)

    n_hawkish = 0
    n_dovish = 0
    n_neutral = 0
    n_total = 0

    for label in stance_labels:
        key = label.strip().lower()
        canonical = _STANCE_LABEL_MAP.get(key)
        if canonical is None:
            raise ValueError(f"Unexpected stance label from ZettaQuant: {label!r}")
        if canonical == "Irrelevant":
            continue
        n_total += 1
        if canonical == "Hawkish":
            n_hawkish += 1
        elif canonical == "Dovish":
            n_dovish += 1
        elif canonical == "Neutral":
            n_neutral += 1

    score = (n_hawkish - n_dovish) / n_total if n_total else 0.0

    return SentimentResult(
        hawkish_score=score,
        n_hawkish=n_hawkish,
        n_dovish=n_dovish,
        n_neutral=n_neutral,
        n_target_sentences=n_total,
    )


def load_client() -> ZettaQuantClient:
    """
    Load the ZettaQuant API client from environment variables.
    """
    api_key = os.getenv("ZQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing required environment variable: ZQ_API_KEY")

    batch_size = int(os.getenv("ZQ_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    max_req_per_min = int(os.getenv("ZQ_MAX_REQ_PER_MIN", str(DEFAULT_MAX_REQ_PER_MIN)))
    timeout_seconds = float(os.getenv("ZQ_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))

    logger.info(
        "Initializing ZettaQuant client (batch_size=%d, max_req_per_min=%d, timeout_seconds=%.1f)",
        batch_size,
        max_req_per_min,
        timeout_seconds,
    )
    return ZettaQuantClient(
        api_key=api_key,
        batch_size=batch_size,
        max_req_per_min=max_req_per_min,
        timeout_seconds=timeout_seconds,
    )
