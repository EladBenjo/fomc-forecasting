"""
Hawkish/dovish sentiment scoring using FOMC-RoBERTa.

Model: gtfintechlab/FOMC-RoBERTa
  - Fine-tuned on FOMC communications
  - 3-class: LABEL_0=Dovish, LABEL_1=Hawkish, LABEL_2=Neutral
  - Reference: Shah et al. (2023), ACL. https://huggingface.co/gtfintechlab/FOMC-RoBERTa

Design:
  - Run at sentence level (model was trained on sentences, not documents)
  - Pre-filter sentences by economic keyword presence to cut inference time ~5x
  - Score = (n_hawkish - n_dovish) / n_target_sentences ∈ [-1, 1]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Economic keyword filter (ported from research notebook)
# Keeps sentences that mention topics relevant to monetary policy sentiment.
# ---------------------------------------------------------------------------

_ECONOMIC_KEYWORDS: set[str] = {
    # Inflation & expectations
    "inflation", "inflation expectation", "core cpi", "core pce",
    "price stability", "commodity prices", "price level", "deflation",
    "disinflation", "inflationary", "price pressure",
    # Interest rates & monetary policy
    "interest rate", "federal funds rate", "rate hike", "rate cut",
    "hawkish", "dovish", "forward guidance", "accommodative", "restrictive",
    "tightening", "easing", "quantitative easing", "balance sheet",
    "tapering", "normalization",
    # Labor market & growth
    "employment", "unemployment", "labor market", "wage growth",
    "payroll", "gdp", "economic growth", "recession", "soft landing",
    "output gap",
    # Financial stability
    "exchange rate", "credit spreads", "yield curve", "financial conditions",
    "banking sector", "liquidity", "systemic risk",
}


def _is_economic(sentence: str) -> bool:
    low = sentence.lower()
    return any(kw in low for kw in _ECONOMIC_KEYWORDS)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    hawkish_score: float    # (n_hawkish - n_dovish) / n_target; ∈ [-1, 1]
    n_hawkish: int
    n_dovish: int
    n_neutral: int
    n_target_sentences: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_document(
    text: str,
    sentences: list[str],
    *,
    pipeline,             # transformers.Pipeline — caller owns the lifecycle
    max_length: int = 512,
) -> SentimentResult:
    """
    Classify each economic-relevant sentence and return aggregate sentiment.

    Parameters
    ----------
    text:       Full document text (unused here, kept for future use).
    sentences:  Pre-split sentences from normalizer.split_sentences().
    pipeline:   A loaded transformers text-classification pipeline for
                gtfintechlab/FOMC-RoBERTa.
    max_length: Truncation limit — model max is 512 tokens.
    """
    target = [s for s in sentences if _is_economic(s) and len(s) > 10]

    if not target:
        logger.debug("No economic sentences found — returning neutral result.")
        return SentimentResult(
            hawkish_score=0.0,
            n_hawkish=0,
            n_dovish=0,
            n_neutral=0,
            n_target_sentences=0,
        )

    results = pipeline(
        target,
        truncation=True,
        max_length=max_length,
        batch_size=32,
    )

    counts = {"LABEL_0": 0, "LABEL_1": 0, "LABEL_2": 0}
    for r in results:
        counts[r["label"]] += 1

    n_dovish  = counts["LABEL_0"]
    n_hawkish = counts["LABEL_1"]
    n_neutral = counts["LABEL_2"]
    n_total   = len(target)

    score = (n_hawkish - n_dovish) / n_total if n_total else 0.0

    return SentimentResult(
        hawkish_score=score,
        n_hawkish=n_hawkish,
        n_dovish=n_dovish,
        n_neutral=n_neutral,
        n_target_sentences=n_total,
    )


def load_pipeline(device: int = -1):
    """
    Load the FOMC-RoBERTa classification pipeline.

    Parameters
    ----------
    device: -1 = CPU, 0 = first GPU. Use 0 if a GPU is available.
    """
    from transformers import pipeline as hf_pipeline

    logger.info("Loading gtfintechlab/FOMC-RoBERTa (device=%d)...", device)
    return hf_pipeline(
        "text-classification",
        model="gtfintechlab/FOMC-RoBERTa",
        device=device,
    )
