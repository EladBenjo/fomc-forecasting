"""
Meeting-over-meeting novelty scoring via TF-IDF cosine distance.

Novelty measures how different a document is from the previous document of
the same type. High novelty = the Fed is saying something new; low novelty =
largely repeated language (e.g., boilerplate between consecutive statements).

Design choices:
  - TF-IDF (not embeddings): captures vocabulary shifts, not semantic
    similarity — we want lexical novelty, not topical similarity.
  - Fit on the full corpus so IDF weights reflect the whole document set.
  - Computed per source_type to avoid cross-type comparisons (comparing
    a statement to a speech is not meaningful).
  - First document in a series gets novelty=1.0 (no baseline to compare).
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_novelty(
    texts_by_date: dict[str, str],
    *,
    max_features: int = 10_000,
) -> dict[str, float]:
    """
    Compute novelty score for each document relative to its predecessor.

    Parameters
    ----------
    texts_by_date:
        Mapping of {date_str: document_text}, sorted chronologically by key.
        Dates should be ISO strings ("YYYY-MM-DD") so lexicographic sort
        matches chronological order.
    max_features:
        TF-IDF vocabulary size cap.

    Returns
    -------
    Mapping of {date_str: novelty_score ∈ [0, 1]}.
    The first date always gets 1.0 (no prior document).
    """
    if not texts_by_date:
        return {}

    dates = sorted(texts_by_date.keys())
    texts = [texts_by_date[d] for d in dates]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    novelty: dict[str, float] = {dates[0]: 1.0}
    for i in range(1, len(dates)):
        sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i - 1])[0, 0]
        novelty[dates[i]] = float(1.0 - np.clip(sim, 0.0, 1.0))

    return novelty


def compute_novelty_by_type(
    records: list[dict],
    *,
    max_features: int = 10_000,
) -> dict[str, float]:
    """
    Convenience wrapper: group records by source_type, compute novelty within
    each group, and return a flat {doc_id: novelty} mapping.

    Each record must have keys: "doc_id", "source_type", "date", "text".
    """
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["source_type"]].append(r)

    result: dict[str, float] = {}
    for source_type, group in groups.items():
        # Deduplicate by date — if two docs share a date, keep the longer text
        by_date: dict[str, dict] = {}
        for r in group:
            date = r["date"]
            if date not in by_date or len(r["text"]) > len(by_date[date]["text"]):
                by_date[date] = r

        texts_by_date = {date: rec["text"] for date, rec in by_date.items()}
        novelty_by_date = compute_novelty(texts_by_date, max_features=max_features)

        # Map back to doc_id — all docs on same date get the same novelty
        for r in group:
            result[r["doc_id"]] = novelty_by_date.get(r["date"], float("nan"))

    return result
