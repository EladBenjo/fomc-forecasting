"""Utility helpers for document-level text feature extraction."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

_ROLE_NAME_MAP = {
    # Chairs / Chairmen
    "alan greenspan": "Chairman",
    "ben bernanke": "Chairman",
    "ben s bernanke": "Chairman",
    "janet yellen": "Chairman",
    "janet l yellen": "Chairman",
    "jerome powell": "Chairman",
    "jerome h powell": "Chairman",
    # Vice Chairs / Vice Chairmen
    "alice rivlin": "Vice Chairman",
    "alice m rivlin": "Vice Chairman",
    "roger ferguson": "Vice Chairman",
    "roger w ferguson": "Vice Chairman",
    "donald kohn": "Vice Chairman",
    "donald l kohn": "Vice Chairman",
    "stanley fischer": "Vice Chairman",
    "richard clarida": "Vice Chairman",
    "richard h clarida": "Vice Chairman",
    "lael brainard": "Vice Chairman",
    "philip jefferson": "Vice Chairman",
    "philip n jefferson": "Vice Chairman",
    "michael barr": "Vice Chairman",
    "michael s barr": "Vice Chairman",
    # Governors
    "edward kelley": "Governor",
    "edward w kelley": "Governor",
    "laurence meyer": "Governor",
    "laurence h meyer": "Governor",
    "lawrence lindsey": "Governor",
    "lawrence b lindsey": "Governor",
    "edward gramlich": "Governor",
    "edward m gramlich": "Governor",
    "mark olson": "Governor",
    "mark w olson": "Governor",
    "susan bies": "Governor",
    "susan s bies": "Governor",
    "kevin warsh": "Governor",
    "kevin m warsh": "Governor",
    "randall kroszner": "Governor",
    "randall s kroszner": "Governor",
    "frederic mishkin": "Governor",
    "frederic s mishkin": "Governor",
    "elizabeth duke": "Governor",
    "elizabeth a duke": "Governor",
    "sarah bloom raskin": "Governor",
    "jeremy stein": "Governor",
    "jeremy c stein": "Governor",
    "daniel tarullo": "Governor",
    "daniel k tarullo": "Governor",
    "randal quarles": "Governor",
    "randal k quarles": "Governor",
    "michelle bowman": "Governor",
    "michelle w bowman": "Governor",
    "christopher waller": "Governor",
    "christopher j waller": "Governor",
    "lisa cook": "Governor",
    "lisa d cook": "Governor",
    "adriana kugler": "Governor",
    "adriana d kugler": "Governor",
}

_VICE_CHAIR_RE = re.compile(
    r"\bvice(?:\s|-)?(?:chair(?:man)?|chair\s+for\s+supervision)\b",
    flags=re.IGNORECASE,
)
_CHAIR_RE = re.compile(r"\bchair(?:man)?\b", flags=re.IGNORECASE)
_GOVERNOR_RE = re.compile(r"\bgovernor\b", flags=re.IGNORECASE)


def count_words(text: str) -> int:
    """Count word-like tokens from canonical document text."""
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def compute_target_sentences_ratio(
    *,
    n_target_sentences: int,
    total_sentences_count: int,
) -> float:
    """Compute target-sentence share over all split sentences."""
    if total_sentences_count <= 0:
        return 0.0
    return float(n_target_sentences) / float(total_sentences_count)


def infer_role(
    *,
    source_type: str,
    speaker: str | None = None,
    title: str | None = None,
    event: str | None = None,
) -> str | None:
    """
    Infer role for speech records using metadata keywords and a curated name map.

    Returns one of {"Chairman", "Vice Chairman", "Governor"} or None.
    Non-speech records always return None.
    """
    if str(source_type).lower() != "speech":
        return None

    metadata_blob = " ".join([speaker or "", title or "", event or ""]).strip()
    if metadata_blob:
        if _VICE_CHAIR_RE.search(metadata_blob):
            return "Vice Chairman"
        if _CHAIR_RE.search(metadata_blob):
            return "Chairman"
        if _GOVERNOR_RE.search(metadata_blob):
            return "Governor"

    normalized_name = _normalize_person_name(speaker or "")
    return _ROLE_NAME_MAP.get(normalized_name)


def _normalize_person_name(name: str) -> str:
    lowered = name.strip().lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\b(?:jr|sr|ii|iii|iv)\b", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


__all__ = [
    "count_words",
    "compute_target_sentences_ratio",
    "infer_role",
]
