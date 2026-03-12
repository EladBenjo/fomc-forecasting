"""
Minimal text normalization for Fed documents.

Keeps original casing and punctuation — FOMC-RoBERTa is case-sensitive and
was trained on unmodified Fed text. We only fix encoding artifacts and
excessive whitespace that accumulate during PDF extraction.
"""

import re


def normalize(text: str) -> str:
    """Clean whitespace and encoding artifacts from extracted Fed text."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines (PDF extraction leaves many)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse inline whitespace runs
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, then further split at contrastive conjunctions.

    Ported from the original research notebook. The extra splits at words like
    'but' and 'however' matter for sentiment: 'inflation is contained but risks
    remain elevated' should produce two separate sentiment signals.
    """
    _CONTRASTIVE = {"but", "however", "although", "while"}

    # Basic sentence splitting on terminal punctuation
    raw = re.split(r"(?<=[.!?])\s+", text)

    sentences: list[str] = []
    for sent in raw:
        sent = sent.strip()
        if not sent:
            continue
        split_done = False
        for kw in _CONTRASTIVE:
            pattern = rf"\b{kw}\b"
            if re.search(pattern, sent, re.IGNORECASE):
                parts = re.split(pattern, sent, maxsplit=1, flags=re.IGNORECASE)
                sentences.extend(p.strip() for p in parts if p.strip())
                split_done = True
                break
        if not split_done:
            sentences.append(sent)

    return sentences
