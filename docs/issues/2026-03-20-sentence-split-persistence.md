# Issue: Persist Sentence Splits for Auditability and Reuse

## Summary
Sentence splitting currently happens on the fly during feature extraction and is not persisted.  
This makes it hard to audit preprocessing behavior, debug model outputs, and reuse sentence boundaries in later pipelines.

## Problem
- We cannot inspect the exact sentence units sent to ZettaQuant models after a run.
- We cannot quantify sentence-level coverage/quality without rerunning splitting.
- Future sentence-level feature work (or alternative models) must re-split every time.

## Proposal
Add a durable sentence store generated during feature extraction:
- Persist normalized + split sentences per `(source_type, doc_id, sentence_idx)`.
- Include a stable `sentence_hash` for joins with cache/inference outputs.
- Keep provenance columns (`run_id`, `created_at`, splitter version).

## Suggested Storage
- SQLite table in `data/features/doc_level/features_state.sqlite3` (or dedicated DB):
  - `source_type TEXT`
  - `doc_id INTEGER`
  - `sentence_idx INTEGER`
  - `sentence_text TEXT`
  - `sentence_hash TEXT`
  - `run_id TEXT`
  - `created_at TEXT DEFAULT CURRENT_TIMESTAMP`
  - `PRIMARY KEY (source_type, doc_id, sentence_idx, run_id)`

## Acceptance Criteria
- Sentence rows are persisted for every processed document.
- Same document can be reproduced exactly for a given `run_id`.
- Basic sanity checks available:
  - sentence count per doc
  - empty/short sentence rate
  - top repeated sentences
- Pipeline resume mode does not duplicate sentence rows for already checkpointed docs.

## Tradeoffs
- Pros:
  - full audit trail of model input units
  - easier QA/debugging and downstream reuse
  - enables sentence-level analytics without recomputation
- Cons:
  - added storage growth
  - additional write overhead during feature runs
  - versioning complexity when splitter logic changes

## Priority
Medium-high (Phase 2 reliability/observability extension).

## Labels (if opened on GitHub)
- `enhancement`
- `pipeline`
- `reliability`
- `nlp`
