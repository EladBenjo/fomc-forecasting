# Phase 3 Notebook-First Plan (Persistent Source of Truth)

## Goal

Close Phase 3 with a production-ready T5YIE target pipeline and leak-free merged dataset, using notebook findings as the authoritative specification before implementation.

## Current-State Snapshot

- Ingest and DB consolidation are complete.
- Feature pipeline exists and currently outputs document-level sentiment + novelty to `data/features/doc_level/features.parquet`.
- Sentiment uses ZettaQuant API (`cb_inflation_relevancy_label`, `cb_stance_label`).
- Feature extraction reliability addendum is active: checkpoint + cache + atomic finalize.
- Topic modeling is deferred.
- This phase is intentionally notebook-gated before additional production coding.

## Locked Decisions

- Target scope: T5YIE first.
- Row unit for merged dataset: both `document` and `speech` feature rows.
- Time alignment policy: as-of join using latest target on or before feature date (no lookahead).
- Split boundaries:
  - train: `date <= 2016-12-31`
  - val: `2017-01-01 <= date <= 2020-12-31`
  - test: `date >= 2021-01-01`

## Notebook Acceptance Checklists

### Notebook 1: `notebooks/T5YIE_ts_analysis.ipynb`

Must explicitly decide and document:
- Final target representation(s) to carry into production (level, diff, or both).
- ADF/KPSS interpretation and final stationarity decision used for modeling targets.
- Missing-value policy and any date-frequency assumptions.

Completion criteria:
- The chosen target transform is explicit and reproducible.
- The rationale is written in notebook markdown, not implied by code only.

### Notebook 2: Joint TS + Text EDA Notebook

Must explicitly decide and document:
- Exact as-of merge behavior and no-lookahead checks.
- Coverage and missingness behavior by `source_type` (`document`, `speech`).
- Final v1 feature/target columns and schema assumptions for builder code.

Completion criteria:
- Merge policy and leakage checks are validated on real data.
- Final dataset shape assumptions are clear enough to implement directly.

## Post-Gate Implementation Sequence

Implementation begins only after both notebook checklists pass.

1. **Fetch module**
- Fetch T5YIE from FRED using `FRED_API_KEY`.
- Write `data/targets/t5yie.parquet` with standardized date/value columns.

2. **Transformation module**
- Apply notebook-approved target transforms.
- Persist transformed columns and metadata needed by builder.

3. **Alignment + dataset builder**
- Merge features with transformed target via as-of on/before date.
- Include both `document` and `speech` rows.
- Write `data/targets/model_dataset_t5yie.parquet`.

4. **Split artifact generation**
- Generate `data/splits/time_splits.json` with locked cutoffs.

5. **Phase 4 interface stubs**
- Add baseline model/evaluation/tracking stubs only after Phase 3 artifacts are stable.

## Artifact Contract

Phase 3 output artifacts:
- `data/targets/t5yie.parquet`
- `data/targets/model_dataset_t5yie.parquet`
- `data/splits/time_splits.json`

## Execution Policy

- Notebook-first workflow is authoritative for Phase 3 decisions.
- No production-code mutations during notebook exploration.
- Commit discipline is mandatory: one small logical commit per step.

## Feature Extraction Reliability & Cost Guardrails

This addendum is active during notebook work and later implementation:
- Use SQLite document checkpointing for resumable feature extraction.
- Use sentence-level inference cache keyed by `(model_id, sentence_hash, sentence_text)` to reduce repeated ZettaQuant calls.
- Use atomic parquet finalization to avoid partial output corruption on interruption.

Operational defaults:
- Resume enabled by default.
- Checkpoint flush every 25 docs.
- Exponential backoff retry budget of 5 for transient API failures.
