# FOMC Forecasting - Full Project Roadmap

## Context

This project demonstrates a production-grade NLP + ML pipeline for predicting inflation expectations from Federal Reserve communications.

Primary deliverables:
- A clean Jupyter notebook flow showing the end-to-end quantitative pipeline.
- A Streamlit app with ingest status, feature charts, model predictions, and a RAG chat interface grounded in source documents.

Strategy: finish the quantitative pipeline first, then layer in RAG.

---

## Phase Roadmap

### Phase 1: Ingest Hardening (done)

**Goal:** Make the ingest layer reliable and portfolio-presentable.

- `common/paths.py` - centralized path constants
- `configs/sources.yaml` - URLs, rate limits, category rules
- `ingest/storage/migrations/` - versioned SQL schema files
- `ingest/validators/completeness.py` - data quality checks with CLI + API

---

### Phase 1.5: DB Consolidation (done)

**Goal:** Merge `speeches.db` and `catalog.sqlite` into a single `data/catalog/fedtext.db`.

- `ingest/storage/migrations/004_consolidate.sql` - unified schema
- `scripts/consolidate_dbs.py` - one-time migration via SQLite `ATTACH`
- Shared DB access through `common/paths.py` and `common/db.py`

---

### Phase 2: Feature Engineering (done)

**Goal:** Compute document-level text features and store them for modeling.

Current state:
- `text/cleaning/normalizer.py` - whitespace/encoding cleanup and sentence splitting (no lowercasing step)
- `text/features/sentiment.py` - ZettaQuant API sentiment stack:
  - relevancy model: `cb_inflation_relevancy_label`
  - stance model: `cb_stance_label`
- `text/features/novelty.py` - cosine-distance novelty vs prior same-source document
- Output artifact: `data/features/doc_level/features.parquet`

Status note:
- Topic modeling is **deferred** (not removed) and is not required to close current Phase 2/3 priorities.
- Non-blocking backlog item: persist sentence splits for auditability/reuse (optional reliability extension; does not reopen Phase 2).

### Feature Extraction Reliability & Cost Guardrails (done/operational)

Large feature runs are now expected to be checkpointed and cache-backed by default to avoid losing work or wasting API calls.

Mechanisms:
- SQLite document checkpoint state under `data/features/doc_level/` for resume-safe progress.
- SQLite sentence-level inference cache keyed by `(model_id, sentence_hash, sentence_text)` to avoid repeated ZettaQuant calls.
- Atomic parquet finalization (`temp -> replace`) so `features.parquet` is never partially overwritten.

Execution defaults:
- Resume behavior enabled by default.
- Checkpoint commit cadence every 25 docs.
- Transient API retries with exponential backoff (default max retries: 5).

Tradeoff summary:
- Checkpoint DB adds local state management but gives robust resume/idempotence.
- Sentence cache grows over time but materially reduces repeated API costs.
- Atomic finalize uses temporary extra disk space but avoids corrupt final outputs.

---

### Phase 3: Target Variable + Dataset Builder (implementation in progress)

**Goal:** Fetch inflation expectations from FRED and build leak-free modeling datasets.

Locked decisions:
- Target scope: `T5YIE` first
- Modeling dataset granularity: monthly
- Time alignment: target month `M` uses feature month `M-1` (strict no-lookahead)
- Fixed calendar splits:
  - train: `date <= 2016-12-31`
  - val: `2017-01-01 <= date <= 2020-12-31`
  - test: `date >= 2021-01-01`

Execution slices (must stay separate):
1. TS fetch
2. TS transformation
3. Time alignment + dataset build

Planned artifacts:
- `data/targets/t5yie_raw.parquet`
- `data/targets/t5yie_diff1.parquet`
- `data/targets/model_dataset_t5yie.parquet`
- `data/splits/time_splits.json`
- `data/targets/manifests/model-dataset-t5yie-*.json`
- `data/targets/model_dataset_registry.sqlite3`

Current state (as of April 2, 2026):
- T5YIE target fetch/transform pipeline is versioned and operational.
- T5YIE monthly model dataset builder is implemented with:
  - deterministic monthly aggregation + `M-1 -> M` alignment
  - explicit no-lookahead checks
  - fixed split artifact generation
  - manifest + registry version tracking for model dataset outputs
- Remaining Phase 3 scope includes extending equivalent production flow to additional target streams (e.g., `EXPINF1YR`).

---

### Phase 3 Notebook-First Gate (T5YIE complete, second stream pending - backlog)

TODO: using 2 time series for this project EXPINF1YR and T5YIE, 2 different streams. Update accordingly.

T5YIE production implementation followed this notebook gate. For additional target streams, keep the same gate:

1. `notebooks/T5YIE_ts_analysis.ipynb` finalized for stationarity and transformation decisions.
2. Joint TS+text EDA notebook finalized for merge/alignment decisions, leakage checks, and schema lock-in.
3. Only then implement production modules from the approved notebook decisions.

Notebook-first policy:
- Notebook findings are the source of truth for Phase 3 implementation choices.
- No production-code mutations during notebook exploration.
- Small commits only: one logical step per commit.

Implemented T5YIE production modules:
- `datasets/build_dataset/alignment.py` - monthly aggregation + no-lookahead alignment + missingness semantics
- `datasets/build_dataset/builder.py` - dataset build orchestration + split artifact + CLI
- `datasets/build_dataset/versioning.py` - model dataset manifest + registry writes

Operational command:
- `python -m datasets.build_dataset.builder`

---

### Phase 4: Baseline Models (done/operational)

**Goal:** Establish performance floor before ML.

- `models/baselines/sarimax.py` - Phase 4 benchmark runner + CLI (`python -m models.baselines.sarimax`)
- `models/evaluation/walk_forward.py` - leakage-safe expanding one-step SARIMAX evaluation
- `models/evaluation/metrics.py` - MAE, RMSE, directional accuracy
- `models/tracking/run_logger.py` - run manifest + registry tracking
- `notebooks/t5yie_phase4_baseline_vs_exog_benchmark.ipynb`

Operational artifacts:
- `data/models/baselines/t5yie/<run_version>/predictions.parquet`
- `data/models/baselines/t5yie/<run_version>/results_table.json`
- `data/models/baselines/t5yie/<run_version>/paired_comparison.json`
- `data/models/baselines/t5yie/<run_version>/run_summary.json`
- `data/models/baselines/t5yie/<run_version>/run_config.json`
- `data/models/baselines/manifests/<run_version>.json`
- `data/models/baselines/run_registry.sqlite3`

Status note:
- Baseline vs exogenous variants are productionized with deterministic evaluation and versioned artifacts.
- Current benchmark conclusion remains: no meaningful test-period lift from exogenous variants over univariate baseline.

---

### Phase 5: ML Models (next)

**Goal:** Beat baselines with gradient boosting on text features.

- `models/ml/xgboost_model.py` - XGBoost with walk-forward CV
- Feature importance plots in `reports/figures/`

---

### Phase 6: RAG Layer (main upgrade over notebook)

**Goal:** Enable semantic search over Fed communications.

- `text/embedding/embedder.py` - embedding workflow
- `text/embedding/store.py` - vector store integration
- Embed chunks and support question -> top-k chunk retrieval

---

### Phase 7: Streamlit Demo App

**Goal:** Polished portfolio demo.

Pages:
1. Pipeline Status - ingest counts, coverage, scrape summary
2. Feature Explorer - sentiment over time, novelty spikes, topic diagnostics
3. Model Results - walk-forward predictions, metrics, feature importance
4. RAG Chat - grounded Q&A on Fed communications

Structure: `app/main.py` + `app/pages/`

---

## Unified CLI (target state)

```text
fedtext ingest    # discovery + fetch + parse
fedtext chunk     # chunk parsed docs
fedtext embed     # embed all chunks
fedtext features  # compute sentiment/novelty
fedtext train     # build dataset + run models
fedtext app       # launch Streamlit
```

---

## Recommended Sequence

```text
Phase 1 (done)
  -> Phase 1.5 (done)
  -> Phase 2 (done)
  -> Phase 3 notebook gate + production implementation for T5YIE (done)
  -> Phase 3 extension to second target stream (active)
  -> Phase 4 baselines (done)
  -> Phase 5 ML
  -> Phase 6 RAG
  -> Phase 7 Streamlit + CLI polish
```
