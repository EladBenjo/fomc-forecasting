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

### Phase 2: Feature Engineering (active/mostly complete)

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

---

### Phase 3: Target Variable + Dataset Builder (notebook-first execution)

**Goal:** Fetch inflation expectations from FRED and build leak-free modeling datasets.

Locked decisions:
- Target scope: `T5YIE` first
- Row unit: include both `document` and `speech` feature rows in one merged dataset
- Time alignment: as-of join using latest target on or before each feature date (no lookahead)
- Fixed calendar splits:
  - train: `date <= 2016-12-31`
  - val: `2017-01-01 <= date <= 2020-12-31`
  - test: `date >= 2021-01-01`

Execution slices (must stay separate):
1. TS fetch
2. TS transformation
3. Time alignment + dataset build

Planned artifacts:
- `data/targets/t5yie.parquet`
- `data/targets/model_dataset_t5yie.parquet`
- `data/splits/time_splits.json`

---

### Phase 3 Notebook-First Gate (Active)

Production Phase 3 code starts only after both notebooks are approved:

1. `notebooks/T5YIE_ts_analysis.ipynb` finalized for stationarity and transformation decisions.
2. Joint TS+text EDA notebook finalized for merge/alignment decisions, leakage checks, and schema lock-in.
3. Only then implement production modules from the approved notebook decisions.

Notebook-first policy:
- Notebook findings are the source of truth for Phase 3 implementation choices.
- No production-code mutations during notebook exploration.
- Small commits only: one logical step per commit.

---

### Phase 4: Baseline Models

**Goal:** Establish performance floor before ML.

- `models/baselines/ar.py` - AR(p)
- `models/baselines/sarimax.py` - SARIMAX + text features
- `models/evaluation/walk_forward.py` - rolling/expanding window CV
- `models/evaluation/metrics.py` - MAE, RMSE, directional accuracy
- `models/tracking/run_logger.py` - file-based experiment tracking
- `notebooks/20_modeling.ipynb`

---

### Phase 5: ML Models

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
  -> Phase 2 (active/mostly complete)
  -> Phase 3 notebook gates (active)
  -> Phase 3 production implementation
  -> Phase 4 baselines
  -> Phase 5 ML
  -> Phase 6 RAG
  -> Phase 7 Streamlit + CLI polish
```
