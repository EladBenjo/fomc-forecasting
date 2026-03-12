# FOMC Forecasting — Full Project Roadmap

## Context

This is a portfolio project demonstrating a production-grade NLP + ML pipeline for predicting
inflation expectations from Federal Reserve communications. The end deliverable is:

- A clean **Jupyter notebook** showing the end-to-end pipeline
- A **Streamlit app** with: ingest status, feature charts, model predictions, and a RAG chat
  interface for querying Fed documents

RAG is the main upgrade vs. the original notebook. The approach is: solidify the quantitative
pipeline first, then layer in RAG at the end.

---

## Phase Roadmap

### Phase 1: Ingest Hardening ✅

**Goal:** Make the ingest layer reliable and portfolio-presentable.

- `common/paths.py` — centralised path constants
- `configs/sources.yaml` — URLs, rate limits, category rules
- `ingest/storage/migrations/` — versioned SQL schema files
- `ingest/validators/completeness.py` — data quality checks with CLI + API

---

### Phase 1.5: DB Consolidation

**Goal:** Merge `speeches.db` and `catalog.sqlite` into a single `data/catalog/fedtext.db`.

The split was a historical accident (speeches prototyped first). The `chunks` table already has
a `source_type` column designed for this. One DB = simpler connections, easier cross-source
queries, cleaner portfolio story.

- `ingest/storage/migrations/004_consolidate.sql` — unified schema
- `scripts/consolidate_dbs.py` — one-time migration via SQLite `ATTACH`
- Update `common/paths.py`, `common/db.py`, and all pipeline imports

---

### Phase 2: Feature Engineering

**Goal:** Compute document-level text features and store them for modeling.

- `text/cleaning/normalizer.py` — lowercasing, whitespace, boilerplate removal
- `text/features/sentiment.py` — hawkish/dovish scores (VADER → ROBER-FOMC)
- `text/features/novelty.py` — cosine distance vs. previous meeting
- `text/features/topics.py` — LDA topic distribution
- Output: `data/features/doc_level/` as parquet
- `notebooks/10_eda.ipynb` — feature exploration

---

### Phase 3: Target Variable + Dataset Builder

**Goal:** Fetch inflation expectations from FRED and join with text features.

- FRED fetcher (`MICH` or `T5YIE` — higher-frequency series preferred)
- `datasets/build_dataset/builder.py` — merge features + target on meeting date
- `datasets/schema/fields.py` — canonical column names and dtypes
- `data/splits/time_splits.json` — fixed train/val/test boundaries (no leakage)

---

### Phase 4: Baseline Models

**Goal:** Establish performance floor before ML.

- `models/baselines/ar.py` — AR(p)
- `models/baselines/sarimax.py` — SARIMAX + text features
- `models/evaluation/walk_forward.py` — rolling/expanding window CV
- `models/evaluation/metrics.py` — MAE, RMSE, directional accuracy
- `models/tracking/run_logger.py` — file-based experiment tracking
- `notebooks/20_modeling.ipynb`

---

### Phase 5: ML Models

**Goal:** Beat baselines with gradient boosting on text features.

- `models/ml/xgboost_model.py` — XGBoost with walk-forward CV
- Feature importance plots → `reports/figures/`

---

### Phase 6: RAG Layer *(main upgrade over notebook)*

**Goal:** Enable semantic search over Fed communications.

- `text/embedding/embedder.py` — `sentence-transformers` (`all-MiniLM-L6-v2`)
- `text/embedding/store.py` — `sqlite-vec` vector store
- Embed all chunks; store in `embedding BLOB` column (already in schema)
- Query function: question → top-k chunks with metadata
- `notebooks/00_quickstart.ipynb` — end-to-end RAG demo

---

### Phase 7: Streamlit Demo App

**Goal:** Polished portfolio demo.

Pages:
1. **Pipeline Status** — ingest counts, coverage chart, scrape summary
2. **Feature Explorer** — sentiment over time, novelty spikes, topic trends
3. **Model Results** — walk-forward predictions vs. actuals, metric table, feature importance
4. **RAG Chat** — ask a question about Fed policy, get answer grounded in source documents

Structure: `app/main.py` + `app/pages/`

---

## Unified CLI

```
fedtext ingest    # discovery + fetch + parse
fedtext chunk     # chunk parsed docs
fedtext embed     # embed all chunks
fedtext features  # compute sentiment/novelty/topics
fedtext train     # build dataset + run models
fedtext app       # launch Streamlit
```

---

## Recommended Sequence

```
Phase 1 (done)
    ↓
Phase 1.5 (DB consolidation)
    ↓
Phase 2 (features) ──────────────────────────────────┐
    ↓                                                 │
Phase 3 (target + dataset)                           │ can overlap
    ↓                                                 │
Phase 4 (baselines) ─────────────────────────────────┘
    ↓
Phase 5 (ML)
    ↓
Phase 6 (RAG)
    ↓
Phase 7 (Streamlit demo) + CLI polish
```
