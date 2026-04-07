# FOMC Forecasting

Predicting inflation expectations from Federal Reserve communications using NLP features and walk-forward machine learning models.

A portfolio project converting original research notebooks into a production-grade pipeline.

---

## What it does

1. **Ingests** Fed speeches (1996-present) and FOMC policy documents (statements, minutes)
2. **Extracts text features** - hawkish/dovish sentiment via ZettaQuant central bank classifiers, meeting-over-meeting novelty (TF-IDF cosine distance), and topic diagnostics
3. **Builds and versions** inflation expectation targets from FRED (raw + transformed local artifacts)
4. **Trains** walk-forward models - baseline vs exogenous SARIMAX -> XGBoost - with no lookahead leakage
5. **Serves** a Streamlit demo with pipeline status, feature explorer, model results, and a RAG chat interface grounded in source documents

---

## End deliverables

| Deliverable | Description |
| --- | --- |
| `notebooks/` | End-to-end pipeline narrative (EDA -> features -> models) |
| Streamlit app | 4-page interactive demo (status, features, models, RAG chat) |

---

## Architecture

```text
federalreserve.gov
    |
    v
Discovery -> Fetch -> Parse -> Chunk
    |
    v
data/catalog/fedtext.db          <- unified SQLite (speeches + documents + chunks)
    |
    |-- text features             -> data/features/doc_level/features.parquet
    |      sentiment (ZettaQuant API: relevancy + stance)
    |      novelty   (TF-IDF cosine distance)
    |      topics    (diagnostic only)
    |
    |-- FRED targets              -> data/targets/t5yie_raw.parquet
    |                           -> data/targets/t5yie_diff1.parquet
    |                           -> data/targets/manifests/*.json + dataset_registry.sqlite3
    |
    |-- model dataset (Phase 3)  -> data/targets/model_dataset_t5yie.parquet
    |                           -> data/splits/time_splits.json
    |                           -> data/targets/manifests/model-dataset-t5yie-*.json
    |                           -> data/targets/model_dataset_registry.sqlite3
    |
    |-- model runs (Phase 4)     -> data/models/baselines/t5yie/<run_version>/*
    |                           -> data/models/baselines/manifests/<run_version>.json
    |                           -> data/models/baselines/run_registry.sqlite3
    |
    `-- embeddings (sqlite-vec)   <- RAG retrieval
            |
            v
        Streamlit app / Jupyter notebooks
```

---

## Feature engineering

### Sentiment - ZettaQuant API (refactored)

The sentiment stack was refactored from a local Hugging Face model path to ZettaQuant central bank classifiers served through the ZettaQuant API.

**Models used**
- `cb_inflation_relevancy_label` (Central Bank Inflation Relevancy Classifier)
- `cb_stance_label` (Central Bank Stance Classifier)

**Per-document flow (sentence-level)**
1. Normalize and split text into sentences.
2. Call ZettaQuant relevancy classifier in batches.
3. Keep only relevant sentences.
4. Call ZettaQuant stance classifier on relevant sentences.
5. Aggregate score as `(n_hawkish - n_dovish) / n_target_sentences`, where stance label `Irrelevant` is excluded from `n_target_sentences`.

**Why this refactor**
- Uses newer central-bank-specific models managed by ZettaQuant.
- Removes local model download/auth friction from the feature pipeline.
- Improves maintainability with a single API inference client and model-id routing.

**Why this is easier to extend with ZettaQuant Studio models**
- The sentiment module now routes inference by model ID through one shared API client.
- Adding a new Studio model is a narrow change: define model ID + label mapping + aggregation logic, without changing pipeline orchestration.

**Runtime configuration**
- Required: `ZQ_API_KEY`
- Optional: `ZQ_BATCH_SIZE` (default `10`), `ZQ_MAX_REQ_PER_MIN` (default `10`), `ZQ_TIMEOUT_SECONDS` (default `30`)
- Optional: `ZQ_MAX_RETRIES` (default `5`)

### Feature Extraction Reliability and Cost Controls

The feature pipeline is designed to be interruption-safe and API-cost-aware for long runs.

Mechanisms:
- **Document checkpoint (SQLite):** stores per-document feature rows and resumes by default on rerun.
- **Sentence inference cache (SQLite):** caches model labels by `(model_id, sentence_hash, sentence_text)` so repeated runs do not repay identical ZettaQuant calls.
- **Atomic parquet finalize:** writes a temporary parquet and atomically replaces `features.parquet` only after a successful full materialization.
- **Dataset manifest + registry:** each successful feature build can write a JSON manifest under `data/features/doc_level/manifests/` and upsert a row into `data/features/doc_level/dataset_registry.sqlite3` with run metadata (dataset version, git SHA, input/output hashes, preprocessing versions).
- **Transient retry policy:** exponential backoff for `429`/`5xx`/timeouts (default max retries: `5`).

CLI controls:
- `--checkpoint-every` (default `25`)
- `--resume` / `--no-resume` (default resume enabled)
- `--reset-checkpoint` (recompute selected source types from scratch)
- `--max-retries` (overrides env default)
- `--dataset-version` (optional explicit dataset tag; otherwise auto-generated)
- `--cleaning-version` / `--sentence-split-version` (semantic versions stored in manifest/registry)
- `--manifest` / `--no-manifest` (enable/disable manifest+registry writes; enabled by default)

Comparison helper:
- `python scripts/compare_feature_versions.py <manifest_a.json> <manifest_b.json> --out reports/data_diff.md`

Tradeoffs (and why these choices were made):
- SQLite checkpointing adds a local state file but gives robust idempotent resume.
- Sentence-level cache adds storage growth over time but greatly reduces repeated inference cost.
- Atomic finalize briefly uses extra disk space but avoids partial/corrupt final parquet outputs.

### Novelty - TF-IDF cosine distance

**Why TF-IDF, not embeddings:** We want lexical novelty - new vocabulary and new policy phrases - not semantic similarity. Embedding-based cosine similarity would score a paraphrased statement as nearly identical to the original even when new terminology was introduced. That vocabulary shift is exactly the signal we want to capture.

**Computed per source type:** Each statement is compared to the previous statement; each speech to the previous speech. Cross-type comparison (statement vs. speech) would be noise.

### Topics - intentionally omitted from final model

The research notebook implemented both rule-based (keyword counting) and zero-shot (BART-large-MNLI) topic classification. Both approaches produced poor class separation across meetings, and topic features were excluded from the final predictive model.

---

## Target + Model Dataset Pipelines (Phase 3)

### Target series pipeline (FRED fetch + transform)

The target pipeline fetches FRED series and materializes canonical raw + transformed outputs with run metadata.

Artifacts:
- `data/targets/t5yie_raw.parquet` (columns: `date`, `t5yie`)
- `data/targets/t5yie_diff1.parquet` (columns: `date`, `t5yie_diff1`)
- `data/targets/manifests/<dataset_version>.json`
- `data/targets/dataset_registry.sqlite3`

Version metadata fields:
- `dataset_version`, `created_at_utc`, `git_sha`, `series_id`, `transform_id`
- `raw_output_path`, `raw_output_sha256`, `raw_output_rows`
- `transformed_output_path`, `transformed_output_sha256`, `transformed_output_rows`
- `fetch_start`, `fetch_end`

Runtime configuration:
- Required: `FRED_API_KEY`

CLI:
- `python -m fedtext.targets.pipeline --series-id T5YIE --transform diff1`
- Optional window: `--start YYYY-MM-DD --end YYYY-MM-DD`
- Optional version tag: `--dataset-version <tag>`
- Metadata writes: `--manifest` / `--no-manifest` (default enabled)

### Model dataset builder pipeline (monthly no-lookahead)

The model dataset builder implements approved notebook decisions for `t5yie_diff1`:
- monthly target aggregation (`target_month`)
- monthly feature aggregation (`feature_month`)
- strict no-lookahead alignment `feature_month_used = target_month - 1`
- fixed time splits:
  - train: `date <= 2016-12-31`
  - val: `2017-01-01 <= date <= 2020-12-31`
  - test: `date >= 2021-01-01`

Artifacts:
- `data/targets/model_dataset_t5yie.parquet`
- `data/splits/time_splits.json`
- `data/targets/manifests/model-dataset-t5yie-<dataset_version>.json`
- `data/targets/model_dataset_registry.sqlite3`

Builder metadata fields include:
- `dataset_version`, `created_at_utc`, `git_sha`
- input paths + hashes + row counts (`features`, `target`)
- output dataset/split/summary paths + hashes + row counts
- split boundaries and selected feature columns

CLI:
- `python -m datasets.build_dataset.builder`
- Optional outputs:
  - `--output-dataset-path <path>`
  - `--split-output-path <path>`
  - `--summary-output-path <path>`
- Optional versioning:
  - `--dataset-version <tag>`
  - `--manifest` / `--no-manifest` (default enabled)
  - `--manifest-out-dir <dir>`
  - `--manifest-registry-path <path>`

Notebook EDA DB helper:
- `python scripts/build_targets_eda_db.py`
- Builds one SQLite DB combining:
  - `features_doc_level` (from `data/features/doc_level/features.parquet`)
  - `target_raw` and `target_transformed` (from `data/targets/*.parquet`)
  - `build_metadata`
- If target parquet artifacts are missing, the script auto-fetches them via `fedtext.targets.pipeline` by default.

### Phase 4 SARIMAX benchmark pipeline (baseline vs exogenous)

The Phase 4 benchmark productionizes `notebooks/t5yie_phase4_baseline_vs_exog_benchmark.ipynb` with fixed model variants and expanding-window one-step evaluation.

Inputs:
- `data/targets/model_dataset_t5yie.parquet`
- `data/splits/time_splits.json`

Command:
- `python -m models.baselines.sarimax`

Default model setup:
- order `(1, 0, 0)`, trend `"c"`, min train observations `36`
- variants:
  - `baseline_univariate` (no exogenous features)
  - `exog_minimal_counts` (`hawkish_score`, `novelty`, `doc_count`)
  - `exog_share_variant` (`hawkish_score`, `novelty`, `hawkish_share`, `dovish_share`)

Versioned run artifacts:
- `data/models/baselines/t5yie/<run_version>/predictions.parquet`
- `data/models/baselines/t5yie/<run_version>/results_table.json`
- `data/models/baselines/t5yie/<run_version>/paired_comparison.json`
- `data/models/baselines/t5yie/<run_version>/run_summary.json`
- `data/models/baselines/t5yie/<run_version>/run_config.json`

Run metadata tracking:
- `data/models/baselines/manifests/<run_version>.json`
- `data/models/baselines/run_registry.sqlite3`

CLI options:
- `--run-version <tag>`
- `--model-order p,d,q`
- `--model-trend <trend>`
- `--min-train-obs <int>`
- `--meaningful-threshold-pct <float>`
- `--manifest` / `--no-manifest`

---

## Current status

| Phase | Description | Status |
| ----- | ----------- | ------ |
| 1 | Ingest hardening (versioned migrations, validators, YAML config) | done |
| 1.5 | DB consolidation (`speeches.db` + `catalog.sqlite` -> `fedtext.db`) | done |
| 2 | Feature engineering (ZettaQuant sentiment, novelty, topics) | done |
| 3 | Target variable + dataset builder (FRED) | in progress (T5YIE done; additional target stream pending) |
| 4 | Baseline models (SARIMAX benchmark pipeline) | done (production runner + tracking + tests) |
| 5 | ML models (XGBoost) | - |
| 6 | RAG layer (sentence-transformers + sqlite-vec) | - |
| 7 | Streamlit demo app | - |

**Data as of Phase 1.5:** 1,932 speeches, 534 FOMC documents, unified in `fedtext.db`.

**Phase 2 backlog (non-blocking):** sentence-split persistence for auditability/reuse remains an optional reliability extension and does not reopen Phase 2 completion.

---

## Quickstart

```bash
# 1. Install (editable)
pip install -e .

# 2. Ingest speeches (1996-present)
python -m fedtext.ingest.speeches.pipeline

# 3. Ingest FOMC documents (statements + minutes)
python -m fedtext.ingest.documents.pipeline --categories St Mn

# 4. Validate data quality
python -m fedtext.ingest.validators.completeness

# 5. Configure ZettaQuant API key (required for sentiment features)
# PowerShell:
#   $env:ZQ_API_KEY="your_key_here"
# Bash:
#   export ZQ_API_KEY="your_key_here"

# 6. Build text features
python -m fedtext.text.features.pipeline

# 7. Build and version FRED target series (raw + diff1)
# PowerShell:
#   $env:FRED_API_KEY="your_key_here"
# Bash:
#   export FRED_API_KEY="your_key_here"
python -m fedtext.targets.pipeline --series-id T5YIE --transform diff1

# 8. Build SQLite DB for TS + text-feature EDA (notebook-ready)
python scripts/build_targets_eda_db.py

# 9. Build monthly model dataset + fixed time splits (versioned)
python -m datasets.build_dataset.builder

# 10. Run Phase 4 SARIMAX baseline vs exogenous benchmark (versioned run artifacts)
python -m models.baselines.sarimax
```

For faster testing, add `--limit N` to ingest/feature commands.

---

## Data sources

| Source | Content | URL |
| ------ | ------- | --- |
| Federal Reserve | Governor/President speeches | https://www.federalreserve.gov/newsevents/speech.htm |
| Federal Reserve | FOMC statements/minutes | https://www.federalreserve.gov/monetarypolicy.htm |
| FRED | Michigan inflation expectations (`MICH`) | https://fred.stlouisfed.org |
| FRED | 5yr breakeven inflation (`T5YIE`) | https://fred.stlouisfed.org |
| ZettaQuant | Central bank model inference API | https://api.zettaquant.ai |

---

## Project layout

```text
src/fedtext/
|-- common/          # db connection, path constants
|-- ingest/          # Fed text ingest only (speeches/documents)
|   |-- speeches/    # discovery + fetch
|   |-- documents/   # discovery + fetch + parse
|   |-- storage/     # versioned SQL migrations
|   `-- validators/  # data quality checks
|-- targets/         # FRED fetch + transforms + target storage/versioning
`-- text/            # cleaning, chunker, features

configs/             # sources.yaml - URLs, rate limits, categories
data/catalog/        # fedtext.db (SQLite)
data/targets/        # target parquets + model dataset parquet + manifests + registries
data/splits/         # fixed split artifact json for modeling
data/models/         # phase4 model run artifacts + manifests + registry
models/              # phase4 benchmark/evaluation/tracking modules
scripts/             # one-off utilities
notebooks/           # research + portfolio notebooks
tests/models/        # phase4 benchmark/evaluation/tracking tests
docs/                # roadmap
```

---

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the detailed phase breakdown.

---

## References

- ZettaQuant API, model inference endpoint and central bank classifiers (`cb_inflation_relevancy_label`, `cb_stance_label`): https://api.zettaquant.ai
- ZettaQuant platform and Studio (for managed model lifecycle and deployment): https://api.zettaquant.ai/v1/models
- Shah, A., Papadopoulos, S., & Guo, T. (2023). *Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis*. Proceedings of ACL 2023. (historical context for earlier model choices)

---

## Data usage note

All data is scraped from public Federal Reserve websites (`federalreserve.gov`). US government works are not subject to copyright (17 U.S.C. Section 105). Scraping is rate-limited and non-commercial.
